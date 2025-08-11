import os
import cv2
import csv
import time
import math
import argparse
import numpy as np

# Optional ONNX Runtime (faster & more flexible). Falls back to OpenCV DNN if unavailable.
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou_xywh(a, b):
    """IoU for boxes in [cx, cy, w, h] normalized [0..1] image space."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2

    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter_area + 1e-9
    return inter_area / union


def nms_xywh(boxes, scores, thresh=0.3, limit=2):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < limit:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou_xywh(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < thresh]
    return keep


def load_anchors_csv(path):
    """Load anchors.csv with columns: x_center,y_center,w,h (normalized to 0..1)."""
    anchors = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Try to infer columns
        colmap = {name: idx for idx, name in enumerate(header)}
        for row in reader:
            if not row:
                continue
            x = float(row[colmap.get("x_center", 0)])
            y = float(row[colmap.get("y_center", 1)])
            w = float(row[colmap.get("w", 2)])
            h = float(row[colmap.get("h", 3)])
            anchors.append([x, y, w, h])
    return np.asarray(anchors, dtype=np.float32)


# ---------------------------------------------------------------
# BlazeHand (detector + landmark) ONNX wrappers
# ---------------------------------------------------------------

class ONNXWrapper:
    def __init__(self, model_path, input_shape, fp16=True):
        self.model_path = model_path
        self.input_shape = input_shape  # (W, H)

        self.use_ort = False
        self.ort_sess = None
        self.cv_net = None

        if ORT_AVAILABLE:
            try:
                avail = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
                self.ort_sess = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.ort_sess.get_inputs()[0].name
                self.use_ort = True
            except Exception:
                self.ort_sess = None
                self.use_ort = False

        if not self.use_ort:
            self.cv_net = cv2.dnn.readNet(model_path)
            # Try CUDA backend if available
            try:
                self.cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16 if fp16 else cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                # CPU fallback
                self.cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, img_rgb):
        # img_rgb: HxWx3, uint8
        W, H = self.input_shape
        blob = cv2.dnn.blobFromImage(img_rgb, scalefactor=1/255.0, size=(W, H), swapRB=False, crop=False)
        if self.use_ort:
            inp = np.transpose(blob[0], (1, 2, 0))  # to HWC
            inp = np.expand_dims(inp, 0).astype(np.float32)
            outs = self.ort_sess.run(None, {self.input_name: inp})
            return outs
        else:
            self.cv_net.setInput(blob)
            outs = self.cv_net.forward(self.cv_net.getUnconnectedOutLayersNames() or [])
            # OpenCV may return a single ndarray instead of list depending on model
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            return outs


class BlazeHandDetector:
    """
    BlazePalm-like palm detector wrapper.

    Assumptions (work well with the Unity/HF models):
      * Input: 128x128 RGB, float32 in [0,1]
      * Outputs: regressors [1, N, 18], scores [1, N, 1]
      * Anchors: anchors.csv with N rows (x_center,y_center,w,h)
      * Decode scales (x/y/w/h): 128 (if results look off, try 256)
    """
    def __init__(self, onnx_path, anchors_csv, score_thresh=0.75, nms_thresh=0.3, max_hands=2):
        self.net = ONNXWrapper(onnx_path, input_shape=(192, 192))
        self.anchors = load_anchors_csv(anchors_csv)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.max_hands = int(max_hands)
        self.x_scale = 192.0
        self.y_scale = 192.0
        self.w_scale = 192.0
        self.h_scale = 192.0

    def _split_outputs(self, outs):
        # Try to pick (regressors, scores) by shape
        regs = None
        conf = None
        for o in outs:
            s = o.shape
            if len(s) == 3 and s[-1] == 18:
                regs = o
            elif len(s) == 3 and s[-1] in (1,):
                conf = o
        # Sometimes order is [scores, regs]
        if regs is None or conf is None:
            # Fallback: assume outs[0]=regs, outs[1]=scores
            if len(outs) >= 2:
                if outs[0].shape[-1] == 18:
                    regs, conf = outs[0], outs[1]
                else:
                    regs, conf = outs[1], outs[0]
            else:
                raise RuntimeError("Unexpected detector outputs: need [regs, scores].")
        return regs, conf

    def detect(self, img_rgb):
        H, W, _ = img_rgb.shape
        outs = self.net.infer(cv2.resize(img_rgb, (192, 192), interpolation=cv2.INTER_LINEAR))
        regs, conf = self._split_outputs(outs)
        regs = regs.reshape(-1, 18)
        scores = sigmoid(conf.reshape(-1))

        # Filter by score
        keep = np.where(scores >= self.score_thresh)[0]
        if keep.size == 0:
            return []

        regs = regs[keep]
        scores = scores[keep]
        anchors = self.anchors[keep]

        # Decode to boxes (cx,cy,w,h) and 7 keypoints (unused here, but kept for ROI improvement)
        cx = regs[:, 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        cy = regs[:, 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
        w  = regs[:, 2] / self.w_scale * anchors[:, 2]
        h  = regs[:, 3] / self.h_scale * anchors[:, 3]
        boxes = np.stack([cx, cy, w, h], axis=1)

        # NMS (in normalized coordinates)
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        keep_idx = nms_xywh(boxes, scores, thresh=self.nms_thresh, limit=self.max_hands)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        # Convert to absolute pixel XYXY with some padding (square)
        dets = []
        for (cx, cy, w, h), sc in zip(boxes, scores):
            # Add a little padding and make square ROI
            side = max(w, h) * 1.6
            x1 = int((cx - side / 2) * W)
            y1 = int((cy - side / 2) * H)
            x2 = int((cx + side / 2) * W)
            y2 = int((cy + side / 2) * H)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
            dets.append(((x1, y1, x2, y2), float(sc)))
        return dets


class BlazeHandLandmarks:
    """
    Landmark head wrapper. Assumptions:
      * Input: 224x224 RGB, float32 in [0,1]
      * Output[0]: 1x(63 or 84) -> first 63 are 21*(x,y,z). x,y in [0..1] relative to the cropped ROI
    """
    def __init__(self, onnx_path):
        self.net = ONNXWrapper(onnx_path, input_shape=(224, 224))

    def infer_landmarks(self, crop_rgb):
        outs = self.net.infer(cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR))
        # Take first output and first 63 values
        arr = outs[0].reshape(-1)
        if arr.shape[0] < 63:
            raise RuntimeError("Unexpected landmarks output size: expected >=63 values")
        xyz = arr[:63].reshape(21, 3)
        return xyz


# ---------------------------------------------------------------
# Gesture helpers (reuse from your MediaPipe version)
# ---------------------------------------------------------------

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_thumb_extended(pts, handedness_label="Right"):
    tip = pts[4]; ip = pts[3]; mcp = pts[2]
    if handedness_label == "Right":
        return tip[0] < ip[0] < mcp[0]
    else:
        return tip[0] > ip[0] > mcp[0]


def is_finger_extended(pts, tip_id):
    pip_id = tip_id - 2
    return pts[tip_id][1] < pts[pip_id][1]


def count_fingers(pts, handedness_label):
    cnt = 0
    if is_thumb_extended(pts, handedness_label):
        cnt += 1
    for t in [8, 12, 16, 20]:
        if is_finger_extended(pts, t):
            cnt += 1
    return cnt


def is_thumbs_up(pts, handedness_label="Right"):
    thumb_ok = is_thumb_extended(pts, handedness_label)
    others_folded = all(not is_finger_extended(pts, t) for t in [8, 12, 16, 20])
    tip = pts[4]; mcp = pts[2]
    vertical = abs(tip[1] - mcp[1]) > abs(tip[0] - mcp[0])
    return thumb_ok and others_folded and vertical


def is_peace(pts):
    return (is_finger_extended(pts, 8) and is_finger_extended(pts, 12) and
            (not is_finger_extended(pts, 16)) and (not is_finger_extended(pts, 20)))


def is_ok_gesture(pts):
    palm_scale = dist(pts[0], pts[9]) + 1e-6
    close = dist(pts[4], pts[8]) / palm_scale < 0.25
    return close


def classify_gesture(pts, handedness_label="Right"):
    if is_thumbs_up(pts, handedness_label):
        return ("THUMBS_UP", "👍")
    if is_ok_gesture(pts):
        return ("OK", "👌")
    if is_peace(pts):
        return ("PEACE", "✌️")
    cnt = count_fingers(pts, handedness_label)
    return (f"{cnt}_FINGERS", f"{cnt}")


# ---------------------------------------------------------------
# Main app
# ---------------------------------------------------------------

def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


def open_capture(use_csi, width, height, fps, device_index):
    if use_csi:
        return cv2.VideoCapture(gstreamer_pipeline(width, height, width, height, fps, 0), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi", action="store_true", help="使用 CSI 相機（預設 USB）")
    ap.add_argument("--device", type=int, default=0, help="USB 攝影機索引")
    ap.add_argument("--cam_w", type=int, default=1280)
    ap.add_argument("--cam_h", type=int, default=720)
    ap.add_argument("--cam_fps", type=int, default=30)
    ap.add_argument("--max_hands", type=int, default=2)
    ap.add_argument("--score", type=float, default=0.75, help="Palm detector 信心門檻")
    ap.add_argument("--nms", type=float, default=0.3, help="NMS IoU 門檻")
    ap.add_argument("--mirror", action="store_true", help="鏡像顯示（像自拍）")
    ap.add_argument("--models_dir", type=str, default=os.path.join(os.path.dirname(__file__), "models", "hands"))
    args = ap.parse_args()

    anchors_csv = os.path.join(args.models_dir, "anchors.csv")
    det_onnx   = os.path.join(args.models_dir, "hand_detector.onnx")
    lmk_onnx   = os.path.join(args.models_dir, "hand_landmarks_detector.onnx")

    if not (os.path.isfile(anchors_csv) and os.path.isfile(det_onnx) and os.path.isfile(lmk_onnx)):
        raise SystemExit(f"找不到模型或 anchors：{anchors_csv}, {det_onnx}, {lmk_onnx}")

    detector = BlazeHandDetector(det_onnx, anchors_csv, score_thresh=args.score, nms_thresh=args.nms, max_hands=args.max_hands)
    landmark = BlazeHandLandmarks(lmk_onnx)

    cap = open_capture(args.csi, args.cam_w, args.cam_h, args.cam_fps, args.device)
    if not cap.isOpened():
        print("無法開啟攝影機"); return

    prev_t = time.time()
    fps_ema = 0.0
    alpha = 0.9

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1) Palm detection on full image
            dets = detector.detect(rgb)

            # 2) For each ROI, run landmarks and draw
            for (x1, y1, x2, y2), sc in dets:
                roi = rgb[y1:y2, x1:x2].copy()
                if roi.size == 0:
                    continue
                xyz = landmark.infer_landmarks(roi)  # 21x3, x/y normalized to [0,1] in ROI

                # Map back to absolute image coords
                points = []
                for i in range(21):
                    px = x1 + xyz[i, 0] * max(1, x2 - x1)
                    py = y1 + xyz[i, 1] * max(1, y2 - y1)
                    points.append((float(px), float(py)))

                # Draw skeleton
                # MediaPipe hand connections (subset)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),     # thumb
                    (0,5),(5,6),(6,7),(7,8),     # index
                    (0,9),(9,10),(10,11),(11,12),# middle
                    (0,13),(13,14),(14,15),(15,16),# ring
                    (0,17),(17,18),(18,19),(19,20) # pinky
                ]
                for (i, j) in connections:
                    cv2.line(frame, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), (0,255,0), 2, cv2.LINE_AA)
                for (px, py) in points:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 200, 255), -1, cv2.LINE_AA)

                # Simple gesture classification (uses 2D only)
                # When mirrored, handedness can flip visually; you can pass "Left"/"Right" based on ROI position if needed.
                g_name, g_text = classify_gesture(points, "Right")
                cv2.putText(frame, f"{g_name} {g_text} ({sc:.2f})", (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # FPS
            now = time.time()
            inst = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            fps_ema = inst if fps_ema == 0 else alpha*fps_ema + (1-alpha)*inst
            draw_fps(frame, fps_ema)

            cv2.imshow("BlazeHand ONNX (no MediaPipe)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
