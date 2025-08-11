import cv2
import time
import argparse
import os

# ---------- Camera helpers (USB / CSI) ----------
def gstreamer_pipeline(
    capture_width=1280, capture_height=720,
    display_width=1280, display_height=720,
    framerate=30, flip_method=0,
):
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
        return cv2.VideoCapture(
            gstreamer_pipeline(width, height, width, height, fps, 0),
            cv2.CAP_GSTREAMER
        )
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

# ---------- Pose pairs for COCO / MPI ----------
POSE_PAIRS = {
    "COCO": [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
             (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)],
    "MPI":  [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,14),(14,8),
             (8,9),(9,10),(14,11),(11,12),(12,13)]
}

N_PARTS = {"COCO": 18, "MPI": 16}   # OpenCV sample定義（MPI含背景通道）

# ---------- Load network ----------
def load_openpose(model_root, dataset):
    if dataset.upper() == "COCO":
        proto = os.path.join(model_root, "pose/coco/pose_deploy_linevec.prototxt")
        weights = os.path.join(model_root, "pose/coco/pose_iter_440000.caffemodel")
    else:
        proto = os.path.join(model_root, "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        weights = os.path.join(model_root, "pose/mpi/pose_iter_160000.caffemodel")

    if not (os.path.isfile(proto) and os.path.isfile(weights)):
        raise FileNotFoundError(f"找不到模型：\n  {proto}\n  {weights}\n請先下載。")

    net = cv2.dnn.readNet(weights, proto)
    # try enable CUDA
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # only 2~3 FPS
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("[INFO] 使用 CUDA")
        else:
            raise RuntimeError("CUDA 不可用")
    except Exception as e:
        print(f"[WARN] CUDA 無法啟用（{e}），改用 CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", type=str, default=os.path.expanduser("./models/openpose"),
                    help="包含 pose/coco 和 pose/mpi 的根目錄")
    ap.add_argument("--dataset", type=str, default="COCO", choices=["COCO","MPI"],
                    help="選擇 COCO(18點) 或 MPI(15點)")
    ap.add_argument("--width", type=int, default=368, help="網路輸入寬度（常用 368）")
    ap.add_argument("--height", type=int, default=368, help="網路輸入高度（常用 368）")
    ap.add_argument("--thr", type=float, default=0.1, help="關節點信心閾值")
    ap.add_argument("--csi", action="store_true", help="使用 CSI 相機（預設 USB）")
    ap.add_argument("--device", type=int, default=0, help="USB 攝影機索引")
    ap.add_argument("--cam_w", type=int, default=1280)
    ap.add_argument("--cam_h", type=int, default=720)
    ap.add_argument("--cam_fps", type=int, default=30)
    args = ap.parse_args()

    cap = open_capture(args.csi, args.cam_w, args.cam_h, args.cam_fps, args.device)
    if not cap.isOpened():
        print("無法開啟攝影機"); return

    net = load_openpose(args.model_root, args.dataset)
    pairs = POSE_PAIRS[args.dataset]
    n_parts = N_PARTS[args.dataset]

    prev_t = time.time()
    fps_ema = 0.0
    alpha = 0.9

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            blob = cv2.dnn.blobFromImage(frame, 1.0/255,
                                         (args.width, args.height),
                                         (0,0,0), swapRB=False, crop=False)
            net.setInput(blob)
            out = net.forward()  # shape: [1, n_parts, H, W]
            H, W = out.shape[2], out.shape[3]

            points = []
            for n in range(n_parts):
                heatmap = out[0, n, :, :]
                _, conf, _, pt = cv2.minMaxLoc(heatmap)
                x = int(pt[0] * frame.shape[1] / W)
                y = int(pt[1] * frame.shape[0] / H)
                points.append((x, y) if conf > args.thr else None)

            # draw
            for a,b in pairs:
                if a < len(points) and b < len(points):
                    pa, pb = points[a], points[b]
                    if pa and pb:
                        cv2.line(frame, pa, pb, (0,200,0), 2, cv2.LINE_AA)
                        cv2.circle(frame, pa, 3, (0,0,200), -1, cv2.LINE_AA)
                        cv2.circle(frame, pb, 3, (0,0,200), -1, cv2.LINE_AA)

            # fps
            now = time.time()
            inst = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            fps_ema = inst if fps_ema == 0 else alpha*fps_ema + (1-alpha)*inst
            draw_fps(frame, fps_ema)

            cv2.imshow(f"OpenPose ({args.dataset})", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
