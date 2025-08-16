import cv2
import time
import argparse
import os

# ---- Camera helpers ----
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def open_capture(use_csi, width, height, framerate, device_index):
    if use_csi:
        return cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=width, capture_height=height,
                display_width=width, display_height=height,
                framerate=framerate, flip_method=0
            ),
            cv2.CAP_GSTREAMER
        )
    else:
        cap = cv2.VideoCapture(device_index)
        # 嘗試設定解析度/幀率（USB cam 支援與否看裝置）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, framerate)
        return cap

# ---- DNN Face Detector ----
def load_face_net(model_dir):
    proto = os.path.join(model_dir, "deploy.prototxt")
    model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not (os.path.isfile(proto) and os.path.isfile(model)):
        raise FileNotFoundError(
            f"找不到模型檔，請先下載：\n  {proto}\n  {model}"
        )
    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net

def try_enable_cuda(net):
    # 檢查是否有 CUDA 環境（OpenCV 需編譯含 CUDA 與 cuDNN）
    try:
        has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        has_cuda = False

    if has_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("[INFO] DNN backend=CUDA, target=CUDA")
            return True
        except Exception as e:
            print(f"[WARN] 嘗試啟用 CUDA 失敗，改用 CPU。原因：{e}")
    else:
        print("[INFO] 未偵測到可用 CUDA 或 OpenCV 未啟用 CUDA，使用 CPU。")
    # CPU fallback
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return False

def draw_fps(frame, fps):
    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.path.expanduser("./models/face"),
                        help="模型檔所在資料夾（含 deploy.prototxt 與 res10_*.caffemodel）")
    parser.add_argument("--csi", action="store_true", help="使用 CSI 相機（預設為 USB）")
    parser.add_argument("--device", type=int, default=0, help="USB 攝影機的裝置索引（預設 0）")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30, help="期望輸入 FPS（實際依裝置）")
    parser.add_argument("--conf", type=float, default=0.5, help="臉部偵測信心值門檻 0~1")
    args = parser.parse_args()

    # 開相機
    cap = open_capture(args.csi, args.width, args.height, args.fps, args.device)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    # 載入模型
    net = load_face_net(args.model_dir)
    using_cuda = try_enable_cuda(net)

    # FPS 計算用
    prev_t = time.time()
    fps_smoothed = 0.0
    alpha = 0.9  # 指數平滑參數

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("讀取影像失敗，結束")
                break

            # DNN 前處理：300x300 BGR, 減均值（OpenCV 樣本值）
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0),  # mean BGR
                swapRB=False, crop=False
            )
            net.setInput(blob)
            detections = net.forward()

            (h, w) = frame.shape[:2]
            # 繪製偵測框
            for i in range(0, detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < args.conf:
                    continue
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"{confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

            # FPS 計算（指數平滑，讓顯示穩定）
            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_smoothed = alpha * fps_smoothed + (1 - alpha) * inst_fps if fps_smoothed > 0 else inst_fps

            draw_fps(frame, fps_smoothed)
            cv2.imshow("Face Detection (CUDA {} )".format("ON" if using_cuda else "OFF"), frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
