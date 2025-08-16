import cv2
import mediapipe as mp

# === 修改這裡：選擇你的相機格式並填合適的寬高/幀率 ===
DEVICE = "/dev/video0"
WIDTH, HEIGHT, FPS = 640, 480, 30

# 如果攝影機是 MJPEG：
gst = (f"v4l2src device={DEVICE} ! "
       f"image/jpeg,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
       "jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")

# 如果是 YUY2，改用這個（把上面那段註解掉）：
# gst = (f"v4l2src device={DEVICE} ! "
#        f"video/x-raw,format=YUY2,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
#        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")

# 如果是 H.264，改用這個（Jetson 可用 nvv4l2decoder / avdec_h264 任選）：
# gst = (f"v4l2src device={DEVICE} ! "
#        f"video/x-h264,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
#        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera via GStreamer")

mp_face_mesh = mp.solutions.face_mesh
# 468 點：refine_landmarks=False；設 True 會變 478（含虹膜）。 
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=False,         # ★ 你要 468 點就 False
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        h, w = frame.shape[:2]
        for lmks in res.multi_face_landmarks:
            for lm in lmks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("FaceMesh 468", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
