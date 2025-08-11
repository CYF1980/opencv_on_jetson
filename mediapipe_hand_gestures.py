import cv2
import time
import argparse
import os
import math

import mediapipe as mp

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

# ---------- Drawing & FPS ----------
def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

# ---------- Geometry helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_thumb_extended(landmarks, handedness_label):
    # 透過拇指尖與 IP/MCP 的相對 x 位置 + 與掌心距離，判斷是否伸直
    # 右手：拇指尖(4) x < 拇指IP(3) x < 拇指MCP(2) x 代表向右張開（鏡頭未鏡像）
    # 左手：反之
    tip = landmarks[4]
    ip  = landmarks[3]
    mcp = landmarks[2]
    if handedness_label == "Right":
        return tip.x < ip.x < mcp.x
    else:
        return tip.x > ip.x > mcp.x

def is_finger_extended(landmarks, tip_id):
    # 對食指~小指：tip.y < pip.y 視為伸直（影像座標 y 向下）
    pip_id = tip_id - 2
    return landmarks[tip_id].y < landmarks[pip_id].y

def count_fingers(landmarks, handedness_label):
    # 計算伸出的手指數：拇指 + (食/中/無名/小指)
    count = 0
    if is_thumb_extended(landmarks, handedness_label):
        count += 1
    for tip in [8, 12, 16, 20]:
        if is_finger_extended(landmarks, tip):
            count += 1
    return count

def is_thumbs_up(landmarks, handedness_label):
    # 拇指伸直、其他四指彎曲、且拇指方向大致上「上下」而非水平
    thumb_ok = is_thumb_extended(landmarks, handedness_label)
    others_folded = all(not is_finger_extended(landmarks, t) for t in [8,12,16,20])

    # 拇指尖與指根的 y 差距要明顯
    tip = landmarks[4]
    mcp = landmarks[2]
    vertical = abs(tip.y - mcp.y) > abs(tip.x - mcp.x)  # 近似垂直
    return thumb_ok and others_folded and vertical

def is_peace(landmarks):
    # 食指、中指伸直，無名、小指彎曲
    return (is_finger_extended(landmarks, 8) and
            is_finger_extended(landmarks, 12) and
            (not is_finger_extended(landmarks, 16)) and
            (not is_finger_extended(landmarks, 20)))

def is_ok_gesture(landmarks):
    # OK：拇指尖(4) 與 食指尖(8) 距離很近，且中、無名、小指大多伸直或自然
    t = landmarks[4]
    i = landmarks[8]
    # 距離用「相對手掌大小」正規化：以手掌（0 wrist 到中指 MCP(9)）距離為尺度
    palm_scale = dist(landmarks[0], landmarks[9]) + 1e-6
    close = dist(t, i) / palm_scale < 0.25
    return close

def classify_gesture(landmarks, handedness_label):
    # 回傳 (gesture_name, extra_text)
    if is_thumbs_up(landmarks, handedness_label):
        return ("THUMBS_UP", "👍")
    if is_ok_gesture(landmarks):
        return ("OK", "👌")
    if is_peace(landmarks):
        return ("PEACE", "✌️")
    cnt = count_fingers(landmarks, handedness_label)
    return (f"{cnt}_FINGERS", f"{cnt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi", action="store_true", help="使用 CSI 相機（預設 USB）")
    ap.add_argument("--device", type=int, default=0, help="USB 攝影機索引")
    ap.add_argument("--cam_w", type=int, default=1280)
    ap.add_argument("--cam_h", type=int, default=720)
    ap.add_argument("--cam_fps", type=int, default=30)
    ap.add_argument("--max_hands", type=int, default=2)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.5)
    ap.add_argument("--mirror", action="store_true", help="鏡像顯示（像自拍）")
    args = ap.parse_args()

    cap = open_capture(args.csi, args.cam_w, args.cam_h, args.cam_fps, args.device)
    if not cap.isOpened():
        print("無法開啟攝影機"); return

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    prev_t = time.time()
    fps_ema = 0.0
    alpha = 0.9

    # 注意：MediaPipe Hands 主要跑 CPU，但在 Jetson 640x480 也能拿到不錯 FPS
    with mp_hands.Hands(
        max_num_hands=args.max_hands,
        model_complexity=1,            # 0 較快 / 1 平衡 / 2 較準
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track
    ) as hands:

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # MediaPipe 要求不可寫以加速
                rgb.flags.writeable = False
                res = hands.process(rgb)

                # 繪製結果
                if res.multi_hand_landmarks:
                    for hand_lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                        # 畫骨架
                        mp_draw.draw_landmarks(
                            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )

                        # 手勢判斷
                        label = handedness.classification[0].label  # "Left"/"Right"
                        g_name, g_text = classify_gesture(hand_lms.landmark, label)

                        # 顯示文字
                        x0 = int(hand_lms.landmark[0].x * frame.shape[1])
                        y0 = int(hand_lms.landmark[0].y * frame.shape[0]) - 10
                        cv2.putText(frame, f"{label} {g_name} {g_text}",
                                    (x0, max(30, y0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # FPS
                now = time.time()
                inst = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                fps_ema = inst if fps_ema == 0 else alpha*fps_ema + (1-alpha)*inst
                draw_fps(frame, fps_ema)

                cv2.imshow("MediaPipe Hands - Gestures", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
