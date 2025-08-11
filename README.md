# OpenCV on Jetson (CUDA Accelerated)

## Reference
https://huggingface.co/camenduru/

## Usage:
### 開一個能看到系統套件的 venv：
python3 -m venv venv --system-site-packages
source venv/bin/activate
### 安裝 mediapipe（避免拉入 pip 的 OpenCV）：
python -m pip install --upgrade pip
pip install --no-deps mediapipe==0.10.14
# 另外把必要依賴裝起來（不含 opencv）
pip install "protobuf<4" absl-py attrs "numpy<2"

### USB Camera
python3 face_fps_cuda.py --width 640 --height 480 --fps 30
python3 openpose_realtime_cuda.py --model_root ./models/openpose --dataset COCO --cam_w 640 --cam_h 480 --cam_fps 30
python3 openpose_realtime_cuda.py --model_root ./models/openpose --dataset MPI --cam_w 640 --cam_h 480 --cam_fps 30


### CSI Camera
python3 face_fps_cuda.py --csi --width 1280 --height 720 --fps 30




















