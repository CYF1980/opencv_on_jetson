# OpenCV on Jetson (CUDA Accelerated)

## Reference
https://huggingface.co/camenduru/

## Usage:
### USB Camera
python3 face_fps_cuda.py --width 640 --height 480 --fps 30
python3 openpose_realtime_cuda.py --model_root ./models/openpose --dataset COCO --cam_w 640 --cam_h 480 --cam_fps 30
python3 openpose_realtime_cuda.py --model_root ./models/openpose --dataset MPI --cam_w 640 --cam_h 480 --cam_fps 30


### CSI Camera
python3 face_fps_cuda.py --csi --width 1280 --height 720 --fps 30




















