#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname "$0")"

# Hand landmark (ONNX) 來源：Keijiro 的 HandLandmarkBarracuda 專案（已轉好 ONNX）
# ONNX 檔位於其套件目錄下。這裡提供直接抓取的 raw URL。
# 若 GitHub raw 風控擋下，可手動到 repo 下載：Packages/jp.keijiro.mediapipe.handlandmark/ONNX/hand_landmark.onnx
curl -L -o hand_landmark.onnx \
  https://raw.githubusercontent.com/keijiro/HandLandmarkBarracuda/main/Packages/jp.keijiro.mediapipe.handlandmark/ONNX/hand_landmark.onnx

# BlazePalm / palm_detection_lite.onnx
# 來源：Unity 的 Hugging Face 「inference-engine-blaze-hand」
# 之後要做全 DNN 版會用到（需要 anchor decode）
curl -L -o palm_detection_lite.onnx \
  https://huggingface.co/unity/inference-engine-blaze-hand/resolve/1efb68abeaf82dab918712f82fe389afd3302215/palm_detection_lite.onnx
echo "Done."
