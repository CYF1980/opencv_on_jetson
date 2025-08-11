#!/usr/bin/env bash
# Download Blaze Hand ONNX models + anchors for our Jetson repo
# Target dir = this script's directory (models/hands)
# Files:
#   - hand_detector.onnx (input 192x192)
#   - hand_landmarks_detector.onnx (input 224x224)
#   - anchors.csv (for detector decoding)
# Source: https://huggingface.co/unity/inference-engine-blaze-hand

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BASE="https://huggingface.co/unity/inference-engine-blaze-hand/resolve/main"
DET_URL="$BASE/models/hand_detector.onnx"
LMK_URL="$BASE/models/hand_landmarks_detector.onnx"
ANCH_URL="$BASE/data/anchors.csv"

# SHA256 from Hugging Face LFS details (helps catch corrupted downloads)
DET_SHA="c620c7c17de68a6568d0ce9e1ee1335531b7c7a6567dfd1150856e20921cbba9"
LMK_SHA="e18a95135b40c732ea53d2dd6af66cbec6d3f8bf0296bb529a4c4be0e8349ec1"
# anchors.csv SHA not listed on the page; we skip strict verification for it.

calc_sha256() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    echo ""  # no hasher available
  fi
}

fetch() {
  local url="$1" out="$2" expected_sha="${3:-}"
  local tmp="${out}.part"

  # Skip if already correct
  if [[ -f "$out" && -n "$expected_sha" ]]; then
    local have
    have="$(calc_sha256 "$out")"
    if [[ "$have" == "$expected_sha" ]]; then
      echo "✔ ${out##*/} already present (SHA256 OK)"
      return 0
    else
      echo "… ${out##*/} exists but hash mismatched, re-downloading"
    fi
  fi

  echo "↓ Downloading ${out##*/}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$tmp" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$tmp" "$url"
  else
    echo "Error: need curl or wget installed" >&2
    exit 1
  fi

  # Verify if we know the expected SHA
  if [[ -n "$expected_sha" ]]; then
    local got
    got="$(calc_sha256 "$tmp")"
    if [[ -z "$got" ]]; then
      echo "(warn) No SHA256 tool found to verify $out"
    elif [[ "$got" != "$expected_sha" ]]; then
      echo "✗ SHA256 mismatch for ${out##*/}: got $got, expected $expected_sha" >&2
      rm -f "$tmp"
      exit 1
    fi
  fi

  mv -f "$tmp" "$out"
  echo "✔ Saved ${out##*/} -> $out"
}

mkdir -p "$HERE"

fetch "$DET_URL" "$HERE/hand_detector.onnx" "$DET_SHA"
fetch "$LMK_URL" "$HERE/hand_landmarks_detector.onnx" "$LMK_SHA"
fetch "$ANCH_URL" "$HERE/anchors.csv" ""

echo
echo "All files ready in: $HERE"
echo "  - $(ls -lh "$HERE" | awk 'NR>1{print $9, $5}' | column -t)"