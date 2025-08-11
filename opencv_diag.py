#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
opencv_diag.py
一鍵檢測：OpenCV 版本、安裝路徑、CUDA 編譯與可用性。

用法：
  python3 opencv_diag.py
  python3 opencv_diag.py --json      # 以 JSON 輸出（方便記錄）
"""

import sys
import json
import platform
from textwrap import indent

def safe_bool_from_buildinfo(build_info: str, key: str):
    # 回傳 True/False/None（None 表示找不到該鍵）
    for line in build_info.splitlines():
        if line.strip().startswith(key):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("YES", "TRUE", "ON", "1"):
                return True
            if val in ("NO", "FALSE", "OFF", "0"):
                return False
            return None
    return None

def extract_from_buildinfo(build_info: str, keys):
    out = {}
    for k in keys:
        v = None
        for line in build_info.splitlines():
            if line.strip().startswith(k):
                v = line.split(":", 1)[1].strip()
                break
        out[k] = v
    return out

def main():
    info = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "opencv": {},
        "cuda": {
            "compiled_with_cuda": None,
            "compiled_with_cudnn": None,
            "runtime_device_count": 0,
            "devices": [],
            "notes": [],
        },
        "errors": [],
    }

    try:
        import cv2
    except Exception as e:
        info["errors"].append(f"匯入 OpenCV 失敗：{e!r}")
        output(info)
        return 1

    # 版本與安裝路徑
    info["opencv"]["version"] = getattr(cv2, "__version__", None)
    info["opencv"]["module_path"] = getattr(cv2, "__file__", None)

    # 取得 build information（判斷是否為 CUDA 編譯）
    build_info = ""
    try:
        build_info = cv2.getBuildInformation()
        info["opencv"]["has_build_info"] = True
    except Exception as e:
        info["opencv"]["has_build_info"] = False
        info["errors"].append(f"取得 build information 失敗：{e!r}")

    if build_info:
        # 直接解析幾個重要開關
        info["cuda"]["compiled_with_cuda"] = safe_bool_from_buildinfo(build_info, "Use CUDA")
        info["cuda"]["compiled_with_cudnn"] = safe_bool_from_buildinfo(build_info, "Use cuDNN")

        # 額外摘一些常用欄位（可幫助判斷來源/功能）
        extra = extract_from_buildinfo(build_info, [
            "NVIDIA CUDA", "NVIDIA GPU arch", "cuDNN", "CUDART", "FFmpeg", "GStreamer",
            "Parallel framework", "OpenCL", "OpenVX", "NEON", "Build type", "Install path",
        ])
        info["opencv"]["build_fields"] = extra

    # 於 runtime 檢查 CUDA 能否使用
    try:
        has_cuda_namespace = hasattr(cv2, "cuda")
        if has_cuda_namespace:
            try:
                count = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception as e:
                count = 0
                info["cuda"]["notes"].append(f"查詢 CUDA 裝置數量失敗：{e!r}")

            info["cuda"]["runtime_device_count"] = int(count)

            # 列出裝置名稱
            if count and count > 0:
                for i in range(count):
                    try:
                        name = cv2.cuda.getDeviceName(i)
                    except Exception:
                        name = f"device_{i}"
                    info["cuda"]["devices"].append({"index": i, "name": name})
        else:
            info["cuda"]["notes"].append("此 OpenCV 模組沒有 cv2.cuda 命名空間（通常代表非 CUDA 版本或平台不支援）。")
    except Exception as e:
        info["errors"].append(f"檢查 CUDA 期間發生例外：{e!r}")

    # 情境化備註（幫你判讀）
    if info["cuda"]["compiled_with_cuda"] is True and info["cuda"]["runtime_device_count"] == 0:
        info["cuda"]["notes"].append("此 OpenCV 似乎以 CUDA 編譯，但在目前環境找不到可用的 CUDA 裝置或驅動程式。")
    if info["cuda"]["compiled_with_cuda"] is False:
        info["cuda"]["notes"].append("此 OpenCV 不是以 CUDA 編譯（pip 安裝的預設情況很常見）。")

    return output(info)

def output(info: dict):
    # 若指定 --json 就輸出 JSON；否則用人類可讀格式
    if "--json" in sys.argv:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    # Human-readable
    print("=== OpenCV 診斷報告 ===")
    print(f"Python: {info['python']['implementation']} {info['python']['version']}")
    print(f"OS/Arch: {info['os']['platform']} | {info['os']['machine']} | {info['os']['processor']}")
    print()
    if info["errors"]:
        print("⚠️ 錯誤：")
        for e in info["errors"]:
            print(f"  - {e}")
        print()

    print("OpenCV：")
    print(f"  版本：{info['opencv'].get('version')}")
    print(f"  模組路徑：{info['opencv'].get('module_path')}")
    if info["opencv"].get("has_build_info"):
        print("  Build fields（節選）：")
        fields = info["opencv"].get("build_fields") or {}
        for k, v in fields.items():
            print(f"    - {k}: {v}")
    else:
        print("  （無法取得 build information）")

    print()
    print("CUDA：")
    cw = info["cuda"]["compiled_with_cuda"]
    cd = info["cuda"]["compiled_with_cudnn"]
    print(f"  以 CUDA 編譯：{cw}")
    print(f"  以 cuDNN 編譯：{cd}")
    print(f"  Runtime 可用裝置數：{info['cuda']['runtime_device_count']}")
    if info["cuda"]["devices"]:
        print("  裝置列表：")
        for d in info["cuda"]["devices"]:
            print(f"    - #{d['index']}: {d['name']}")
    if info["cuda"]["notes"]:
        print("  備註：")
        for n in info["cuda"]["notes"]:
            print(f"    - {n}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
