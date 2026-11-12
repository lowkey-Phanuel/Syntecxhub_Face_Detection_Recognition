"""
download_models.py
Downloads the pre-trained OpenCV DNN models (from the official OpenCV model zoo):
  - YuNet  : face detector
  - SFace  : face recognizer (face embeddings)
  - Haar cascade for frontal faces (classic detector, optional)

Usage:  python download_models.py
"""

import os
import urllib.request

from face_engine import DETECTOR_MODEL, HAAR_MODEL, MODELS_DIR, RECOGNIZER_MODEL

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    DETECTOR_MODEL: ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_MODEL: ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    HAAR_MODEL: "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
                "haarcascade_frontalface_default.xml",
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, url in MODELS.items():
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"[OK] already downloaded: {os.path.basename(path)}")
            continue
        print(f"[..] downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[OK] saved {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("All models ready.")


if __name__ == "__main__":
    main()
