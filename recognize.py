"""
recognize.py
Detect and recognise faces in an image, a video file, or a live webcam feed.
Draws a bounding box and a name label on every face (handles multiple faces).

Examples
--------
  python recognize.py --webcam                         # live webcam
  python recognize.py --image group_photo.jpg          # one image
  python recognize.py --video clip.mp4 --save          # video, save annotated copy
  python recognize.py --webcam --detector haar         # use Haar cascade instead of DNN
"""

import argparse
import os
import time

import cv2

from face_engine import BASE_DIR, FaceEngine

OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def run_image(engine, path, show=True):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    image, results = engine.annotate(image)

    print(f"{len(results)} face(s) found in {os.path.basename(path)}:")
    for name, score, box in results:
        print(f"  - {name:<12} similarity={score:.2f}  box={tuple(int(v) for v in box)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "annotated_" + os.path.basename(path))
    cv2.imwrite(out, image)
    print(f"Saved: {out}")

    if show:
        cv2.imshow("Face Recognition", image)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(engine, source, save=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    print("Press Q to quit, S to save a screenshot.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, results = engine.annotate(frame)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        info = f"Faces: {len(results)}   FPS: {fps:.1f}   Detector: {engine.method.upper()}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shot = os.path.join(OUTPUT_DIR, f"screenshot_{int(now)}.jpg")
            cv2.imwrite(shot, frame)
            print(f"Saved {shot}")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {OUTPUT_DIR}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Face detection and recognition")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", action="store_true", help="use the webcam")
    src.add_argument("--image", help="path to an image")
    src.add_argument("--video", help="path to a video file")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--detector", choices=["dnn", "haar"], default="dnn", help="face detector (default dnn)")
    p.add_argument("--save", action="store_true", help="save the annotated video")
    p.add_argument("--no-show", action="store_true", help="don't open a window (image mode)")
    args = p.parse_args()

    engine = FaceEngine(detector=args.detector)
    if not engine.db:
        print("[i] No one registered yet - every face will show as 'Unknown'.")
        print("    Register someone with: python register_face.py --name YourName --webcam")

    if args.image:
        run_image(engine, args.image, show=not args.no_show)
    elif args.video:
        run_stream(engine, args.video, save=args.save)
    else:
        run_stream(engine, args.camera, save=args.save)


if __name__ == "__main__":
    main()
