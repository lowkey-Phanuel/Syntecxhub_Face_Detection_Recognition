"""
register_face.py
Add new people to the face database (capture & register faces).

Examples
--------
  # Capture 20 photos of yourself from the webcam
  python register_face.py --name Erasmus --webcam --samples 20

  # Register someone from a folder of their photos
  python register_face.py --name Kwame --images path/to/kwame_photos

  # Rebuild the whole database from the dataset/ folder
  python register_face.py --rebuild

  # See who is registered / remove someone
  python register_face.py --list
  python register_face.py --remove Kwame
"""

import argparse
import os
import shutil
import time

import cv2
import numpy as np

from face_engine import DATASET_DIR, IMAGE_EXTS, FaceEngine, load_db, save_db


def largest_face(faces):
    return max(faces, key=lambda f: f["box"][2] * f["box"][3])


def embeddings_from_folder(engine, folder):
    """Compute one embedding per image (using the largest face in each)."""
    vectors, skipped = [], 0
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith(IMAGE_EXTS):
            continue
        image = cv2.imread(os.path.join(folder, file))
        faces = engine.detect(image) if image is not None else []
        if not faces:
            skipped += 1
            continue
        vectors.append(engine.embed(image, largest_face(faces)))
    return vectors, skipped


def register_from_images(engine, name, source):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    for file in os.listdir(source):
        if file.lower().endswith(IMAGE_EXTS):
            shutil.copy(os.path.join(source, file), person_dir)
    return finish_registration(engine, name, person_dir)


def register_from_webcam(engine, name, samples, camera):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Look at the camera and move your head slightly between shots.")
    print("Photos are taken automatically when exactly ONE face is visible. Press Q to stop.")
    taken, last = 0, 0.0
    while taken < samples:
        ok, frame = cap.read()
        if not ok:
            break
        faces = engine.detect(frame)
        preview = frame.copy()
        for f in faces:
            x, y, w, h = f["box"]
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 200, 0), 2)

        if len(faces) == 1 and time.time() - last > 0.4:
            cv2.imwrite(os.path.join(person_dir, f"{name}_{int(time.time() * 1000)}.jpg"), frame)
            taken, last = taken + 1, time.time()

        msg = f"{name}: {taken}/{samples}" if len(faces) == 1 else "Need exactly one face in view"
        cv2.putText(preview, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Register face", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return finish_registration(engine, name, person_dir)


def finish_registration(engine, name, person_dir):
    vectors, skipped = embeddings_from_folder(engine, person_dir)
    if not vectors:
        print(f"[!] No faces found for {name}. Nothing registered.")
        return
    db = load_db()
    db[name] = np.array(vectors)
    save_db(db)
    print(f"[OK] Registered {name} with {len(vectors)} face samples (skipped {skipped} images without a face).")


def rebuild(engine):
    db = {}
    for name in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(folder):
            continue
        vectors, skipped = embeddings_from_folder(engine, folder)
        if vectors:
            db[name] = np.array(vectors)
            print(f"[OK] {name}: {len(vectors)} samples (skipped {skipped})")
    save_db(db)
    print(f"Database rebuilt: {len(db)} people.")


def main():
    p = argparse.ArgumentParser(description="Register faces for recognition")
    p.add_argument("--name", help="person's name (used as the label)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam", action="store_true", help="capture samples from the webcam")
    src.add_argument("--images", help="folder of photos of this person")
    p.add_argument("--samples", type=int, default=20, help="number of webcam photos (default 20)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--rebuild", action="store_true", help="rebuild database from dataset/")
    p.add_argument("--list", action="store_true", help="list registered people")
    p.add_argument("--remove", help="remove a registered person")
    args = p.parse_args()

    if args.list:
        db = load_db()
        print("Registered people:" if db else "No one registered yet.")
        for name, vecs in db.items():
            print(f"  - {name} ({len(vecs)} samples)")
        return
    if args.remove:
        db = load_db()
        db.pop(args.remove, None)
        save_db(db)
        shutil.rmtree(os.path.join(DATASET_DIR, args.remove), ignore_errors=True)
        print(f"Removed {args.remove}.")
        return

    engine = FaceEngine(detector="dnn")
    if args.rebuild:
        rebuild(engine)
    elif args.name and args.webcam:
        register_from_webcam(engine, args.name, args.samples, args.camera)
    elif args.name and args.images:
        register_from_images(engine, args.name, args.images)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
