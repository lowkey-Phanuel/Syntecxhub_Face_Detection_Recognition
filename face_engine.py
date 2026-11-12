"""
face_engine.py
Shared core for the Face Detection & Recognition project.

- Detection  : OpenCV DNN face detector (YuNet)  -> cv2.FaceDetectorYN
               or the classic Haar cascade      -> cv2.CascadeClassifier
- Recognition: OpenCV DNN face embeddings (SFace) -> cv2.FaceRecognizerSF
               Each face is turned into a 128-number "embedding". Two faces of
               the same person give embeddings that point in a similar
               direction (high cosine similarity).
"""

import os
import pickle

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_PATH = os.path.join(BASE_DIR, "face_db.pkl")

DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
HAAR_MODEL = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")

# Cosine-similarity threshold recommended by the SFace authors.
# Above this value -> same person.
MATCH_THRESHOLD = 0.363

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class FaceEngine:
    def __init__(self, detector="dnn", score_threshold=0.8):
        if detector == "dnn":
            self._check_models()
            self.detector = cv2.FaceDetectorYN.create(
                DETECTOR_MODEL, "", (320, 320), score_threshold, 0.3, 5000
            )
        elif detector == "haar":
            cascade = HAAR_MODEL
            if not os.path.exists(cascade) and hasattr(cv2, "data"):
                cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.detector = cv2.CascadeClassifier(cascade)
            if self.detector.empty():
                raise FileNotFoundError("Haar cascade not found. Run:  python download_models.py")
        else:
            raise ValueError("detector must be 'dnn' or 'haar'")
        self.method = detector

        self._check_models()
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        self.db = load_db()

    @staticmethod
    def _check_models():
        for path in (DETECTOR_MODEL, RECOGNIZER_MODEL):
            if not os.path.exists(path) or os.path.getsize(path) < 100_000:
                raise FileNotFoundError(
                    f"Model missing or incomplete: {path}\n"
                    "Run:  python download_models.py"
                )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, image):
        """Return a list of faces. Each face is a dict with 'box' (x, y, w, h),
        'score' and 'raw' (the detector row, used for alignment)."""
        h, w = image.shape[:2]
        faces = []

        if self.method == "dnn":
            self.detector.setInputSize((w, h))
            _, result = self.detector.detect(image)
            if result is not None:
                for row in result:
                    x, y, bw, bh = row[:4].astype(int)
                    faces.append({"box": (x, y, bw, bh), "score": float(row[14]), "raw": row})
        else:  # haar
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, bw, bh) in boxes:
                faces.append({"box": (x, y, bw, bh), "score": 1.0, "raw": None})
        return faces

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def embed(self, image, face):
        """Turn one detected face into a normalised embedding vector."""
        if face["raw"] is not None:
            # DNN detector gives 5 landmarks (eyes, nose, mouth) -> proper alignment
            aligned = self.recognizer.alignCrop(image, face["raw"])
        else:
            # Haar gives only a box -> plain crop and resize
            x, y, bw, bh = face["box"]
            crop = image[max(y, 0):y + bh, max(x, 0):x + bw]
            aligned = cv2.resize(crop, (112, 112))
        feature = self.recognizer.feature(aligned).flatten()
        return feature / np.linalg.norm(feature)

    def identify(self, embedding):
        """Compare an embedding with every registered person.
        Returns (name, similarity). Name is 'Unknown' below the threshold."""
        best_name, best_score = "Unknown", -1.0
        for name, vectors in self.db.items():
            score = float(np.max(np.dot(vectors, embedding)))  # cosine similarity
            if score > best_score:
                best_name, best_score = name, score
        if best_score < MATCH_THRESHOLD:
            return "Unknown", best_score
        return best_name, best_score

    def annotate(self, image):
        """Detect + recognise every face and draw boxes and labels.
        Returns the annotated image and a list of (name, score, box)."""
        results = []
        for face in self.detect(image):
            name, score = self.identify(self.embed(image, face)) if self.db else ("Unknown", 0.0)
            results.append((name, score, face["box"]))
            draw_label(image, face["box"], name, score)
        return image, results


def draw_label(image, box, name, score):
    x, y, w, h = box
    color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)  # green known, red unknown
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y - th - 10, 0)
    cv2.rectangle(image, (x, top), (x + tw + 8, top + th + 10), color, -1)
    cv2.putText(image, text, (x + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ----------------------------------------------------------------------
# Face database: {name: np.array of embeddings (n_samples x 128)}
# ----------------------------------------------------------------------
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
