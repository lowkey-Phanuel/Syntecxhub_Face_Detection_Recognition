# Face Detection & Recognition

**Syntecxhub Internship – Artificial Intelligence, Project 1**

A Python application that detects faces in images, videos or a live webcam feed and recognises registered people by name. Built with OpenCV.

## Features

- **Face detection** with two methods:
  - **DNN** (default) – OpenCV's YuNet deep-learning detector, accurate on angled and small faces
  - **Haar cascade** – the classic OpenCV detector (`--detector haar`)
- **Face recognition** with **face embeddings** (OpenCV SFace model). Each face becomes a 128-number vector; faces are matched by cosine similarity against registered people.
- **Bounding boxes and name labels** on every face, with the match score. Known people are green, unknown faces are red.
- **Multiple faces** handled in the same frame.
- **Registration scripts** to add new people from the webcam or from a folder of photos.
- Works on **images, video files and live webcam**, with FPS and face count shown on screen.

## How it works

```
Image / frame
     │
     ▼
Face detector (YuNet DNN or Haar)  ──►  box + 5 landmarks per face
     │
     ▼
Align & crop face (112×112)
     │
     ▼
SFace model  ──►  128-D embedding
     │
     ▼
Compare with face_db.pkl (cosine similarity)
     │
     ├── similarity ≥ 0.363  →  person's name
     └── otherwise           →  "Unknown"
```

## Project structure

| File | Purpose |
| --- | --- |
| `face_engine.py` | Core: detection, embeddings, matching, drawing, face database |
| `download_models.py` | Downloads the pre-trained models into `models/` |
| `register_face.py` | Captures and registers new people |
| `recognize.py` | Runs detection + recognition on webcam, image or video |
| `dataset/` | Saved photos of each registered person (one folder per person) |
| `face_db.pkl` | Stored face embeddings (created when you register someone) |
| `output/` | Annotated images, videos and screenshots |

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/Syntecxhub_Face_Detection_Recognition.git
cd Syntecxhub_Face_Detection_Recognition

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python download_models.py
```

## Usage

### 1. Register people

```bash
# Capture 20 photos from the webcam (taken automatically when one face is visible)
python register_face.py --name Erasmus --webcam --samples 20

# Or register from a folder of photos
python register_face.py --name Kwame --images path/to/kwame_photos

# List / remove registered people
python register_face.py --list
python register_face.py --remove Kwame

# Rebuild the database from everything in dataset/
python register_face.py --rebuild
```

### 2. Recognise faces

```bash
python recognize.py --webcam                        # live webcam (Q = quit, S = screenshot)
python recognize.py --image group_photo.jpg         # single image, saved to output/
python recognize.py --video clip.mp4 --save         # video file, annotated copy saved
python recognize.py --webcam --detector haar        # use the Haar cascade detector
```

## Results

Test on a combined image with three faces (two people, one shown twice, flipped and rotated):

| Face | Result (DNN) | Result (Haar) |
| --- | --- | --- |
| Person A, mirrored | Correct, similarity 0.93 | Correct, 0.62 |
| Person B, small face | Correct, 0.89 | Correct, 0.85 |
| Person A, rotated | Correct, 0.95 | Correct, 0.84 |
| Person B after being removed from the database | "Unknown", 0.14 | — |

The DNN detector gives tighter boxes and higher match scores because its landmarks allow proper face alignment; Haar only gives a box.

## Tips for accuracy

- Register 15–30 photos per person with different angles and lighting.
- Use good, even lighting when running recognition.
- If someone is wrongly recognised, register more varied photos of them. If known people show as "Unknown", lower `MATCH_THRESHOLD` in `face_engine.py` slightly (e.g. 0.33).

## Privacy

Face photos and embeddings are personal data. `dataset/` and `face_db.pkl` are excluded from Git by `.gitignore`, so no one's face is uploaded. Only register people who have agreed to it.

## Technologies

Python · OpenCV (DNN module, `FaceDetectorYN`, `FaceRecognizerSF`, `CascadeClassifier`) · NumPy

Models: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) from the OpenCV Model Zoo.
