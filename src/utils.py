import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
DB_PATH = os.path.join(BASE_DIR, "data", "students.db")
TRAINER_PATH = os.path.join(MODELS_DIR, "trainer.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.pickle")

# Cascade classifier path for face detection
# Using default OpenCV haarcascade
FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") if 'cv2' in globals() else None

def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [DATASET_DIR, MODELS_DIR, ATTENDANCE_DIR, os.path.dirname(DB_PATH)]:
        os.makedirs(d, exist_ok=True)

import cv2 # Import here to avoid global scope issues if cv2 is not installed yet
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
