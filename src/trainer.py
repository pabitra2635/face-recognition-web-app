import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from src.utils import DATASET_DIR, TRAINER_PATH, LABELS_PATH

class ModelTrainer:
    def __init__(self):
        # Create the recognizer
        # Note: requires opencv-contrib-python
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            print("[ERROR] LBPH recognizer not found. Please install 'opencv-contrib-python'.")
            self.recognizer = None

    def train(self):
        if self.recognizer is None:
            return False

        faces = []
        ids = []
        label_map = {} # Maps integer ID to (Name, RollNumber)
        
        # Get all subdirectories in dataset/
        user_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        
        if not user_dirs:
            print("[WARNING] No dataset found in 'dataset/'. Please register faces first.")
            return False

        print("[INFO] Loading dataset and training model...")
        
        # We need a progress bar for images
        print(f"[INFO] Loading images...")
        current_id = 0
        for user_dir in tqdm(user_dirs, desc="Processing Users"):
            user_path = os.path.join(DATASET_DIR, user_dir)
            images = [os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            parts = user_dir.rsplit('_', 1)
            if len(parts) == 2:
                name, roll_number = parts
                current_id += 1
                label_map[current_id] = {"name": name, "roll": roll_number}
                
                for img_path in images:
                    img_numpy = np.array(Image.open(img_path).convert('L'), 'uint8')
                    faces.append(img_numpy)
                    ids.append(current_id)

        if not faces:
            print("[WARNING] No face images found.")
            return False

        # Training with progress bar (fake progress for the actual training call, but real for loading)
        # Actually, let's use tqdm for the image loading phase which is the bulk
        print(f"[INFO] Processing {len(faces)} images...")
        
        # Training
        self.recognizer.train(faces, np.array(ids))
        
        # Save model
        os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
        self.recognizer.save(TRAINER_PATH)
        
        # Save labels mapping
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)
            
        print(f"[SUCCESS] Model trained and saved to {TRAINER_PATH}.")
        return True

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
