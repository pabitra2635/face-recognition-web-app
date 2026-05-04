import cv2
import os
import sys
from src.database_manager import DatabaseManager
from src.utils import DATASET_DIR, get_face_cascade

class DatasetGenerator:
    def __init__(self):
        self.db = DatabaseManager()
        self.face_cascade = get_face_cascade()

    def capture_images(self, name, roll_number, num_samples=50):
        """Capture images from webcam and save to dataset directory."""
        user_dir = os.path.join(DATASET_DIR, f"{name}_{roll_number}")
        os.makedirs(user_dir, exist_ok=True)

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("[ERROR] Camera not found!")
            return False

        print(f"[INFO] Starting capture for {name} ({roll_number})...")
        print("[INFO] Look at the camera and wait until the process is complete.")

        count = 0
        while count < num_samples:
            ret, frame = cam.read()
            if not ret:
                print("[ERROR] Failed to grab frame!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                file_path = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(file_path, face_img)
                
                # Draw rectangle for visual feedback
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Registering Face - Press 'q' to Cancel", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Capture cancelled by user.")
                break

        cam.release()
        cv2.destroyAllWindows()

        if count >= num_samples:
            self.db.add_student(roll_number, name)
            print(f"[SUCCESS] Captured {count} images for {name}.")
            return True
        else:
            print("[WARNING] Not enough images captured.")
            return False

if __name__ == "__main__":
    # For testing independently
    name = input("Enter Name: ")
    roll = input("Enter Roll Number: ")
    gen = DatasetGenerator()
    gen.capture_images(name, roll)
