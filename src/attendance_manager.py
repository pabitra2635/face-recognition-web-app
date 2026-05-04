import cv2
import os
import pickle
import pandas as pd
from datetime import datetime
from src.utils import TRAINER_PATH, LABELS_PATH, ATTENDANCE_DIR, get_face_cascade

class AttendanceManager:
    def __init__(self):
        self.face_cascade = get_face_cascade()
        self.recognizer = None
        self.label_map = {}
        self.attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
        self._load_model()
        self._initialize_attendance_file()

    def _load_model(self):
        """Load trained LBPH model and label mapping."""
        if not os.path.exists(TRAINER_PATH) or not os.path.exists(LABELS_PATH):
            print("[ERROR] Model files not found. Please train the model first.")
            return False
        
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(TRAINER_PATH)
            with open(LABELS_PATH, "rb") as f:
                self.label_map = pickle.load(f)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def _initialize_attendance_file(self):
        """Create the attendance CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["Name", "Roll Number", "Date", "Time"])
            df.to_csv(self.attendance_file, index=False)

    def log_attendance(self, name, roll_number):
        """Log attendance if not already logged today."""
        df = pd.read_csv(self.attendance_file)
        
        # Check if already logged today
        if not ((df['Roll Number'].astype(str) == str(roll_number)) & (df['Date'] == datetime.now().strftime('%Y-%m-%d'))).any():
            new_entry = {
                "Name": name,
                "Roll Number": roll_number,
                "Date": datetime.now().strftime('%Y-%m-%d'),
                "Time": datetime.now().strftime('%H:%M:%S')
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(self.attendance_file, index=False)
            print(f"[INFO] Attendance logged for {name} ({roll_number})")
            return True
        return False

    def start_recognition(self):
        """Start real-time recognition and attendance logging."""
        if self.recognizer is None:
            print("[ERROR] Recognizer not initialized.")
            return

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("[ERROR] Camera not found!")
            return

        print("[INFO] Starting real-time attendance...")
        print("[INFO] Press 'q' to exit.")

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                # Recognition
                user_id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])

                # Confidence for LBPH is distance (lower is better)
                # Typically < 100 is a good match
                if confidence < 70:
                    user_data = self.label_map.get(user_id, {"name": "Unknown", "roll": "N/A"})
                    name = user_data["name"]
                    roll = user_data["roll"]
                    label = f"{name} ({roll})"
                    color = (0, 255, 0)
                    
                    # Log attendance
                    self.log_attendance(name, roll)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                # Draw UI
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Conf: {round(100 - confidence)}%", (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Attendance System - Press 'q' to Exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    manager = AttendanceManager()
    manager.start_recognition()
