from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import os
import time
import shutil
import pandas as pd
from datetime import datetime
from src.database_manager import DatabaseManager
from src.dataset_generator import DatasetGenerator
from src.trainer import ModelTrainer
from src.attendance_manager import AttendanceManager
from src.utils import DATASET_DIR, ATTENDANCE_DIR, get_face_cascade, ensure_dirs

app = Flask(__name__)
ensure_dirs()

# Initialize Managers
db = DatabaseManager()
face_cascade = get_face_cascade()

class CameraManager:
    def __init__(self):
        self.video = None
    
    def get_video(self):
        if self.video is None or not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
        return self.video

    def release_video(self):
        if self.video is not None:
            self.video.release()
            self.video = None

cam_manager = CameraManager()

def gen_frames(mode="recognition", name=None, roll=None):
    """Video streaming generator."""
    video = cam_manager.get_video()
    
    # If mode is registration, we might want to track counts
    count = 0
    num_samples = 50
    
    if mode == "attendance":
        manager = AttendanceManager()
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if mode == "registration" and name and roll:
            user_dir = os.path.join(DATASET_DIR, f"{name}_{roll}")
            os.makedirs(user_dir, exist_ok=True)
            
            for (x, y, w, h) in faces:
                if count < num_samples:
                    count += 1
                    cv2.imwrite(os.path.join(user_dir, f"{count}.jpg"), gray[y:y+h, x:x+w])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Registration Complete!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # We should stop or signal completion
            
            if count >= num_samples:
                db.add_student(roll, name)

        elif mode == "attendance":
            for (x, y, w, h) in faces:
                user_id, confidence = manager.recognizer.predict(gray[y:y+h, x:x+w])
                
                if confidence < 75: # Lower is better for LBPH
                    user_data = manager.label_map.get(user_id, {"name": "Unknown", "roll": "N/A"})
                    n = user_data["name"]
                    r = user_data["roll"]
                    manager.log_attendance(n, r)
                    label = f"{n} ({r})"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    students = db.get_all_students()
    return render_template('index.html', students=students)

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/attendance_page')
def attendance_page():
    return render_template('attendance.html')

@app.route('/logs')
def logs():
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    
    all_students = db.get_all_students()
    present_data = []
    
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        present_data = df.to_dict('records')
    
    present_rolls = [str(p['Roll Number']) for p in present_data]
    absent_students = [s for s in all_students if str(s[0]) not in present_rolls]
    
    return render_template('logs.html', 
                           logs=present_data, 
                           absent=absent_students, 
                           date=date_str,
                           total=len(all_students),
                           present_count=len(present_data))

@app.route('/delete_student/<roll>', methods=['POST'])
def delete_student(roll):
    # Get student name to delete dataset folder
    student_name = db.get_student_name(roll)
    if student_name:
        # Delete from DB
        db.delete_student(roll)
        # Delete dataset folder
        folder_name = f"{student_name}_{roll}"
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Student not found"})

@app.route('/video_feed/<mode>')
def video_feed(mode):
    name = request.args.get('name')
    roll = request.args.get('roll')
    return Response(gen_frames(mode, name, roll),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['POST'])
def train():
    trainer = ModelTrainer()
    success = trainer.train()
    return jsonify({"success": success})

@app.route('/release_camera', methods=['POST'])
def release_camera():
    cam_manager.release_video()
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
