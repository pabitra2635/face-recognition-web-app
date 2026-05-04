from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import os
import time
import shutil
import pandas as pd
import base64
import numpy as np
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

# Removed local CameraManager as camera will be captured via JS in the browser

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    image_data = data.get('image')
    mode = data.get('mode', 'attendance')
    
    try:
        header, encoded = image_data.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        response_data = {"status": "success"}

        if mode == "registration":
            name = data.get('name')
            roll = data.get('roll')
            count = data.get('count', 0)
            num_samples = 50
            
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
            
            if count >= num_samples:
                db.add_student(roll, name)
                
            response_data['count'] = count

        elif mode == "attendance":
            manager = AttendanceManager()
            for (x, y, w, h) in faces:
                try:
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
                except Exception:
                    label = "Unknown"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        response_data['image'] = f"data:image/jpeg;base64,{processed_image}"
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    trainer = ModelTrainer()
    success = trainer.train()
    return jsonify({"success": success})

@app.route('/release_camera', methods=['POST'])
def release_camera():
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
