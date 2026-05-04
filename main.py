import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset_generator import DatasetGenerator
from src.trainer import ModelTrainer
from src.attendance_manager import AttendanceManager
from src.utils import ensure_dirs

def main_menu():
    ensure_dirs()
    
    while True:
        print("\n" + "="*30)
        print(" FACE RECOGNITION ATTENDANCE ")
        print("="*30)
        print("1. Register New Student")
        print("2. Train Recognition Model")
        print("3. Start Real-Time Attendance")
        print("4. Exit")
        print("="*30)
        
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            name = input("Enter Student Name: ").strip()
            roll = input("Enter Roll Number: ").strip()
            if name and roll:
                gen = DatasetGenerator()
                gen.capture_images(name, roll)
            else:
                print("[WARNING] Name and Roll Number are required!")
                
        elif choice == '2':
            trainer = ModelTrainer()
            trainer.train()
            
        elif choice == '3':
            manager = AttendanceManager()
            manager.start_recognition()
            
        elif choice == '4':
            print("Exiting system. Goodbye!")
            break
            
        else:
            print("[ERROR] Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n[INFO] System interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
