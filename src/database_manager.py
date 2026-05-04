import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path="data/students.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create the students table if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                roll_number TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add_student(self, roll_number, name):
        """Add or update a student in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO students (roll_number, name)
                VALUES (?, ?)
            ''', (roll_number, name))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding student: {e}")
            return False

    def get_student_name(self, roll_number):
        """Fetch student name by roll number."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM students WHERE roll_number = ?', (roll_number,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching student: {e}")
            return None

    def delete_student(self, roll_number):
        """Remove a student from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting student: {e}")
            return False

    def get_all_students(self):
        """Fetch all student records."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT roll_number, name FROM students')
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Error fetching all students: {e}")
            return []
