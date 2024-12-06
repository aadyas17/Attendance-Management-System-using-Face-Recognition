import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from datetime import datetime
import threading
import time
import shutil
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the main window
window = tk.Tk()
window.title("Attendance Management System")
window.geometry("900x700")
window.configure(bg="lightblue")

# Ensure the necessary directories exist
directories = ["dataset", "trainer", "attendance"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Haarcascade path for face detection
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def capture_images(user_id, user_name):
    cam = cv2.VideoCapture(0)
    sample_count = 0
    start_time = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Failed to access the camera!")
            messagebox.showerror("Error", "Failed to access the camera!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            cv2.imwrite(f"dataset/{user_name}.{user_id}.{sample_count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Samples: {sample_count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Taking Images", frame)

        if sample_count >= 50 or cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 10:
            break

    cam.release()
    cv2.destroyAllWindows()
    logging.info(f"Images saved for {user_name} with ID {user_id}!")
    messagebox.showinfo("Success", f"Images saved for {user_name} with ID {user_id}!")

def take_images():
    user_id = id_entry.get()
    user_name = name_entry.get()

    if user_id == "" or user_name == "":
        messagebox.showerror("Input Error", "Please fill in both Name and ID!")
        return

    threading.Thread(target=capture_images, args=(user_id, user_name)).start()

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    image_paths = [os.path.join("dataset", img) for img in os.listdir("dataset")]

    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces.append(image_np)
        ids.append(id)

    recognizer.train(faces, np.array(ids))
    recognizer.write("trainer/trainer.yml")

    logging.info("Model trained successfully!")
    messagebox.showinfo("Success", "Model trained successfully!")

def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    def recognize():
        cam = cv2.VideoCapture(0)
        start_time = time.time()

        while True:
            ret, frame = cam.read()
            if not ret:
                logging.error("Failed to access the camera!")
                messagebox.showerror("Error", "Failed to access the camera!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                confidence = round(100 - confidence, 2)

                if confidence > 50:
                    user_name = [file.split('.')[0] for file in os.listdir("dataset") if int(file.split('.')[1]) == id][0]
                    mark_attendance(user_name, id)
                    cv2.putText(frame, f"{user_name}, {confidence}%", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Recognizing Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 10:
                break

        cam.release()
        cv2.destroyAllWindows()

    threading.Thread(target=recognize).start()

def mark_attendance(name, id):
    file_path = "attendance/attendance.csv"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("ID,Name,Date,Time\n")

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(file_path, "a") as f:
        f.write(f"{id},{name},{date},{time}\n")

def manual_attendance():
    file_path = "attendance/manual_attendance.csv"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("ID,Name,Date,Time\n")

    user_id = id_entry.get()
    user_name = name_entry.get()

    if user_id == "" or user_name == "":
        messagebox.showerror("Input Error", "Please fill in both Name and ID!")
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(file_path, "a") as f:
        f.write(f"{user_id},{user_name},{date},{time}\n")

    messagebox.showinfo("Success", "Manual attendance recorded!")

def check_registered_students():
    registered_students = os.listdir("dataset")
    if not registered_students:
        messagebox.showinfo("Info", "No registered students found!")
        return

    # Extract unique student names and IDs
    unique_students = set()
    for filename in registered_students:
        parts = filename.split('.')
        if len(parts) >= 2:
            student_name = parts[0]
            student_id = parts[1]
            unique_students.add(f"{student_name}.{student_id}")

    students = "\n".join(unique_students)
    messagebox.showinfo("Registered Students", f"Registered Students:\n{students}")

def clear_previous_entries():
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
    messagebox.showinfo("Success", "Previous entries and registered students have been cleared!")

# UI Elements
title_label = tk.Label(window, text="Attendance Management System", font=("Arial", 24), bg="lightblue")
title_label.grid(row=0, column=0, columnspan=3, pady=20)

name_label = tk.Label(window, text="Name:", font=("Arial", 14), bg="lightblue")
name_label.grid(row=1, column=0, padx=20, pady=10, sticky="e")
name_entry = tk.Entry(window, font=("Arial", 14))
name_entry.grid(row=1, column=1, padx=20, pady=10, sticky="w")

id_label = tk.Label(window, text="ID:", font=("Arial", 14), bg="lightblue")
id_label.grid(row=2, column=0, padx=20, pady=10, sticky="e")
id_entry = tk.Entry(window, font=("Arial", 14))
id_entry.grid(row=2, column=1, padx=20, pady=10, sticky="w")

button_color = "#5DADE2"  # A complementary color to light blue

take_images_button = tk.Button(window, text="Take Images", font=("Arial", 12), bg=button_color, fg="white", command=take_images)
take_images_button.grid(row=3, column=0, padx=20, pady=10)

train_model_button = tk.Button(window, text="Train Model", font=("Arial", 12), bg=button_color, fg="white", command=train_model)
train_model_button.grid(row=3, column=1, padx=20, pady=10)

recognize_faces_button = tk.Button(window, text="Recognize Faces", font=("Arial", 12), bg=button_color, fg="white", command=recognize_faces)
recognize_faces_button.grid(row=3, column=2, padx=20, pady=10)

manual_attendance_button = tk.Button(window, text="Manual Attendance", font=("Arial", 12), bg=button_color, fg="white", command=manual_attendance)
manual_attendance_button.grid(row=4, column=0, padx=20, pady=10)

check_registered_students_button = tk.Button(window, text="Check Registered Students", font=("Arial", 12), bg=button_color, fg="white", command=check_registered_students)
check_registered_students_button.grid(row=4, column=1, padx=20, pady=10)

clear_previous_entries_button = tk.Button(window, text="Clear Previous Entries", font=("Arial", 12), bg=button_color, fg="white", command=clear_previous_entries)
clear_previous_entries_button.grid(row=4, column=2, padx=20, pady=10)

exit_button = tk.Button(window, text="Exit", font=("Arial", 12), bg="red", fg="white", command=window.quit)
exit_button.grid(row=5, column=1, padx=20, pady=20)

# Run the main loop
window.mainloop()