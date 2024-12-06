import tkinter as tk
import cv2
import numpy as np
import time
import threading
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the cv2.face module is available
if not hasattr(cv2, 'face'):
    raise ImportError("The 'cv2.face' module is not available. Make sure you have the opencv-contrib-python package installed.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = 'trainer/trainer.yml'

if not os.path.exists(model_path):
    logging.error("Model not found. Please train the model first.")
    raise FileNotFoundError("Model not found. Please train the model first.")

recognizer.read(model_path)
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

def capture_and_recognize():
    cam = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        ret, im = cam.read()
        if not ret:
            logging.error("Failed to access the camera!")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            cv2.rectangle(im, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
            cv2.putText(im, str(Id), (x, y-40), font, 2, (255, 255, 255), 3)

        cv2.imshow('im', im)
        if cv2.waitKey(10) & 0xFF == ord('q') or (time.time() - start_time) > 10:
            break
    cam.release()
    cv2.destroyAllWindows()

def start_recognition():
    threading.Thread(target=capture_and_recognize).start()

# GUI setup
window = tk.Tk()
window.title("Face Recognition Testing")
window.geometry('400x200')
window.configure(background='lightblue')

tk.Label(window, text="Face Recognition Testing", bg="lightblue", fg="black", font=('times', 15, 'bold')).pack(pady=20)

button_color = "#5DADE2"  # A complementary color to light blue

tk.Button(window, text="Start Recognition", command=start_recognition, fg="white", bg=button_color, width=20, height=2, font=('times', 12, 'bold')).pack(pady=20)

window.mainloop()