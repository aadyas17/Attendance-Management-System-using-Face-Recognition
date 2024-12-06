import tkinter as tk
import cv2
import os
import numpy as np
import time
import threading
import logging
from PIL import Image, ImageEnhance, ImageOps

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main window setup
window = tk.Tk()
window.title("Simple Attendance System")
window.geometry('600x400')
window.configure(background='lightblue')

# Function to capture and save images
def capture_images(enrollment, name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sampleNum = 0
    start_time = time.time()

    while True:
        ret, img = cam.read()
        if not ret:
            logging.error("Failed to access the camera!")
            Notification.configure(text="Failed to access the camera!", bg="red", fg="white")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            # Save the captured face image
            cv2.imwrite("TrainingImage/" + name + "." + enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Capturing Images', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum >= 50 or (time.time() - start_time) > 10:
            break

    cam.release()
    cv2.destroyAllWindows()
    logging.info(f"Image Captured and Saved for Enrollment: {enrollment} Name: {name}")
    Notification.configure(text="Image Captured and Saved for Enrollment: " + enrollment + " Name: " + name, bg="green", fg="white")

def take_img():
    enrollment = txt.get()
    name = txt2.get()
    if enrollment == '' or name == '':
        Notification.configure(text="Enrollment & Name required!", bg="red", fg="white")
    else:
        threading.Thread(target=capture_images, args=(enrollment, name)).start()

def augment_image(image):
    # Convert to PIL image
    pil_image = Image.fromarray(image)
    
    # Apply transformations
    augmented_images = [pil_image]
    augmented_images.append(ImageOps.mirror(pil_image))
    augmented_images.append(ImageOps.flip(pil_image))
    augmented_images.append(pil_image.rotate(15))
    augmented_images.append(pil_image.rotate(-15))
    
    # Convert back to numpy array
    augmented_images = [np.array(img) for img in augmented_images]
    return augmented_images

def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = "trainer/trainer.yml"
    
    if not os.path.exists(model_path):
        Notification.configure(text="Model not found. Please train the model first.", bg="red", fg="white")
        logging.error("Model not found. Please train the model first.")
        return

    recognizer.read(model_path)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def recognize():
        cam = cv2.VideoCapture(0)
        start_time = time.time()

        while True:
            ret, img = cam.read()
            if not ret:
                logging.error("Failed to access the camera!")
                Notification.configure(text="Failed to access the camera!", bg="red", fg="white")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                confidence = round(100 - confidence, 2)

                if confidence > 50:
                    try:
                        name = [file.split('.')[0] for file in os.listdir("TrainingImage") if int(file.split('.')[1]) == id][0]
                        cv2.putText(img, f"{name}, {confidence}%", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except IndexError:
                        cv2.putText(img, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Recognizing Faces", img)

            if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 10:
                break

        cam.release()
        cv2.destroyAllWindows()

    threading.Thread(target=recognize).start()

# GUI Elements
message = tk.Label(window, text="Simple Attendance System", bg="lightblue", fg="black", width=40, height=2, font=('times', 15, 'bold'))
message.place(x=50, y=20)

lbl = tk.Label(window, text="Enter Enrollment: ", width=20, height=2, fg="black", bg="lightblue", font=('times', 12, 'bold'))
lbl.place(x=50, y=100)

txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 12))
txt.place(x=250, y=115)

lbl2 = tk.Label(window, text="Enter Name: ", width=20, fg="black", bg="lightblue", height=2, font=('times', 12, 'bold'))
lbl2.place(x=50, y=150)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 12))
txt2.place(x=250, y=165)

Notification = tk.Label(window, text="", bg="lightblue", fg="black", width=40, height=2, font=('times', 12))
Notification.place(x=50, y=250)

button_color = "#5DADE2"  # A complementary color to light blue

takeImg = tk.Button(window, text="Capture Images", command=take_img, fg="white", bg=button_color, width=15, height=2, font=('times', 12, 'bold'))
takeImg.place(x=50, y=320)

recognizeBtn = tk.Button(window, text="Recognize Faces", command=recognize_faces, fg="white", bg=button_color, width=15, height=2, font=('times', 12, 'bold'))
recognizeBtn.place(x=250, y=320)

window.mainloop()