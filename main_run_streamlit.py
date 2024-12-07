import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            logging.warning(f"Skipping file {imagePath} due to incorrect format")
            continue
        faces = detector.detectMultiScale(imageNp, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            logging.warning(f"No faces found in {imagePath}")
            continue
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

def trainModel():
    faces, Ids = getImagesAndLabels('datasets')
    recognizer.train(faces, np.array(Ids))
    recognizer.write('trainer/trainer.yml')
    logging.info("Model trained and saved successfully!")

st.title("Attendance Management System using Face Recognition")

if st.button("Train Model"):
    trainModel()
    st.success("Model trained and saved successfully!")