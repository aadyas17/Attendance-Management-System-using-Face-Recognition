import cv2
import os
import numpy as np
from PIL import Image
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            logging.warning(f"Skipping file {imagePath} due to incorrect format")
            continue
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # If a face is there then append that in the list as well as Id of it
        if len(faces) == 0:
            logging.warning(f"No faces found in {imagePath}")
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
            logging.info(f"Face found in {imagePath}, ID: {Id}")
    return faceSamples, Ids

faces, Ids = getImagesAndLabels('TrainingImage')
if len(faces) == 0 or len(Ids) == 0:
    logging.error("No faces found. Ensure that the images contain detectable faces.")
else:
    recognizer.train(faces, np.array(Ids))
    recognizer.save('trainer/trainer.yml')
    logging.info("Model trained and saved successfully!")