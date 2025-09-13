import cv2
import numpy as np
from PIL import Image
import os
import pyttsx3

engine = pyttsx3.init()
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = "dataset"

def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples, ids = [], []
    for image_path in image_paths:
        gray_image = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_image, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces = face_cascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

print("[🔍] Reading images from dataset...")
engine.say("Training is starting")
engine.runAndWait()

faces, ids = get_images_and_labels(path)
print(f"[📊] {len(faces)} face samples collected.")
recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")
engine.say("Model training complete")
engine.runAndWait()
print("[✅] Training complete. Model saved as trainer.yml.")
