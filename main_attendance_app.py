import cv2
import numpy as np
import pyttsx3
import os
from datetime import datetime

engine = pyttsx3.init()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = {}
if os.path.exists("names.txt"):
    with open("names.txt", "r", encoding="utf-8") as f:
        for line in f:
            id, name = line.strip().split(",")
            names[int(id)] = name

def mark_attendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w", encoding="utf-8") as f:
            f.write("Name,Time\n")
    with open("attendance.csv", "r+", encoding="utf-8") as f:
        lines = f.readlines()
        logged = [line.split(",")[0] for line in lines]
        if name not in logged:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{now}\n")
            engine.say(f"Welcome {name}")
            engine.runAndWait()

def recognize_faces(frame, gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 60:
            name = names.get(id, "Unknown")
            mark_attendance(name)
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

print("\n📌 Choose Mode:")
print("1. Live Webcam Recognition")
print("2. Photo Upload Recognition")
choice = input("Enter 1 or 2: ")

if choice == "1":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = recognize_faces(frame, gray)
        cv2.imshow("🎦 Webcam Attendance", processed)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == "2":
    print("\n📁 Available images in 'uploads/' folder:")
    for img in os.listdir("uploads"):
        print(" -", img)
    img_name = input("Enter image file name: ")
    img_path = os.path.join("uploads", img_name)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = recognize_faces(img, gray)
        cv2.imshow("📸 Image Recognition", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


