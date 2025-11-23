import cv2
import os
import pyttsx3
import numpy as np
from PIL import Image
from datetime import datetime
import speech_recognition as sr

engine = pyttsx3.init()
speech_recognizer = sr.Recognizer()

def speak(text):
    print(f"🗣️ {text}")
    engine.say(text)
    engine.runAndWait()

DATASET_DIR = "dataset"
NAMES_FILE = "names.txt"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
if not os.path.exists(NAMES_FILE):
    open(NAMES_FILE, "w", encoding="utf-8").close()

def register_face():
    id = input("🆔 Enter ID: ")
    name = input("👤 Enter Name: ")
    with open(NAMES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{id},{name}\n")
    cap = cv2.VideoCapture(0)
    count = 0
    speak(f"Capturing face data for {name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{DATASET_DIR}/User.{id}.{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}/50", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("📸 Capturing Faces - Press ESC to Stop", frame)
        if cv2.waitKey(1) == 27 or count >= 50:
            break
    cap.release()
    cv2.destroyAllWindows()
    speak(f"Face data collection completed for {name}")
    print(f"[✅ DONE] Collected {count} images for {name} (ID: {id})")

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
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
    speak("Training is starting")
    faces, ids = get_images_and_labels(DATASET_DIR)
    print(f"[📊] {len(faces)} face samples collected.")
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    speak("Model training complete")
    print("[✅] Training complete. Model saved as trainer.yml.")

def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
            f.write("Name,Time\n")
    with open(ATTENDANCE_FILE, "r+", encoding="utf-8") as f:
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines if "," in line]
        if name not in logged_names:
            with sr.Microphone() as source:
                speak(f"{name}, please say 'present' to confirm your attendance")
                speech_recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = speech_recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    command = speech_recognizer.recognize_google(audio).lower()
                    if "present" in command or "yes" in command:
                        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"{name},{time_now}\n")
                        speak(f"Attendance confirmed for {name}")
                except Exception as e:
                    print(f"[⚠️] Speech error: {e}")

def recognize_faces_live():
    if not os.path.exists(TRAINER_FILE):
        print("[❌ ERROR] trainer.yml not found. Please train the model first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    names_dict = {}
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    try:
                        id_str, name = line.split(",", 1)
                        names_dict[int(id_str)] = name
                    except:
                        continue
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 80:
                name = names_dict.get(id, "Unknown")
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("🎦 Webcam Attendance", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

while True:
    print("\n📌 Choose Mode:")
    print("1. Register new face")
    print("2. Train model")
    print("3. Live Webcam Recognition & Attendance")
    print("4. Exit")
    choice = input("Enter 1,2,3 or 4: ")
    if choice == "1":
        register_face()
    elif choice == "2":
        train_model()
    elif choice == "3":
        recognize_faces_live()
    elif choice == "4":
        break
    else:
        print("❌ Invalid choice, try again.")
