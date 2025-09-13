import cv2
import os
import pyttsx3
import speech_recognition as sr
from datetime import datetime

engine = pyttsx3.init()
speech_recognizer = sr.Recognizer()

UPLOAD_DIR = "dataset"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer.yml"
NAMES_FILE = "names.txt"
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(TRAINER_PATH):
    print("[❌ ERROR] trainer.yml not found. Please train the model first.")
    exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

names = {}
if os.path.exists(NAMES_FILE):
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                try:
                    id_str, name = line.split(",", 1)
                    names[int(id_str)] = name
                except ValueError:
                    continue
else:
    print("[❌ ERROR] names.txt file not found.")
    exit()

def speak(text):
    print(f"🗣️ {text}")
    engine.say(text)
    engine.runAndWait()

def listen_for_present():
    with sr.Microphone() as source:
        speak("Please say present to confirm your attendance")
        speech_recognizer.adjust_for_ambient_noise(source)
        try:
            audio = speech_recognizer.listen(source, timeout=5, phrase_time_limit=3)
            command = speech_recognizer.recognize_google(audio).lower()
            if "present" in command or "yes" in command:
                return True
        except Exception as e:
            print(f"[⚠️] Speech error: {e}")
    return False

def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
            f.write("Name,Time\n")
    with open(ATTENDANCE_FILE, "r+", encoding="utf-8") as f:
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines if "," in line]
        if name not in logged_names:
            if listen_for_present():
                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{name},{time_now}\n")
                speak(f"Attendance confirmed for {name}")

images = [img for img in os.listdir(UPLOAD_DIR) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"[⚠️ WARNING] No image files found in {UPLOAD_DIR}/")
else:
    for image_name in images:
        image_path = os.path.join(UPLOAD_DIR, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 60:
                name = names.get(id, "Unknown")
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(f"📸 {image_name}", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

