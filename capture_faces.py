import cv2
import os
import pyttsx3

engine = pyttsx3.init()
engine.say("Welcome to Smart Face Registration")
engine.runAndWait()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

id = input("🆔 Enter ID: ")
name = input("👤 Enter Name: ")

if not os.path.exists("dataset"):
    os.makedirs("dataset")

with open("names.txt", "a", encoding="utf-8") as f:
    f.write(f"{id},{name}\n")

cap = cv2.VideoCapture(0)
count = 0
engine.say(f"Capturing face data for {name}")
engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        filename = f"dataset/User.{id}.{count}.jpg"
        cv2.imwrite(filename, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("📸 Capturing Faces - Press ESC to Stop", frame)

    if cv2.waitKey(1) == 27 or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
engine.say(f"Face data collection completed for {name}")
engine.runAndWait()
print(f"[✅ DONE] Collected {count} images for {name} (ID: {id})")

