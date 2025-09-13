# 🤖 Smart Face Recognition Attendance System

A real-time **AI-powered attendance system** using OpenCV, face recognition, and voice interaction — built by Shrutika Darade 👩‍💻

This project allows you to:
- 👤 Detect and recognize faces using webcam or images
- 🗣️ Use voice greetings
- 📋 Log attendance in real-time
- 🎙️ Control using voice commands

---

## 🔍 Features

✅ Face registration via webcam  
✅ Real-time recognition from webcam  
✅ Image upload recognition  
✅ Voice-based greetings  
✅ Attendance logging (`attendance.csv`)  
✅ Voice command control system  
✅ Supports multiple users

---

## 💻 Technologies Used

- Python
- OpenCV (cv2, face recognition)
- Pyttsx3 (text-to-speech)
- SpeechRecognition (voice commands)
- NumPy
- Pandas
- Pillow (image support)

---

## 📂 Folder Structure

Smart-Face-Attendance-System/
├── capture_faces.py # Register new face via webcam
├── train_model.py # Train model from dataset
├── main_attendance_app.py # Unified live + photo recognition system
├── recognize_image.py # Run recognition on an uploaded photo
├── voice_command_listener.py # Control app with voice
├── names.txt # Store ID,name mapping
├── trainer.yml # Trained model (auto-generated)
├── attendance.csv # Attendance logs (auto-generated)
├── dataset/ # Face dataset (auto-generated)
├── requirements.txt # Install dependencies
├── .gitignore # Hide sensitive/generated files
└── README.md # This file


---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shruti6104/Face-Voice-Attendance.git
cd Face-Voice-Attendance

2️⃣ Install Dependencies
baash
pip install -r requirements.txt

3️⃣ Register Your Face
bash
python capture_faces.py

4️⃣ Train the Model
bash
python train_model.py

5️⃣ Start the System (Webcam + Photo Support)
bash
python main_attendance_app.py
✍️ Sample: names.txt
1,ahmed
2,nazeh

✅ Output Sample (attendance.csv)
Name,Time
ahmed,2025-06-21 20:45:12
🔊 Use Voice Commands
bash
python voice_command_listener.py
You can say:

register face

train model

start attendance

exit

📸 Optional: Image Test
Run:
python recognize_image.py
Then enter a photo name like test.jpg.

🧠 Built With
LBPHFaceRecognizer from OpenCV

Haar cascades for detection

pyttsx3 for offline speech

speech_recognition for voice control

💡 Future Improvements
Streamlit web app interface 🌐

Firebase / database attendance logs 🧾

Admin dashboard for multiple class tracking 📊



