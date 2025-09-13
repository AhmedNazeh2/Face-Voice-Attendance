import speech_recognition as sr
import os
import pyttsx3

engine = pyttsx3.init()
recognizer = sr.Recognizer()

def speak(text):
    print(f"🗣️ {text}")
    engine.say(text)
    engine.runAndWait()

def listen_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        return command
    except:
        speak("Sorry, I didn't catch that.")
        return ""

speak("Voice command system activated. Please say a command.")

while True:
    command = listen_command()
    if "register face" in command:
        speak("Opening face registration module")
        os.system("python capture_faces.py")
    elif "train model" in command:
        speak("Training the face recognition model")
        os.system("python face_train.py")
    elif "start attendance" in command:
        speak("Launching attendance system")
        os.system("python face_recognize.py")
    elif "exit" in command or "stop" in command:
        speak("Exiting the system. Goodbye.")
        break
