import os
import cv2
import time
import logging
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from utils.constants import CSV_PATH
from utils.tts import speak
from core.ocr import OCRProcessor

class FaceEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.attendance_file = CSV_PATH
        self.df = self._load_attendance()
        self.ocr_engine = OCRProcessor()
        self.is_authenticated = False
        self.last_authenticated_identity = None
        self.last_ocr_id = None
        self.last_auth_prompt_time = 0

    def _load_attendance(self):
        if os.path.exists(self.attendance_file):
            return pd.read_csv(self.attendance_file, dtype=str)
        else:
            return pd.DataFrame(columns=["date", "name", "id", "emotion", "time"])

    def save_attendance(self, name, id_num, emotion):
        if not self.is_authenticated:
            return
        now = datetime.now()
        row = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "name": name,
            "id": f"'{id_num}",
            "emotion": emotion
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.attendance_file, index=False)
        speak(f"{name} marked present")
        logging.info(f"{name} marked present with emotion {emotion}")

    def analyze_face(self, face_img, ocr_text=""):
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_img)

            # Emotion detection
            analysis = DeepFace.analyze(img_path=temp_path, actions=["emotion"], enforce_detection=False, silent=True)
            emotion = analysis[0]["dominant_emotion"]

            # Identity matching
            matches = DeepFace.find(img_path=temp_path, db_path="Dataset/", model_name="Facenet", enforce_detection=False, silent=True)
            if matches and not matches[0].empty:
                identity_path = matches[0].iloc[0]["identity"]
                identity_name = os.path.basename(identity_path).split("/")[1].split(".")[0]

                if self.last_authenticated_identity != identity_name:
                    self.is_authenticated = True
                    self.last_authenticated_identity = identity_name

                    # Extract name/ID from OCR text
                    name, id_ = self.ocr_engine.extract_name_id(ocr_text)
                    if not name:
                        name = identity_name
                    if not id_:
                        id_ = "unknown"

                    self.last_ocr_id = id_
                    self.save_attendance(name, id_, emotion)
            else:
                self.is_authenticated = False
        except Exception as e:
            self.is_authenticated = False
            logging.error(f"Face analysis failed: {str(e)}")

    def process_faces(self, frame, gray, ocr_text=""):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            self._auth_prompt()
        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            self.analyze_face(roi, ocr_text)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def _auth_prompt(self):
        now = time.time()
        if not self.is_authenticated and now - self.last_auth_prompt_time > 5:
            speak("Please authenticate to use gesture control")
            self.last_auth_prompt_time = now

    def generate_emotion_analytics(self):
        if self.df.empty:
            return
        counts = self.df["emotion"].value_counts()
        counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Emotions")
        import matplotlib.pyplot as plt
        plt.savefig("emotion_analytics.png")
        logging.info("Saved emotion_analytics.png")
