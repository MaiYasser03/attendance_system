import os
import cv2
import time
import logging
import numpy as np
import pandas as pd
import onnxruntime as ort
from datetime import datetime
from utils.constants import CSV_PATH
from utils.tts import speak
from core.ocr import OCRProcessor

EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness",
                  "anger", "disgust", "fear", "contempt"]
CONFIDENCE_THRESHOLD = 80  # lower LBPH distance = better match; tune as needed
DATASET_DIR = "Dataset"
MODEL_PATH = "models/emotion-ferplus.onnx"


class FaceEmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.attendance_file = CSV_PATH
        self.df = self._load_attendance()
        self.ocr_engine = OCRProcessor()
        self.is_authenticated = False
        self.last_authenticated_identity = None
        self.last_ocr_id = None
        self.last_emotion = None
        self.last_auth_prompt_time = 0

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self._train_recognizer()

        self.emotion_session = None
        if os.path.exists(MODEL_PATH):
            self.emotion_session = ort.InferenceSession(MODEL_PATH)
        else:
            logging.warning(f"Emotion model not found at {MODEL_PATH}; emotion detection disabled.")

    def _train_recognizer(self):
        faces, labels = [], []
        label_id = 0
        if not os.path.isdir(DATASET_DIR):
            logging.warning(f"Dataset dir '{DATASET_DIR}' not found; face ID disabled until added.")
            return
        for person_name in sorted(os.listdir(DATASET_DIR)):
            person_dir = os.path.join(DATASET_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
            self.label_map[label_id] = person_name
            for fname in os.listdir(person_dir):
                path = os.path.join(person_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                detected = self.face_cascade.detectMultiScale(img, 1.1, 5)
                if len(detected) == 0:
                    face_img = cv2.resize(img, (200, 200))
                else:
                    x, y, w, h = detected[0]
                    face_img = cv2.resize(img[y:y+h, x:x+w], (200, 200))
                faces.append(face_img)
                labels.append(label_id)
            label_id += 1
        if faces:
            self.recognizer.train(faces, np.array(labels))
            logging.info(f"Trained face recognizer on {len(faces)} images, {label_id} people.")

    def _load_attendance(self):
        if os.path.exists(self.attendance_file):
            return pd.read_csv(self.attendance_file, dtype=str)
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

    def _predict_emotion(self, face_gray):
        if self.emotion_session is None:
            return "unknown"
        try:
            resized = cv2.resize(face_gray, (64, 64)).astype(np.float32)
            input_tensor = resized.reshape(1, 1, 64, 64)
            input_name = self.emotion_session.get_inputs()[0].name
            output = self.emotion_session.run(None, {input_name: input_tensor})[0][0]
            exp = np.exp(output - np.max(output))
            probs = exp / exp.sum()
            return EMOTION_LABELS[int(np.argmax(probs))]
        except Exception as e:
            logging.error(f"Emotion prediction failed: {e}")
            return "unknown"

    def analyze_face(self, face_img, ocr_text=""):
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (200, 200))

            emotion = self._predict_emotion(gray)
            self.last_emotion = emotion

            if not self.label_map:
                self.is_authenticated = False
                return

            label_id, confidence = self.recognizer.predict(gray_resized)
            if confidence < CONFIDENCE_THRESHOLD:
                identity_name = self.label_map.get(label_id)
                if self.last_authenticated_identity != identity_name:
                    self.is_authenticated = True
                    self.last_authenticated_identity = identity_name

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
