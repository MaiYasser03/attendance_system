import base64
import logging
import os
from typing import Any

import cv2
import numpy as np
import pandas as pd

from core.face_emotion import FaceEmotionDetector
from core.ocr import OCRProcessor
from utils.constants import CSV_PATH


class AttendanceCVService:
    """Web-facing wrapper around the attendance CV pipeline."""

    def __init__(self):
        self.face_detector = FaceEmotionDetector()
        self.ocr_engine = OCRProcessor()
        self.last_ocr_text = ""

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return frame

    @staticmethod
    def _encode_image(frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def run_ocr(self, image_bytes: bytes) -> dict[str, Any]:
        frame = self._decode_image(image_bytes)
        text = self.ocr_engine.run_ocr(frame)
        name, id_num = self.ocr_engine.extract_name_id(text)

        if text and text not in ("No text detected", "OCR failed"):
            self.last_ocr_text = text
            self.face_detector.last_ocr_id = id_num

        return {
            "ocr_text": text,
            "name": name,
            "id": id_num,
            "parsed": bool(name and id_num),
        }

    def process_frame(
        self,
        image_bytes: bytes,
        *,
        run_face: bool = True,
        run_ocr: bool = False,
        mirror: bool = True,
    ) -> dict[str, Any]:
        frame = self._decode_image(image_bytes)

        if mirror:
            frame = cv2.flip(frame, 1)

        ocr_result = None
        if run_ocr:
            ocr_result = self.run_ocr(image_bytes)

        result: dict[str, Any] = {
            "authenticated": self.face_detector.is_authenticated,
            "identity": self.face_detector.last_authenticated_identity,
            "emotion": None,
            "name": None,
            "id": self.face_detector.last_ocr_id,
            "faces_detected": 0,
            "attendance_logged": False,
            "ocr_text": self.last_ocr_text or None,
        }

        if ocr_result:
            result["ocr_text"] = ocr_result["ocr_text"]
            result["name"] = ocr_result["name"]
            result["id"] = ocr_result["id"]

        if run_face:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            result["faces_detected"] = len(faces)

            prev_rows = len(self.face_detector.df)

            if len(faces) == 0:
                self.face_detector._auth_prompt()
            else:
                for (x, y, w, h) in faces:
                    roi = frame[y : y + h, x : x + w]
                    self.face_detector.analyze_face(roi, self.last_ocr_text)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            result["authenticated"] = self.face_detector.is_authenticated
            result["identity"] = self.face_detector.last_authenticated_identity
            result["id"] = self.face_detector.last_ocr_id
            result["emotion"] = self.face_detector.last_emotion
            result["attendance_logged"] = len(self.face_detector.df) > prev_rows

            if self.face_detector.last_authenticated_identity:
                name, id_num = self.face_detector.ocr_engine.extract_name_id(
                    self.last_ocr_text
                )
                result["name"] = name or self.face_detector.last_authenticated_identity
                result["id"] = id_num or self.face_detector.last_ocr_id

        auth_color = (0, 255, 0) if result["authenticated"] else (0, 0, 255)
        cv2.putText(
            frame,
            f"Authenticated: {result['authenticated']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            auth_color,
            2,
        )

        if result.get("name"):
            cv2.putText(
                frame,
                f"Name: {result['name']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if result.get("id"):
            cv2.putText(
                frame,
                f"ID: {result['id']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if result.get("emotion"):
            cv2.putText(
                frame,
                f"Emotion: {result['emotion']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        result["annotated_image"] = self._encode_image(frame)
        return result

    def get_attendance(self) -> list[dict[str, str]]:
        if not self.face_detector.df.empty:
            return self.face_detector.df.to_dict(orient="records")

        if CSV_PATH and os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, dtype=str)
            return df.to_dict(orient="records")
        return []

    def get_emotion_analytics_base64(self) -> str | None:
        self.face_detector.generate_emotion_analytics()
        try:
            with open("emotion_analytics.png", "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            return None

    def reset_session(self) -> None:
        self.face_detector.is_authenticated = False
        self.face_detector.last_authenticated_identity = None
        self.face_detector.last_ocr_id = None
        self.last_ocr_text = ""
        logging.info("Session reset")
