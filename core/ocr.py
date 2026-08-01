import cv2
import pytesseract
import time
import logging
import re
from utils.tts import speak
from utils.constants import NAME_RE, ID_RE, NAME_LABEL_CORRECTIONS


class OCRProcessor:
    def __init__(self):
        self.last_ocr_text = None
        self.last_ocr_time = 0

    def run_ocr(self, img_bgr):
        """Run OCR on a BGR image and return extracted text."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            text = pytesseract.image_to_string(gray).strip()
            if text:
                logging.info(f"OCR: {text}")
                self.last_ocr_text = text
                self.last_ocr_time = time.time()
                speak("Text extracted from ID card")
                return text
            else:
                speak("No text detected")
                return "No text detected"
        except Exception as e:
            logging.error(f"OCR failed: {str(e)}")
            speak("OCR failed")
            return "OCR failed"

    def normalize_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ")
        for pattern, replacement in NAME_LABEL_CORRECTIONS:
            text = pattern.sub(replacement, text)
        text = re.sub(r"[^A-Za-z0-9\s:-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_name(self, name: str) -> str:
        name = re.sub(r"[^A-Za-z ]+", " ", name).strip()
        return " ".join(word.capitalize() for word in name.split() if word)

    def extract_name_id(self, text):
        normalized = self.normalize_text(text)
        idn = ID_RE.search(normalized)
        name_match = NAME_RE.search(normalized)
        if name_match:
            return self.clean_name(name_match.group(1)), idn.group(1) if idn else None
        if idn:
            before_id = normalized[: idn.start()].strip()
            fallback_match = re.search(
                r"(?:name|nama|nane|nom[e]?|full name|student)[\s:\-]*([A-Za-z ]{2,})$",
                before_id, re.I,
            )
            if fallback_match:
                return self.clean_name(fallback_match.group(1)), idn.group(1)
        title_candidate = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", normalized)
        if title_candidate:
            return self.clean_name(title_candidate.group(1)), idn.group(1) if idn else None
        return None, idn.group(1) if idn else None
