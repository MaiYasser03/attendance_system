import cv2
import easyocr
import time
import logging
from utils.tts import speak
from utils.constants import NAME_RE, ID_RE

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.last_ocr_text = None
        self.last_ocr_time = 0

    def run_ocr(self, img_bgr):
        """Run OCR on a BGR image and return extracted text."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            text = " ".join(self.reader.readtext(gray, detail=0))
            if text.strip():
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

    def extract_name_id(self, text):
        """Extract name and ID from OCR text."""
        name = NAME_RE.search(text)
        idn = ID_RE.search(text)
        return (
            name.group(1).strip() if name else None,
            idn.group(1) if idn else None
        )
