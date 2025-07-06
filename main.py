import cv2
import time
import logging
from core.ocr import OCRProcessor
from core.face_emotion import FaceEmotionDetector
from core.gesture_control import OneHandGestureControl
from ui.camera_calibration import calibrate_camera
from utils.tts import speak
import utils.logger 
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Failed to open webcam")
        logging.error("Webcam not detected")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Calibration prompt
    print("Press 's' to skip calibration or 'c' to calibrate")
    mtx, dist = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, "Press 's' to skip, 'c' to calibrate, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            speak("Calibration skipped")
            break
        elif key == ord('c'):
            mtx, dist = calibrate_camera(cap)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Calibration")

    # Reinitialize after calibration
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Could not reopen webcam")
        logging.error("Webcam error post-calibration")
        return

    face_detector = FaceEmotionDetector()
    gesture_control = OneHandGestureControl(face_detector)
    ocr_engine = OCRProcessor()

    simplify_pipeline = False
    print("[SPACE] OCR  |  [E] Emotion Stats  |  [ESC] Exit  |  [P] Toggle Simple View")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(raw_frame.copy(), 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        gesture_info = gesture_control.process_gestures(frame, frame_rgb, h, w)
        frame = face_detector.process_faces(frame, gray)

        cv2.putText(frame, f"Authenticated: {face_detector.is_authenticated}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if face_detector.is_authenticated else (0, 0, 255), 2)

        if gesture_info:
            cv2.putText(frame, f"Brightness: {gesture_info['brightness']}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Volume: {gesture_info['volume']}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            face_detector.generate_emotion_analytics()
            speak("System shutting down")
            break
        elif key == ord(' '):  # OCR
            text = ocr_engine.run_ocr(raw_frame)
            name, id_ = ocr_engine.extract_name_id(text)
            if name and id_:
                speak(f"Detected {name}")
                print(f"✅ Name: {name}, ID: {id_}")
            else:
                speak("Could not parse ID")
                print("⚠️ OCR failed.")
        elif key == ord('e'):
            face_detector.generate_emotion_analytics()
            speak("Emotion chart generated")
        elif key == ord('p'):
            simplify_pipeline = not simplify_pipeline
            print(f"Pipeline mode: {'Simplified' if simplify_pipeline else 'Full'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
