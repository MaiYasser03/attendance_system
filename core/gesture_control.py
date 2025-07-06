import cv2
import math
import numpy as np
import mediapipe as mp
import logging
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from utils.tts import speak

class OneHandGestureControl:
    def __init__(self, face_detector):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_distance = None
        self.scale = 1.0
        self.face_detector = face_detector

        # Audio setup
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Init system values
        self.brightness = sbc.get_brightness(display=0)[0]
        self.current_vol = self.volume.GetMasterVolumeLevelScalar() * 100
        logging.info(f"Initial brightness: {self.brightness}")
        logging.info(f"Initial volume: {self.current_vol}")

    def calculate_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def set_volume(self, vol_percent):
        vol_percent = np.clip(vol_percent, 0, 100)
        self.volume.SetMasterVolumeLevelScalar(vol_percent / 100, None)

    def set_brightness(self, brightness_percent):
        brightness_percent = np.clip(brightness_percent, 0, 100)
        sbc.set_brightness(brightness_percent)

    def apply_transform(self, frame):
        if frame is None or frame.size == 0:
            logging.error("Invalid frame in apply_transform")
            return frame
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, self.scale)
        return cv2.warpAffine(frame, M, (w, h))

    def is_authorized_person_present(self):
        if not self.face_detector.is_authenticated:
            speak("Please authenticate to use gesture control")
            logging.warning("Gesture control blocked: No authenticated user")
            return False
        return True

    def process_gestures(self, frame, frame_rgb, h, w):
        gesture_info = {}
        if not self.is_authorized_person_present():
            return gesture_info

        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                wrist = hand_landmarks.landmark[0]

                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))

                cv2.circle(frame, thumb_pos, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_pos, 10, (0, 0, 255), -1)
                cv2.line(frame, thumb_pos, index_pos, (255, 0, 255), 2)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                distance = self.calculate_distance(thumb_pos, index_pos)
                if self.prev_distance is not None:
                    diff = distance - self.prev_distance
                    if abs(diff) > 5:
                        self.scale += 0.05 if diff > 0 else -0.05
                        self.scale = np.clip(self.scale, 0.5, 3.0)
                self.prev_distance = distance

                brightness_target = (1 - wrist.y) * 100
                volume_target = wrist.x * 100
                self.set_brightness(brightness_target)
                self.set_volume(volume_target)

                gesture_info = {
                    'pinch_distance': int(distance),
                    'scale': self.scale,
                    'brightness': int(brightness_target),
                    'volume': int(volume_target)
                }

        return gesture_info
