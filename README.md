# Attendance System with Face Recognition

A comprehensive system for:
- Face recognition and authentication
- Emotion detection
- Hand gesture control
- ID card OCR
- Attendance logging

## Features

- Real-time face detection and recognition
- Emotion analysis (happy, sad, angry, etc.)
- Hand gesture control for volume and brightness
- ID card OCR for authentication
- Attendance logging with CSV backup
- New user registration

## Installation

1. Clone this repository
2. Install dependencies:
3. Prepare your dataset:
- in the `dataset` folder
- Add subfolders for each person (e.g., `dataset/Mai/`, `dataset/Ahmed/`,...etc.)
- Add images named like `Mai1.jpg`, `Mai2.jpg`, etc.

### Key Controls:
- **SPACE**: Capture ID card
- **R**: Enter registration mode
- **C**: Capture image (in registration mode)
- **E**: Generate emotion analytics
- **P**: Toggle processing pipeline
- **ESC**: Quit

📊 Attendance Logging
All attendance entries are stored in attendance.csv with:
Date
Time
Name
ID number
Detected emotion
A backup is saved as attendance_backup.csv
Emotion analytics chart is saved as emotion_analytics.png

🔒 Authentication Process
OCR extracts name and ID from physical ID card.
Face recognition is matched with the dataset folder.
Attendance is logged only when both OCR and face match.
the demo link: https://attendance-system-demo.streamlit.app/ 
<img width="1917" height="858" alt="image" src="https://github.com/user-attachments/assets/64695ab8-8e1a-4d0f-b6e0-eba4d0d03f2d" />


🗣️ Audio Feedback
Text-to-speech (pyttsx3) is used to:
Announce authentication results
Prompt the user for input
Confirm successful attendance logging
