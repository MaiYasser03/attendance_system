# Attendance System — Web Deployment

Face recognition, ID card OCR, and emotion-based attendance logging via **FastAPI** (backend) and **Streamlit** (frontend).

## Architecture

```
Browser (camera) → Streamlit frontend :8501 → FastAPI backend :8000 → CV pipeline
                                                      ↓
                                              attendance.csv
```

## Quick Start (Docker)

```bash
docker compose up --build
```

- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000/docs

First startup downloads DeepFace and EasyOCR models and may take several minutes.

## Quick Start (Local)

**Terminal 1 — Backend:**
```bash
pip install -r requirements-backend.txt
set DISABLE_TTS=true
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
pip install -r requirements-frontend.txt
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

## Usage

1. **Scan ID Card** tab — hold ID to camera, capture, click **Run ID OCR**
2. **Live Camera** tab — allow webcam access; face recognition runs every ~2.5s
3. When face matches someone in `Dataset/`, attendance is logged to `attendance.csv`
4. **Attendance Records** tab — view logged entries

## Dataset

Enroll faces under `Dataset/<PersonName>/` (e.g. `Dataset/Mai/Mai1.jpg`).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/process-frame` | Face detection + recognition |
| POST | `/api/ocr` | ID card OCR |
| GET | `/api/attendance` | List attendance records |
| GET | `/api/analytics/emotion` | Emotion pie chart (base64) |
| POST | `/api/reset` | Reset auth session |

## Original Desktop App

The original OpenCV desktop app still runs via:

```bash
python main.py
```
