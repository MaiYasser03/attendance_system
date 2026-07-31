# Attendance System — Web Deployment

Face recognition, ID card OCR, and emotion-based attendance logging via **FastAPI** (backend) and **Streamlit** (frontend).

## Architecture

```
Browser (camera) → Streamlit frontend :8501 → FastAPI backend :8000 → CV pipeline
                                                      ↓
                                              attendance.csv
```

## Run modes

This repository is trimmed to make the Streamlit frontend deployable on
Streamlit Deploy. There are two main modes:

- Demo mode (no backend): runs entirely inside Streamlit and is suitable
        for quick demos or Streamlit Deploy.
- Full mode (with backend): uses the FastAPI backend for real CV
        processing; the backend requires additional, heavy dependencies and
        must be hosted separately.

Demo mode (Streamlit Deploy or local):

```powershell
pip install -r requirements.txt
# Optional: set DEMO_MODE=1 to enable by default, or toggle in sidebar
streamlit run streamlit_app.py
```

On the Streamlit UI, enable the **Demo mode (no backend)** toggle in the
sidebar to run without a backend. In demo mode the app uses a lightweight
face detector to simulate authentication and shows placeholder data.

Use project models locally (DeepFace / EasyOCR):

If you want the frontend to run the actual models in-process (no separate
backend service), enable **Use local models (loads DeepFace/EasyOCR)** in
the sidebar. You must install the heavy backend dependencies first:

```powershell
pip install -r requirements-backend.txt
```

Warning: these packages (TensorFlow, DeepFace, EasyOCR) are large and may
not install successfully on Streamlit Cloud. For local development this
works, but for Streamlit Deploy consider using the lighter Demo mode or
deploy the backend separately and set `API_URL`.

Run with backend (local developer):

If you want the full backend (FastAPI + CV models), install the backend
dependencies (DeepFace, EasyOCR, TensorFlow, etc.) and run the backend in
a separate terminal, then point `API_URL` at it:

```powershell
# install backend deps (not included in trimmed requirements)
# pip install -r requirements-backend.txt  # if you restore it
set API_URL=http://localhost:8000
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Streamlit Deploy (recommended)

This repository is trimmed for Streamlit Deploy. The app entrypoint is
`streamlit_app.py` which runs the frontend at `frontend/app.py`.

Files kept for Streamlit Deploy:
- `streamlit_app.py` — root entrypoint
- `requirements.txt` — Python deps for the frontend
- `frontend/` — Streamlit UI (`frontend/app.py`)
- `DEPLOYMENT.md` — this document

Local run (developer machine):
```powershell
pip install -r requirements.txt
set API_URL=http://localhost:8000
streamlit run streamlit_app.py
```

Streamlit Cloud setup:
- In the Streamlit app settings, set the `API_URL` secret to your
        backend URL (e.g. `https://your-backend.example.com`) so the frontend
        can reach the FastAPI API.
- Push the repo to GitHub and point Streamlit Deploy to this repository.

Notes:
- The backend (`backend/`) is not deployed here — host it separately
        and set `API_URL` accordingly.
- If you need the full backend bundled, consider a different hosting
        strategy (Render, Railway, or a Docker-based service).

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
