import base64
import os
import time
from io import BytesIO

import av
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PROCESS_INTERVAL_SEC = float(os.getenv("PROCESS_INTERVAL_SEC", "2.5"))

st.set_page_config(
    page_title="Attendance System",
    page_icon="📷",
    layout="wide",
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def api_post(endpoint: str, image_bytes: bytes, params: dict | None = None) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            params=params or {},
            timeout=120,
        )
        response.raise_for_status()
        st.session_state.pop("last_error", None)
        return response.json()
    except requests.RequestException as exc:
        st.session_state["last_error"] = str(exc)
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        st.session_state.pop("last_error", None)
        return response.json()
    except requests.RequestException as exc:
        st.session_state["last_error"] = str(exc)
        return None


class AttendanceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_process_time = 0.0
        self.latest_result: dict = {}
        self.backend_ok = True

    def _call_backend(self, image_bytes: bytes) -> dict | None:
        try:
            response = requests.post(
                f"{API_URL}/api/process-frame",
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                params={"run_face": True, "run_ocr": False, "mirror": True},
                timeout=120,
            )
            response.raise_for_status()
            self.backend_ok = True
            return response.json()
        except requests.RequestException:
            self.backend_ok = False
            return None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_process_time >= PROCESS_INTERVAL_SEC:
            self.last_process_time = now
            _, buffer = cv2.imencode(".jpg", img)
            result = self._call_backend(buffer.tobytes())
            if result:
                self.latest_result = result

        display = img.copy()
        result = self.latest_result

        if not self.backend_ok:
            cv2.putText(
                display,
                "Backend offline - run start.bat",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        auth = result.get("authenticated", False)
        color = (0, 255, 0) if auth else (0, 0, 255)
        cv2.putText(
            display,
            f"Auth: {auth}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        if result.get("identity"):
            cv2.putText(
                display,
                f"Identity: {result['identity']}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
        if result.get("name"):
            cv2.putText(
                display,
                f"Name: {result['name']}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
        if result.get("id"):
            cv2.putText(
                display,
                f"ID: {result['id']}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
        if result.get("emotion"):
            cv2.putText(
                display,
                f"Emotion: {result['emotion']}",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

        return av.VideoFrame.from_ndarray(display, format="bgr24")


def render_status_panel():
    st.subheader("Live Status")
    result = st.session_state.get("cv_result", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Authenticated", "Yes" if result.get("authenticated") else "No")
    col2.metric("Faces Detected", result.get("faces_detected", 0))
    col3.metric("Identity", result.get("identity") or "—")
    col4.metric("Emotion", result.get("emotion") or "—")

    if result.get("name") or result.get("id"):
        st.info(f"**Name:** {result.get('name') or '—'}  |  **ID:** {result.get('id') or '—'}")

    if result.get("attendance_logged"):
        st.success("Attendance logged successfully.")


def render_attendance_table():
    data = api_get("/api/attendance")
    if not data or not data.get("records"):
        st.info("No attendance records yet.")
        return

    df = pd.DataFrame(data["records"])
    st.dataframe(df, use_container_width=True, hide_index=True)


st.title("Attendance System")
st.caption("Face recognition, ID card OCR, and emotion-based attendance logging")

health = api_get("/api/health")
if health:
    st.sidebar.success(f"Backend connected ({API_URL})")
    if not health.get("models_loaded"):
        st.sidebar.info("CV models will load on first scan (may take ~1 min).")
else:
    st.error(
        f"**Backend is not running** at `{API_URL}`.\n\n"
        "Start both services by double-clicking **`start.bat`** in the project folder, "
        "or run in PowerShell:\n```\n.\\start.ps1\n```"
    )
    st.stop()

st.sidebar.markdown("### Controls")
if st.sidebar.button("Reset Session"):
    try:
        requests.post(f"{API_URL}/api/reset", timeout=10)
        st.session_state.pop("cv_result", None)
        st.session_state.pop("ocr_result", None)
        st.session_state.pop("last_error", None)
        st.sidebar.success("Session reset")
    except requests.RequestException as exc:
        st.sidebar.error(str(exc))

if st.sidebar.button("Generate Emotion Chart"):
    chart_data = api_get("/api/analytics/emotion")
    if chart_data and chart_data.get("chart_base64"):
        image = Image.open(BytesIO(base64.b64decode(chart_data["chart_base64"])))
        st.sidebar.image(image, caption="Emotion Analytics")

tab_live, tab_id, tab_records = st.tabs(["Live Camera", "Scan ID Card", "Attendance Records"])

with tab_live:
    st.markdown(
        "Allow camera access when prompted. Face recognition runs automatically every "
        f"{PROCESS_INTERVAL_SEC:.1f}s."
    )
    render_status_panel()

    ctx = webrtc_streamer(
        key="attendance-camera",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=AttendanceVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.caption("Camera active — hold your face in view for authentication.")

with tab_id:
    st.markdown(
        "Hold your **ID card** up to the camera and capture a photo. "
        "Then authenticate with your face on the Live Camera tab."
    )

    id_photo = st.camera_input("Capture ID card")
    if id_photo is not None:
        if st.button("Run ID OCR", type="primary"):
            with st.spinner("Reading ID card (first run loads models, ~1 min)..."):
                result = api_post("/api/ocr", id_photo.getvalue())
            if result:
                st.session_state["ocr_result"] = result
                if result.get("parsed"):
                    st.success(f"Detected **{result['name']}** — ID **{result['id']}**")
                else:
                    st.warning("OCR ran but could not parse name and ID. Raw text:")
                    st.code(result.get("ocr_text", ""))
            elif st.session_state.get("last_error"):
                st.error(f"OCR failed: {st.session_state['last_error']}")

    ocr_result = st.session_state.get("ocr_result")
    if ocr_result:
        st.json(ocr_result)

with tab_records:
    if st.button("Refresh Records"):
        st.rerun()
    render_attendance_table()
