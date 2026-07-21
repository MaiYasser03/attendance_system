import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import utils.logger  # noqa: F401
from services.cv_service import AttendanceCVService

cv_service: AttendanceCVService | None = None
_service_lock = threading.Lock()


def _get_or_create_service() -> AttendanceCVService:
    global cv_service
    if cv_service is not None:
        return cv_service
    with _service_lock:
        if cv_service is None:
            logging.info("Loading CV models (DeepFace, EasyOCR)...")
            cv_service = AttendanceCVService()
            logging.info("CV models ready")
    return cv_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.info("Backend started — CV models load on first request")
    yield


app = FastAPI(
    title="Attendance System API",
    description="Face recognition, emotion detection, and ID card OCR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "attendance-cv",
        "models_loaded": cv_service is not None,
    }


@app.post("/api/process-frame")
async def process_frame(
    file: Annotated[UploadFile, File(...)],
    run_face: bool = Query(True),
    run_ocr: bool = Query(False),
    mirror: bool = Query(True),
):
    service = _get_or_create_service()
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        result = service.process_frame(
            image_bytes,
            run_face=run_face,
            run_ocr=run_ocr,
            mirror=mirror,
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Frame processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ocr")
async def ocr_scan(file: Annotated[UploadFile, File(...)]):
    service = _get_or_create_service()
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        return service.run_ocr(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("OCR failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/attendance")
async def get_attendance():
    if cv_service is None:
        return {"count": 0, "records": []}
    records = cv_service.get_attendance()
    return {"count": len(records), "records": records}


@app.get("/api/analytics/emotion")
async def emotion_analytics():
    service = _get_or_create_service()
    chart_b64 = service.get_emotion_analytics_base64()
    if not chart_b64:
        raise HTTPException(status_code=404, detail="No attendance data for analytics")
    return {"chart_base64": chart_b64}


@app.post("/api/reset")
async def reset_session():
    if cv_service is not None:
        cv_service.reset_session()
    return {"status": "session reset"}
