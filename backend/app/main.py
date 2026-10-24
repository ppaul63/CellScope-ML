"""CellScope ML: FastAPI service hosting multiple health-detection models."""
import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .auth import current_user, router as auth_router
from .db import init_db
from .registry import ModelRegistry, build_registry
from .schemas import ClassInfo, ModelSummary, Prediction
from .training import router as training_router

logging.basicConfig(level=logging.INFO)
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
Image.MAX_IMAGE_PIXELS = 60_000_000  # guard against decompression bombs


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.registry = build_registry()
    app.state.registry.load_all()
    yield


app = FastAPI(title="CellScope ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(auth_router)
app.include_router(training_router)


def registry() -> ModelRegistry:
    return app.state.registry


def summarize(model_id: str) -> ModelSummary:
    reg = registry()
    spec = reg.spec(model_id)
    predictor = reg.predictor(model_id, required=False)
    classes = [ClassInfo(code=code, name=c.get("name", code), description=c.get("description", ""), risk=c.get("risk", "Unknown"))
               for code, c in spec.get("classes", {}).items()]
    summary = ModelSummary(id=model_id, name=spec["name"], category=spec.get("category", "General"),
                           modality=spec.get("modality", ""), summary=spec.get("summary", ""),
                           input_hint=spec.get("input_hint", "Upload an image."),
                           status="ready" if predictor else "not_trained", classes=classes)
    if predictor:
        info = predictor.info
        summary.arch, summary.model_kind, summary.image_size = info["arch"], info["model_kind"], info["image_size"]
        summary.heatmap_available = predictor.heatmap_available
        summary.test = info.get("test") or None
    return summary


@app.get("/api/health")
def health():
    reg = registry()
    ready = [m for m in reg.specs if reg.predictor(m, required=False)]
    return {"status": "ok", "models_registered": len(reg.specs), "models_ready": len(ready)}


@app.get("/api/models", response_model=list[ModelSummary], dependencies=[Depends(current_user)])
def list_models():
    return [summarize(m) for m in registry().specs]


@app.get("/api/models/{model_id}", response_model=ModelSummary, dependencies=[Depends(current_user)])
def get_model(model_id: str):
    return summarize(model_id)


@app.post("/api/models/{model_id}/predict", response_model=Prediction, dependencies=[Depends(current_user)])
async def predict(model_id: str, file: UploadFile = File(...), heatmap: bool = Query(True)):
    predictor = registry().predictor(model_id)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not read image. Upload a JPG, PNG, BMP or TIFF file.")
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format {img.format}. Use JPG, PNG, BMP or TIFF.")
    # Uploaded images are processed in memory and never written to disk.
    return {"model_id": model_id, **predictor.predict(img, heatmap=heatmap)}
