from typing import Literal

from pydantic import BaseModel, ConfigDict


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow "model_id"
    model_id: str
    predicted_class: str
    class_name: str
    description: str
    confidence: float
    probabilities: dict[str, float]
    risk_level: str
    heatmap_png_base64: str | None = None
    inference_ms: float


class ClassInfo(BaseModel):
    code: str
    name: str
    description: str
    risk: str


class ModelSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow "model_kind"
    id: str
    name: str
    category: str
    modality: str
    summary: str
    input_hint: str
    status: Literal["ready", "not_trained"]
    classes: list[ClassInfo]
    arch: str | None = None
    model_kind: str | None = None
    image_size: int | None = None
    heatmap_available: bool = False
    test: dict | None = None
