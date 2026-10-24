"""Registry of detection models: metadata from models.yaml, weights from artifacts/models/<id>/."""
import logging
import os
import threading
from pathlib import Path

import yaml
from fastapi import HTTPException

from .inference import Predictor

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class ModelRegistry:
    def __init__(self, registry_file: Path, artifacts_root: Path):
        self.artifacts_root = artifacts_root
        specs = yaml.safe_load(registry_file.read_text()).get("models", [])
        self.specs: dict[str, dict] = {s["id"]: s for s in specs}
        self._predictors: dict[str, Predictor] = {}
        self._lock = threading.Lock()

    def model_dir(self, model_id: str) -> Path:
        return self.artifacts_root / "models" / model_id

    def spec(self, model_id: str) -> dict:
        if model_id not in self.specs:
            raise HTTPException(404, f"Unknown model '{model_id}'.")
        return self.specs[model_id]

    def predictor(self, model_id: str, required: bool = True) -> Predictor | None:
        """Load a model on first use, so newly exported models appear without a restart."""
        spec = self.spec(model_id)
        with self._lock:
            if model_id not in self._predictors:
                try:
                    self._predictors[model_id] = Predictor(self.model_dir(model_id), spec.get("classes", {}))
                except FileNotFoundError:
                    if required:
                        raise HTTPException(503, f"{spec['name']} has not been trained yet. "
                                                 "Train it and run: python -m ml.export")
                    return None
            return self._predictors[model_id]

    def load_all(self) -> None:
        for model_id in self.specs:
            if self.predictor(model_id, required=False) is None:
                log.warning("Model '%s' has no exported files in %s", model_id, self.model_dir(model_id))


def build_registry() -> ModelRegistry:
    registry_file = Path(os.environ.get("MODEL_REGISTRY", ROOT / "models.yaml"))
    artifacts_root = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
    return ModelRegistry(registry_file, artifacts_root)
