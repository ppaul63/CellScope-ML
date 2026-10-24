"""Model loading and inference for the API."""
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

log = logging.getLogger(__name__)

class Predictor:
    def __init__(self, artifacts_dir: Path, class_details: dict[str, dict] | None = None):
        info_path = artifacts_dir / "model_info.json"
        onnx_path = artifacts_dir / "model.onnx"
        if not info_path.exists() or not onnx_path.exists():
            raise FileNotFoundError(f"No exported model in {artifacts_dir}")
        self.info = json.loads(info_path.read_text())
        self.class_details = class_details or {}
        self.classes = self.info["classes"]
        self.size = self.info["image_size"]
        self.mean = np.array(self.info["mean"], dtype=np.float32)
        self.std = np.array(self.info["std"], dtype=np.float32)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.torch_model = self._load_torch(artifacts_dir / "model.pt")

    def _load_torch(self, path: Path):
        """Optional PyTorch copy of the model, used only for Grad-CAM heatmaps."""
        try:
            root = Path(__file__).resolve().parents[2]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ml.models import load_checkpoint

            model, _ = load_checkpoint(path, "cpu")
            return model
        except Exception as e:
            log.warning("Grad-CAM disabled (%s)", e)
            return None

    @property
    def heatmap_available(self) -> bool:
        return self.torch_model is not None

    def preprocess(self, img: Image.Image) -> np.ndarray:
        # Mirrors the eval transform in ml/dataset.py: resize short side to 1.08x, center crop
        short = int(self.size * 1.08)
        w, h = img.size
        scale = short / min(w, h)
        img = img.resize((max(short, round(w * scale)), max(short, round(h * scale))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        x = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None]

    def predict(self, img: Image.Image, heatmap: bool = True) -> dict:
        t0 = time.perf_counter()
        img = img.convert("RGB")
        x = self.preprocess(img)
        logits = self.session.run(None, {"input": x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = int(probs.argmax())
        label = self.classes[idx]
        details = self.class_details.get(label, {})
        name, desc, risk = details.get("name", label), details.get("description", ""), details.get("risk", "Unknown")
        result = {
            "predicted_class": label, "class_name": name, "description": desc,
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "risk_level": risk, "heatmap_png_base64": None,
        }
        if heatmap and self.torch_model is not None:
            result["heatmap_png_base64"] = self._heatmap(x, idx)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _heatmap(self, x: np.ndarray, idx: int) -> str:
        import torch

        from ml.gradcam import gradcam, overlay

        cam = gradcam(self.torch_model, torch.from_numpy(x), idx)
        # Undo normalisation to show the exact crop the model saw
        crop = ((x[0].transpose(1, 2, 0) * self.std + self.mean).clip(0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        overlay(Image.fromarray(crop), cam).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
