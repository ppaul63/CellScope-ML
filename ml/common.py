"""Helpers shared by the ML scripts."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    # CANCERDET_CONFIG lets tests/experiments point at a different config file
    path = path or os.environ.get("CANCERDET_CONFIG") or ROOT / "ml" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_norm_stats(cfg: dict) -> tuple[list[float], list[float]]:
    """Mean/std computed by prepare_data.py; ImageNet stats as fallback."""
    path = resolve(cfg["data"]["splits_dir"]) / "norm_stats.json"
    if path.exists():
        stats = json.loads(path.read_text())
        return stats["mean"], stats["std"]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
