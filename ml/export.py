"""Export the best model to ONNX for serving, into artifacts/models/<model_id>/.

Usage:  python -m ml.export                 # exports the winner from artifacts/metrics.json
        python -m ml.export --model scratch
"""
import argparse
import json
import shutil

import numpy as np
import onnxruntime as ort
import torch

from ml.common import load_config, resolve
from ml.models import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scratch", "pretrained"])
    args = ap.parse_args()

    cfg = load_config()
    out_dir = resolve(cfg["artifacts_dir"])
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    kind = args.model or metrics.get("winner")
    if kind is None:
        raise SystemExit("Run `python -m ml.evaluate` first or pass --model")

    model, ckpt = load_checkpoint(out_dir / f"{kind}_best.pt", "cpu")
    serve_dir = out_dir / "models" / cfg["model_id"]
    serve_dir.mkdir(parents=True, exist_ok=True)
    size = ckpt["image_size"]
    dummy = torch.randn(1, 3, size, size)
    onnx_path = serve_dir / "model.onnx"
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

    # Parity check: ONNX Runtime must agree with PyTorch
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    assert max_diff < 1e-3, f"ONNX mismatch: max abs diff {max_diff}"

    # Keep the PyTorch checkpoint next to it so the API can render Grad-CAM heatmaps
    shutil.copy(out_dir / f"{kind}_best.pt", serve_dir / "model.pt")
    model_info = {
        "model_id": cfg["model_id"], "model_kind": kind, "arch": ckpt["arch"], "classes": ckpt["classes"], "image_size": size,
        "mean": ckpt["mean"], "std": ckpt["std"], "val": ckpt.get("val"),
        "test": {k: v for k, v in metrics.get("models", {}).get(kind, {}).items() if k != "confusion_matrix"},
    }
    (serve_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))
    print(f"exported {kind} ({ckpt['arch']}) -> {onnx_path} (parity max diff {max_diff:.2e})")


if __name__ == "__main__":
    main()
