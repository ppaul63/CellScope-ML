"""Minimal Grad-CAM, shared by evaluation and the backend API."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "gradcam_layer"):
        return model.gradcam_layer()
    if hasattr(model, "bn2"):  # timm EfficientNet: last conv block before pooling
        return model.bn2
    raise ValueError(f"Don't know the Grad-CAM layer for {type(model).__name__}")


def gradcam(model, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Return an HxW heatmap in [0, 1] for a single (1, 3, H, W) input."""
    acts, grads = {}, {}
    layer = target_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        model.zero_grad()
        logits[0, idx].backward()
        weights = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap over the image."""
    import matplotlib

    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize(img.size, Image.BILINEAR)
    return Image.blend(img.convert("RGB"), heat, alpha)
