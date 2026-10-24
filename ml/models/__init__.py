import torch

from .pretrained import build_pretrained
from .scratch_cnn import ScratchCNN


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """Create a model by architecture name from config."""
    if arch == "scratch_cnn":
        return ScratchCNN(num_classes=num_classes)
    return build_pretrained(arch, num_classes, pretrained=pretrained)


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by ml/train.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
