"""ImageNet-pretrained baseline from timm."""
import timm
import torch.nn as nn


def build_pretrained(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=0.3)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head = model.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable
