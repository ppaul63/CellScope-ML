"""Our own ResNet-style CNN with Squeeze-Excitation, trained from random weights."""
import torch.nn as nn


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch),
            SqueezeExcite(out_ch),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class ScratchCNN(nn.Module):
    def __init__(self, num_classes=5, widths=(40, 80, 160, 320), blocks=(2, 2, 2, 2), dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 7, 2, 3, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        layers, in_ch = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(ResidualBlock(in_ch, w, stride=2 if (j == 0 and i > 0) else 1))
                in_ch = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each residual branch so blocks start as identity
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.body[4].weight)

    def forward(self, x):
        return self.head(self.features(self.stem(x)))

    def gradcam_layer(self):
        return self.features[-1]
