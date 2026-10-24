"""PyTorch dataset and augmentations for cytology images."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


class RandomRot90:
    """Cells have no canonical orientation, so any 90-degree rotation is valid."""

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return T.functional.rotate(img, 90 * k) if k else img


def build_transforms(size: int, mean, std, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            # Stain/scanner variation between labs
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(size * 1.08)),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


class CytologyDataset(Dataset):
    def __init__(self, csv_path, raw_dir, classes, transform):
        self.df = pd.read_csv(csv_path)
        self.raw_dir = raw_dir
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.targets = [self.class_to_idx[l] for l in self.df.label]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.raw_dir / self.df.path.iloc[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]
