"""Image dataset and transforms shared by training, evaluation and the web app."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, augment: dict, train: bool) -> T.Compose:
    """Train: flips, rotation, mild colour jitter (training split only). Eval: resize + normalize only."""
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ops: list = [T.Resize((image_size, image_size))]
    if train:
        ops.append(T.RandomHorizontalFlip())
        if augment.get("vertical_flip", False):
            ops.append(T.RandomVerticalFlip())
        ops.append(T.RandomRotation(augment["rotate_degrees"]))
        ops.append(T.ColorJitter(**augment["color_jitter"]))
    ops += [T.ToTensor(), normalize]
    return T.Compose(ops)


class ImageDataset(Dataset):
    """Rows of a manifest. `path` is relative to the raw data folder."""

    def __init__(self, frame: pd.DataFrame, raw_root: Path, transform, label_col: str, class_index: dict[str, int]):
        self.frame = frame.reset_index(drop=True)
        self.raw_root = Path(raw_root)
        self.transform = transform
        self.label_col = label_col
        self.class_index = class_index
        unknown = set(self.frame[label_col]) - set(class_index)
        if unknown:
            raise ValueError(f"Labels not in class list: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        with Image.open(self.raw_root / row["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.class_index[row[self.label_col]]
