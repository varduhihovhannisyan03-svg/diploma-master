"""Batch probability extraction for evaluation (no labels needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _PathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


@torch.inference_mode()
def predict_probs(model: torch.nn.Module, frame: pd.DataFrame, raw_root: Path, transform,
                  batch_size: int, device: torch.device) -> np.ndarray:
    paths = [Path(raw_root) / p for p in frame["path"]]
    loader = DataLoader(_PathDataset(paths, transform), batch_size=batch_size, shuffle=False)
    model.eval()
    out = [torch.softmax(model(x.to(device)), dim=1).cpu().numpy() for x in loader]
    return np.concatenate(out) if out else np.zeros((0, 0))
