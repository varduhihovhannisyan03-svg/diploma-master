"""Loading and checking trained weights for inference and evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.common.config import resolve_path
from src.inference.ensemble import class_weight_matrix
from src.training.models import build_model


class WeightsMissingError(RuntimeError):
    pass


@dataclass
class LoadedModels:
    models: list[str]
    stage1: dict[str, nn.Module]
    stage2: dict[str, nn.Module]
    classes: list[str]
    threshold: float
    threshold_info: dict
    weight_matrix: np.ndarray | None
    mode: str
    config_sha256: dict = field(default_factory=dict)


def required_files(cfg: dict) -> dict[str, Path]:
    s1 = resolve_path(cfg["stage1"]["weights_dir"])
    s2 = resolve_path(cfg["stage2"]["weights_dir"])
    files: dict[str, Path] = {}
    for name in cfg["training"]["models"]:
        files[f"stage 1 checkpoint {name}"] = s1 / f"{name}.pt"
        files[f"stage 2 checkpoint {name}"] = s2 / f"{name}.pt"
    files["stage 1 threshold"] = resolve_path(cfg["inference"]["stage1_threshold_file"])
    files["stage 2 weights.json"] = resolve_path(cfg["inference"]["stage2_weights_file"])
    return files


def check_weights(cfg: dict) -> None:
    missing = {k: str(v) for k, v in required_files(cfg).items() if not v.exists()}
    if missing:
        lines = "\n".join(f"  - {k}: {v}" for k, v in missing.items())
        raise WeightsMissingError(
            "Model weights are missing (weights/ is not in git).\n" + lines +
            "\nTrain and calibrate first, see README: src.stage1_binary.train, src.stage2_subtype.train, "
            "src.stage1_binary.threshold")


def load_models(cfg: dict, device: torch.device) -> LoadedModels:
    check_weights(cfg)
    names = list(cfg["training"]["models"])
    thr = json.loads(resolve_path(cfg["inference"]["stage1_threshold_file"]).read_text(encoding="utf-8"))
    w = json.loads(resolve_path(cfg["inference"]["stage2_weights_file"]).read_text(encoding="utf-8"))
    classes = list(w["classes"])
    missing_models = [n for n in names if n not in w["models"]]
    if missing_models:
        raise WeightsMissingError(f"weights.json has no entry for {missing_models}; retrain stage 2")
    mode = cfg["inference"]["ensemble_mode"]
    wm = None
    if mode == "weighted":
        wm = class_weight_matrix({n: w["models"][n]["val_f1_per_class"] for n in names}, names, classes)
    stage1, stage2 = {}, {}
    for name in names:
        spec = cfg["training"]["models"][name]
        stage1[name] = _load_ckpt(build_model(spec["timm"], 2, False),
                                  resolve_path(cfg["stage1"]["weights_dir"]) / f"{name}.pt", device)
        stage2[name] = _load_ckpt(build_model(spec["timm"], len(classes), False),
                                  resolve_path(cfg["stage2"]["weights_dir"]) / f"{name}.pt", device)
    return LoadedModels(models=names, stage1=stage1, stage2=stage2, classes=classes,
                        threshold=float(thr["threshold"]), threshold_info=thr, weight_matrix=wm, mode=mode)


def _load_ckpt(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
