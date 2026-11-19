"""Shared two-phase trainer for stage 1 (binary) and stage 2 (subtype).

Phase 1: backbone frozen, head only, `head_epochs` epochs at lr_head.
Phase 2: full fine-tune up to `finetune_epochs` at lr_finetune; early stop after `patience` epochs
         without improvement (patience restarts at the start of phase 2).
Checkpoint selection uses VALIDATION only: ROC-AUC of malignant probability (binary) or
macro-F1 (subtype). The test manifest is never read here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.common.config import seed_everything
from src.training.dataset import ImageDataset, build_transforms
from src.training.models import (build_model, class_weights, freeze_backbone,
                                 set_frozen_batchnorm_eval, unfreeze_all)


@dataclass
class TrainSpec:
    model_name: str          # config key, e.g. resnet50 / vit_b_16
    timm_name: str           # timm architecture
    batch_size: int
    classes: list[str]       # output order
    label_col: str           # column holding the class name
    task: str                # "binary" | "subtype"
    weights_dir: Path
    runs_dir: Path
    extra_meta: dict = field(default_factory=dict)


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    ys, probs, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(crit(logits, y))
            n += len(y)
            ys.append(y.cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs), loss_sum / max(n, 1)


def _score(task: str, y: np.ndarray, probs: np.ndarray, k: int) -> float:
    if task == "binary":
        return float(roc_auc_score(y, probs[:, 1]))
    return float(f1_score(y, probs.argmax(1), labels=list(range(k)), average="macro", zero_division=0))


def run_training(spec: TrainSpec, *, train_df: pd.DataFrame, val_df: pd.DataFrame, raw_root: Path,
                 recipe: dict, device: torch.device, seed: int, meta_base: dict) -> dict:
    seed_everything(seed)
    k = len(spec.classes)
    class_index = {c: i for i, c in enumerate(spec.classes)}
    if spec.task not in ("binary", "subtype"):
        raise ValueError(f"Unknown task: {spec.task}")

    train_ds = ImageDataset(train_df, raw_root, build_transforms(recipe["image_size"], recipe["augment"], True),
                            spec.label_col, class_index)
    val_ds = ImageDataset(val_df, raw_root, build_transforms(recipe["image_size"], recipe["augment"], False),
                          spec.label_col, class_index)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader_tr = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True,
                           num_workers=recipe["num_workers"], generator=gen)
    loader_va = DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False, num_workers=recipe["num_workers"])

    model = build_model(spec.timm_name, k, recipe["pretrained"]).to(device)
    weight = class_weights(train_df[spec.label_col].map(class_index).to_numpy(), k).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    spec.weights_dir.mkdir(parents=True, exist_ok=True)
    run_dir = spec.runs_dir / spec.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = spec.weights_dir / f"{spec.model_name}.pt"

    history: list[dict] = []
    state = {"best": -np.inf, "best_epoch": None, "bad": 0, "epoch": 0, "best_val": None}
    started = time.perf_counter()

    def run_phase(phase: str, epochs: int, lr: float) -> None:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                weight_decay=recipe["weight_decay"])
        for _ in range(epochs):
            state["epoch"] += 1
            model.train()
            if phase == "head":
                set_frozen_batchnorm_eval(model)
            total, n = 0.0, 0
            for x, y in loader_tr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss) * len(y)
                n += len(y)
            yv, pv, vloss = _predict(model, loader_va, device)
            score = _score(spec.task, yv, pv, k)
            row = {"epoch": state["epoch"], "phase": phase, "train_loss": total / max(n, 1),
                   "val_loss": vloss, f"val_{_metric_name(spec.task)}": score}
            history.append(row)
            print(f"[{spec.model_name}] {phase} {state['epoch']} train_loss={row['train_loss']:.4f} "
                  f"val_loss={vloss:.4f} val_{_metric_name(spec.task)}={score:.4f}", flush=True)
            if score > state["best"] + 1e-4:
                state.update(best=score, best_epoch=state["epoch"], bad=0, best_val=(yv, pv))
                torch.save(model.state_dict(), checkpoint)
            else:
                state["bad"] += 1
            if state["bad"] >= recipe["patience"]:
                print(f"[{spec.model_name}] early stopping", flush=True)
                return

    freeze_backbone(model)
    run_phase("head", recipe["head_epochs"], recipe["lr_head"])
    unfreeze_all(model)
    state["bad"] = 0
    if recipe["finetune_epochs"] > 0:
        run_phase("fine_tune", recipe["finetune_epochs"], recipe["lr_finetune"])

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    yv, pv = state["best_val"]
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pred = val_df.reset_index(drop=True)[["image_id", "patient_id", spec.label_col]].copy()
    for c, idx in class_index.items():
        pred[f"prob_{c}"] = pv[:, idx]
    pred.to_csv(run_dir / "val_predictions.csv", index=False)

    per_class_f1 = dict(zip(spec.classes, f1_score(yv, pv.argmax(1), labels=list(range(k)),
                                                    average=None, zero_division=0).astype(float).tolist()))
    meta = {
        **meta_base,
        "model": spec.model_name,
        "timm": spec.timm_name,
        "task": spec.task,
        "classes": spec.classes,
        "checkpoint": checkpoint.name,
        "batch_size": spec.batch_size,
        "selection_metric": _metric_name(spec.task),
        "best_val_score": float(state["best"]),
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "val_f1_per_class": {c: float(v) for c, v in per_class_f1.items()},
        "train_counts": {c: int(v) for c, v in train_df[spec.label_col].value_counts().items()},
        "val_counts": {c: int(v) for c, v in val_df[spec.label_col].value_counts().items()},
        "training_seconds": round(time.perf_counter() - started, 1),
        "device": str(device),
        "seed": seed,
        "recipe": recipe,
        **spec.extra_meta,
    }
    (spec.weights_dir / f"{spec.model_name}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _metric_name(task: str) -> str:
    return "roc_auc" if task == "binary" else "macro_f1"
