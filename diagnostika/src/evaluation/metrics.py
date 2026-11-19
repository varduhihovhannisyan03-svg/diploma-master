"""Metric helpers and patient-level bootstrap. All metric code takes row-index arrays so that the
point estimate and each bootstrap resample go through exactly the same computation."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)


def binary_block(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist() if len(y) else (0, 0, 0, 0)
    pos, neg = p[y == 1], p[y == 0]
    try:
        roc = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))
    except ValueError:
        roc, pr = math.nan, math.nan
    return {
        "sensitivity": float(np.mean(pos >= t)) if len(pos) else math.nan,
        "specificity": float(np.mean(neg < t)) if len(neg) else math.nan,
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def subtype_block(y_idx: np.ndarray, pred_idx: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    if len(y_idx) == 0:
        return {"macro_f1": math.nan, "accuracy": math.nan, "per_class_recall": {c: math.nan for c in classes},
                "confusion": [], "n": 0}
    labels = list(range(k))
    recall = {}
    cm = confusion_matrix(y_idx, pred_idx, labels=labels)
    for i, c in enumerate(classes):
        n_c = cm[i].sum()
        recall[c] = float(cm[i, i] / n_c) if n_c else math.nan
    return {
        "macro_f1": float(f1_score(y_idx, pred_idx, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_idx, pred_idx)),
        "per_class_recall": recall,
        "confusion": cm.tolist(),
        "n": int(len(y_idx)),
    }


def flatten_numeric(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def bootstrap_by_patient(patients: np.ndarray, build: Callable[[np.ndarray], dict], n_resamples: int,
                         seed: int) -> dict[str, dict]:
    """Resample PATIENTS with replacement; all images of a drawn patient are included (duplicated draws
    are included twice). Returns {metric: {low, high, n_valid}} at the 2.5th and 97.5th percentiles."""
    uniq, inverse = np.unique(patients, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[g]:bounds[g + 1]] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        draw = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[g] for g in draw])
        for key, value in flatten_numeric(build(idx)).items():
            samples.setdefault(key, []).append(value)
    ci = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            ci[key] = {"low": math.nan, "high": math.nan, "n_valid": 0}
        else:
            ci[key] = {"low": float(np.percentile(valid, 2.5)), "high": float(np.percentile(valid, 97.5)),
                       "n_valid": int(len(valid))}
    return ci
