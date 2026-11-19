"""Pure NumPy ensemble of benign subtype probabilities. Works on single rows or batches.

Shapes: per_model has shape (M, ..., K): M models, K benign classes (last axis).
  average : E = mean over models of the renormalized per-model probabilities.
  weighted: W[m, c] = validation per-class F1 of model m for class c, normalized across models per class
            (so for each class the M weights sum to 1; if all F1 are 0 for a class, uniform weights are used).
            E = sum over models of W * P, then renormalized over classes.
With equal weights, 'weighted' gives exactly the 'average' result.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def renormalize_benign(probs: np.ndarray) -> np.ndarray:
    """Renormalize over the benign classes (last axis) so each row sums to 1."""
    probs = np.asarray(probs, dtype=float)
    s = probs.sum(axis=-1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("A model gave zero total benign probability")
    return probs / s


def class_weight_matrix(weights: Mapping[str, Mapping[str, float]], models: Sequence[str],
                        classes: Sequence[str]) -> np.ndarray:
    """(M, K) matrix of per-class weights normalized across models."""
    raw = np.array([[float(weights[m][c]) for c in classes] for m in models], dtype=float)
    if np.any(raw < 0):
        raise ValueError("Negative ensemble weight")
    m = len(models)
    col = raw.sum(axis=0, keepdims=True)
    uniform = np.full_like(raw, 1.0 / m)
    return np.where(col > 0, raw / np.where(col > 0, col, 1.0), uniform)


def ensemble_benign(per_model: np.ndarray, mode: str, weight_matrix: np.ndarray | None = None) -> np.ndarray:
    P = renormalize_benign(per_model)
    m, k = P.shape[0], P.shape[-1]
    if mode == "average":
        E = P.mean(axis=0)
    elif mode == "weighted":
        if weight_matrix is None:
            raise ValueError("weighted mode needs a weight matrix")
        if weight_matrix.shape != (m, k):
            raise ValueError(f"weight matrix shape {weight_matrix.shape} != {(m, k)}")
        W = weight_matrix.reshape((m,) + (1,) * (P.ndim - 2) + (k,))
        E = (P * W).sum(axis=0)
    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")
    return E / E.sum(axis=-1, keepdims=True)


def final_distribution(p_malignant, ensemble: np.ndarray) -> np.ndarray:
    """[P(malignant), (1-p)*E_0, ..., (1-p)*E_{K-1}] — always sums to 1."""
    p = np.asarray(p_malignant, dtype=float)[..., None]
    return np.concatenate([p, (1.0 - p) * ensemble], axis=-1)
