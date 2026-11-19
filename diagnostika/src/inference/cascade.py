"""Cascade decision for one lesion: referral, unclear flag, agreement, and the probability table."""
from __future__ import annotations

import numpy as np

from src.inference.ensemble import final_distribution


def decide(p_malignant: float, ensemble_benign: np.ndarray, per_model_benign: np.ndarray,
           classes: list[str], threshold: float, unclear_below: float) -> dict:
    """per_model_benign: (M, K) renormalized per-model subtype probabilities."""
    final = final_distribution(p_malignant, ensemble_benign)
    if abs(float(final.sum()) - 1.0) > 1e-9:
        raise AssertionError("final distribution does not sum to 1")
    labels = ["malignant"] + list(classes)
    top_benign = int(np.argmax(ensemble_benign))
    agreement = int(sum(int(np.argmax(row)) == top_benign for row in per_model_benign))
    referral = bool(p_malignant >= threshold)
    return {
        "probabilities": {lab: float(v) for lab, v in zip(labels, final)},
        "top_class": labels[int(np.argmax(final))],
        "top_benign": classes[top_benign],
        "agreement": agreement,
        "n_models": int(per_model_benign.shape[0]),
        "referral": referral,
        "subtype_for_information_only": referral,
        "unclear": bool(float(final.max()) < unclear_below),
        "threshold": float(threshold),
    }
