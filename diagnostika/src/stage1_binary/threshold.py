"""Choose the stage 1 malignant threshold on the VALIDATION split only.

Rule: the largest threshold t whose validation sensitivity (P(p >= t | malignant)) is >= target.
Largest t gives the highest specificity among the thresholds that meet the sensitivity target.
The test manifest and test predictions are never read here. A clinician must review the target and result.

Run:  python -m src.stage1_binary.threshold --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.config import config_sha256, git_commit, load_config, resolve_path
from src.common.manifest import verify_manifest


def choose_threshold(y_true, p_malignant, target_sensitivity: float) -> dict:
    """Pure function: takes validation labels and probabilities only."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_malignant, dtype=float)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities differ in length")
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("validation split needs both classes to choose a threshold")
    chosen = None
    for t in np.unique(p)[::-1]:  # descending: first feasible value is the largest one
        if np.mean(pos >= t) >= target_sensitivity:
            chosen = float(t)
            break
    if chosen is None:  # cannot happen for target <= 1 (lowest score gives sensitivity 1); guard anyway
        raise RuntimeError("No feasible threshold")
    return {
        "threshold": chosen,
        "target_sensitivity": float(target_sensitivity),
        "val_sensitivity": float(np.mean(pos >= chosen)),
        "val_specificity": float(np.mean(neg < chosen)),
        "val_roc_auc": float(roc_auc_score(y, p)),
        "n_val": int(len(y)),
        "n_val_malignant": int(len(pos)),
        "n_val_benign": int(len(neg)),
        "rule": "largest threshold with validation sensitivity >= target",
    }


def load_ensemble_val(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    models = list(cfg["training"]["models"])
    runs = resolve_path(cfg["stage1"]["runs_dir"])
    frames = []
    for name in models:
        path = runs / name / "val_predictions.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Train stage 1 for {name} first.")
        df = pd.read_csv(path, dtype={"image_id": str})
        frames.append(df[["image_id", "target", "prob_malignant"]].rename(columns={"prob_malignant": name}))
    merged = frames[0]
    for f in frames[1:]:
        if set(f["image_id"]) != set(merged["image_id"]):
            raise SystemExit("Models were not validated on the same images")
        merged = merged.merge(f[["image_id", f.columns[-1]]], on="image_id", validate="one_to_one")
    merged["p_malignant_ensemble"] = merged[models].mean(axis=1)
    return merged, models


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    summary = verify_manifest(resolve_path(cfg["data"]["manifest_dir"]))
    merged, models = load_ensemble_val(cfg)
    y = (merged["target"] == "malignant").astype(int).to_numpy()
    result = choose_threshold(y, merged["p_malignant_ensemble"].to_numpy(), cfg["stage1"]["target_sensitivity"])
    out = {
        **result,
        "ensemble_mode_stage1": "average of the four models' p_malignant",
        "models": models,
        "selection_split": "val",
        "clinician_review_required": True,
        "manifest_sha256": summary["manifest_sha256"],
        "config_sha256": config_sha256(cfg),
        "git_commit": git_commit(),
    }
    path = resolve_path(cfg["inference"]["stage1_threshold_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
