"""Stage 1: benign vs malignant, four architectures.
Run:  python -m src.stage1_binary.train --config config.yaml [--model resnet50 ...] [--device auto]
Uses ONLY binary_train.csv and binary_val.csv. The test manifest is not opened here.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.cli import select_device
from src.common.config import config_sha256, git_commit, load_config, resolve_path
from src.common.manifest import verify_manifest
from src.training.trainer import TrainSpec, run_training

CLASSES = ["benign", "malignant"]


def load_split(manifest_dir, part: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_dir / f"binary_{part}.csv", dtype={"image_id": str, "patient_id": str})
    df["target"] = df["y_malignant"].map({0: "benign", 1: "malignant"})
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", action="append", help="config key; repeat for several; default all")
    ap.add_argument("--device", default=None, help="overrides training.device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest_dir = resolve_path(cfg["data"]["manifest_dir"])
    summary = verify_manifest(manifest_dir)
    raw = resolve_path(cfg["data"]["raw_dir"])
    train_df, val_df = load_split(manifest_dir, "train"), load_split(manifest_dir, "val")
    device = select_device(args.device or cfg["training"]["device"])
    models = cfg["training"]["models"]
    names = args.model or list(models)
    meta_base = {"stage": "stage1_binary", "config_sha256": config_sha256(cfg), "git_commit": git_commit(),
                 "manifest_sha256": summary["manifest_sha256"]}
    for name in names:
        if name not in models:
            raise SystemExit(f"Unknown model {name}; choose from {list(models)}")
        spec = TrainSpec(model_name=name, timm_name=models[name]["timm"], batch_size=models[name]["batch_size"],
                         classes=CLASSES, label_col="target", task="binary",
                         weights_dir=resolve_path(cfg["stage1"]["weights_dir"]),
                         runs_dir=resolve_path(cfg["stage1"]["runs_dir"]))
        meta = run_training(spec, train_df=train_df, val_df=val_df, raw_root=raw, recipe=cfg["training"],
                            device=device, seed=cfg["project"]["seed"], meta_base=meta_base)
        print(f"{name}: best epoch {meta['best_epoch']} val ROC-AUC {meta['best_val_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
