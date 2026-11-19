"""Stage 2: benign subtype (bdb, cb, db, srjd, [other_benign]), four architectures, selection by val macro-F1.
Run:  python -m src.stage2_subtype.train --config config.yaml [--model resnet50 ...]
Uses ONLY benign_subtypes_train.csv and benign_subtypes_val.csv. Writes weights/stage2/weights.json with
per-class validation F1 for each model (used by the 'weighted' ensemble mode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.cli import select_device
from src.common.config import class_subtypes, config_sha256, git_commit, load_config, resolve_path
from src.common.manifest import verify_manifest
from src.training.trainer import TrainSpec, run_training


def load_split(manifest_dir: Path, part: str, classes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(manifest_dir / f"benign_subtypes_{part}.csv", dtype={"image_id": str, "patient_id": str})
    return df[df["subtype"].isin(classes)].reset_index(drop=True)


def update_weights_json(path: Path, classes: list[str], name: str, meta: dict, config_sha: str,
                        manifest_sha: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"classes": classes, "models": {}}
    if data["classes"] != classes:
        raise SystemExit(f"{path} was built for classes {data['classes']}, current config gives {classes}. "
                         "Delete it and retrain all models.")
    data["models"][name] = {"val_macro_f1": meta["best_val_score"], "best_epoch": meta["best_epoch"],
                            "val_f1_per_class": meta["val_f1_per_class"]}
    data.update({"selection_split": "val", "selection_metric": "macro_f1", "weights_metric": "per_class_f1",
                 "config_sha256": config_sha, "manifest_sha256": manifest_sha})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", action="append")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest_dir = resolve_path(cfg["data"]["manifest_dir"])
    summary = verify_manifest(manifest_dir)
    raw = resolve_path(cfg["data"]["raw_dir"])
    classes = class_subtypes(cfg, cfg["stage2"]["include_other_benign"])
    train_df = load_split(manifest_dir, "train", classes)
    val_df = load_split(manifest_dir, "val", classes)
    device = select_device(args.device or cfg["training"]["device"])
    models = cfg["training"]["models"]
    names = args.model or list(models)
    weights_dir = resolve_path(cfg["stage2"]["weights_dir"])
    config_sha, manifest_sha = config_sha256(cfg), summary["manifest_sha256"]
    meta_base = {"stage": "stage2_subtype", "config_sha256": config_sha, "git_commit": git_commit(),
                 "manifest_sha256": manifest_sha, "include_other_benign": cfg["stage2"]["include_other_benign"]}
    for name in names:
        if name not in models:
            raise SystemExit(f"Unknown model {name}; choose from {list(models)}")
        spec = TrainSpec(model_name=name, timm_name=models[name]["timm"], batch_size=models[name]["batch_size"],
                         classes=classes, label_col="subtype", task="subtype", weights_dir=weights_dir,
                         runs_dir=resolve_path(cfg["stage2"]["runs_dir"]))
        meta = run_training(spec, train_df=train_df, val_df=val_df, raw_root=raw, recipe=cfg["training"],
                            device=device, seed=cfg["project"]["seed"], meta_base=meta_base)
        update_weights_json(weights_dir / "weights.json", classes, name, meta, config_sha, manifest_sha)
        print(f"{name}: best epoch {meta['best_epoch']} val macro-F1 {meta['best_val_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
