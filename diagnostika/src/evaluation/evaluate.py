"""Final test evaluation. Run ONCE after the configuration is frozen:
    python -m src.evaluation.evaluate --config config.yaml
Writes runs/evaluation/<run_id>/metrics.json and predictions.csv, plus reports/results.md.
Test images are used only here. Threshold and ensemble settings come from validation-trained artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.cli import select_device
from src.common.config import class_subtypes, config_sha256, git_commit, load_config, resolve_path
from src.common.manifest import verify_manifest
from src.evaluation.metrics import binary_block, bootstrap_by_patient, subtype_block
from src.inference.batch import predict_probs
from src.inference.ensemble import ensemble_benign, final_distribution, renormalize_benign
from src.training.dataset import build_transforms
from src.training.weights_io import load_models


def _read_test(manifest_dir: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_dir / "binary_test.csv", dtype={"image_id": str, "patient_id": str})


def _check_consistency(cfg: dict, manifest_sha: str, config_sha: str, allow_drift: bool) -> None:
    thr = json.loads(resolve_path(cfg["inference"]["stage1_threshold_file"]).read_text(encoding="utf-8"))
    if thr["manifest_sha256"] != manifest_sha:
        raise SystemExit("threshold.json was made from a different manifest; rerun src.stage1_binary.threshold")
    for stage_key in ("stage1", "stage2"):
        for name in cfg["training"]["models"]:
            meta_path = resolve_path(cfg[stage_key]["weights_dir"]) / f"{name}.meta.json"
            if not meta_path.exists():
                raise SystemExit(f"Missing {meta_path}; train the model first")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["manifest_sha256"] != manifest_sha:
                raise SystemExit(f"{meta_path.name} was trained on a different manifest")
            if meta["config_sha256"] != config_sha and not allow_drift:
                raise SystemExit(f"Config changed after training {stage_key}/{name} "
                                 f"(trained {meta['config_sha256'][:12]}, now {config_sha[:12]}). "
                                 "Retrain, or pass --allow-config-drift and document it.")


def _previous_runs(runs_dir: Path, config_sha: str, manifest_sha: str) -> list[str]:
    found = []
    if runs_dir.exists():
        for p in runs_dir.glob("*/metrics.json"):
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("config_sha256") == config_sha and m.get("manifest_sha256") == manifest_sha:
                found.append(m["run_id"])
    return found


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-config-drift", action="store_true")
    ap.add_argument("--force-rerun", action="store_true", help="allow another evaluation with the same config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest_dir = resolve_path(cfg["data"]["manifest_dir"])
    summary = verify_manifest(manifest_dir)
    manifest_sha, config_sha = summary["manifest_sha256"], config_sha256(cfg)
    runs_dir = resolve_path(cfg["evaluation"]["runs_dir"])
    prev = _previous_runs(runs_dir, config_sha, manifest_sha)
    if prev and not args.force_rerun:
        raise SystemExit(f"Evaluation already ran with this config and manifest: {prev}. Use --force-rerun "
                         "only if you have a documented reason; the test set should be looked at once.")
    _check_consistency(cfg, manifest_sha, config_sha, args.allow_config_drift)

    device = select_device(args.device or cfg["training"]["device"])
    bundle = load_models(cfg, device)
    raw = resolve_path(cfg["data"]["raw_dir"])
    test = _read_test(manifest_dir)
    classes = bundle.classes
    K = len(classes)
    other = cfg["data"]["other_benign_label"]
    t = bundle.threshold
    names = bundle.models
    tf_eval = build_transforms(cfg["training"]["image_size"], cfg["training"]["augment"], train=False)
    bs = cfg["training"]["models"]

    p1, p2 = [], []
    for n in names:
        p1.append(predict_probs(bundle.stage1[n], test, raw, tf_eval, bs[n]["batch_size"], device)[:, 1])
        p2.append(predict_probs(bundle.stage2[n], test, raw, tf_eval, bs[n]["batch_size"], device))
    P1 = np.stack(p1)                       # (M, N)
    PB = renormalize_benign(np.stack(p2))   # (M, N, K)
    E = ensemble_benign(PB, bundle.mode, bundle.weight_matrix)  # (N, K)
    p_ens = P1.mean(axis=0)
    final = final_distribution(p_ens, E)

    y = test["y_malignant"].to_numpy().astype(int)
    subtype = test["subtype"].to_numpy()
    sub_idx = np.array([classes.index(s) if s in classes else -1 for s in subtype])
    referral = p_ens >= t
    top_benign = E.argmax(axis=1)
    per_model_top = PB.argmax(axis=2)  # (M, N)
    other_present = bool(((subtype == other) & (y == 0)).any()) and other not in classes

    def build(idx: np.ndarray) -> dict:
        out: dict = {"stage1": {}, "stage2": {}, "cascade": {}}
        for m, n in enumerate(names):
            out["stage1"][n] = binary_block(y[idx], P1[m, idx], t)
        out["stage1"]["ensemble"] = binary_block(y[idx], p_ens[idx], t)
        sel = idx[(y[idx] == 0) & (sub_idx[idx] >= 0)]
        for m, n in enumerate(names):
            out["stage2"][n] = subtype_block(sub_idx[sel], per_model_top[m, sel], classes)
        out["stage2"]["ensemble"] = subtype_block(sub_idx[sel], top_benign[sel], classes)
        benign_ok = np.where(sub_idx[idx] >= 0, (~referral[idx]) & (top_benign[idx] == sub_idx[idx]),
                             (~referral[idx]) & (y[idx] == 0))
        correct = np.where(y[idx] == 1, referral[idx], benign_ok)
        per_class = {}
        labels_row = np.where(y[idx] == 1, "malignant", subtype[idx])
        for lab in ["malignant"] + classes + ([other] if other_present else []):
            mask = labels_row == lab
            per_class[lab] = {"n": int(mask.sum()),
                              "accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
                              "referral_rate": float(referral[idx][mask].mean()) if mask.any() else math.nan}
        out["cascade"] = {"overall_accuracy": float(correct.mean()) if len(idx) else math.nan,
                          "per_class": per_class}
        return out

    everything = np.arange(len(test))
    point = build(everything)
    # Lowest priority: the confusion lists are not numeric, so they are skipped in the bootstrap.
    ci = bootstrap_by_patient(test["patient_id"].to_numpy(), build,
                              cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_seed"])

    warnings = [
        "Threshold chosen on the validation split only; it needs clinician review before any use.",
        "Stage 2 uses benign test images only; stage 1 uses all binary test images.",
    ]
    small = cfg["evaluation"]["small_support_warning"]
    for c in classes:
        n_c = int((sub_idx == classes.index(c)).sum())  # benign test rows only (malignant rows map to -1)
        if n_c < small:
            warnings.append(f"Class {c} has only n={n_c} test images; its estimates are unstable.")
    if other_present:
        warnings.append(f"{other} is not modelled in stage 2 (include_other_benign=false); "
                        "these benign rows count as correct only when not referred.")

    run_id = f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{git_commit()[:7]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "threshold": bundle.threshold_info,
        "ensemble_mode": bundle.mode,
        "models": names,
        "classes": classes,
        "include_other_benign": cfg["stage2"]["include_other_benign"],
        "counts": {
            "test_images": int(len(test)),
            "test_malignant": int((y == 1).sum()),
            "test_benign": int((y == 0).sum()),
            "test_patients": int(test["patient_id"].nunique()),
            "stage2_benign_rows": int(((y == 0) & (sub_idx >= 0)).sum()),
            "per_class_test_support": {c: int((sub_idx == i).sum()) for i, c in enumerate(classes)},
        },
        "metrics": point,
        "bootstrap": {"unit": "patient", "resamples": cfg["evaluation"]["bootstrap_resamples"],
                      "seed": cfg["evaluation"]["bootstrap_seed"], "ci95": ci},
        "warnings": warnings,
    }
    (run_dir / "metrics.json").write_text(_dump(result), encoding="utf-8")
    pred = test[["image_id", "patient_id", "y_malignant", "subtype"]].copy()
    for m, n in enumerate(names):
        pred[f"p_malignant_{n}"] = P1[m]
    pred["p_malignant_ensemble"] = p_ens
    for j, lab in enumerate(["malignant"] + classes):
        pred[f"final_{lab}"] = final[:, j]
    pred["referral"] = referral
    pred["top_benign_ensemble"] = [classes[i] for i in top_benign]
    pred.to_csv(run_dir / "predictions.csv", index=False)
    report = resolve_path(cfg["data"]["reports_dir"])
    report.mkdir(parents=True, exist_ok=True)
    (report / "results.md").write_text(to_markdown(result), encoding="utf-8")
    print(f"Wrote {run_dir}")


def _dump(obj) -> str:
    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(obj), indent=2, ensure_ascii=False)


def _fmt(v, low=None, high=None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:.3f}"
    if low is not None and high is not None and not (math.isnan(low) or math.isnan(high)):
        s += f" [{low:.3f}; {high:.3f}]"
    return s


def to_markdown(r: dict) -> str:
    m, ci = r["metrics"], r["bootstrap"]["ci95"]
    lines = [f"# Results — {r['run_id']}", "",
             f"Config `{r['config_sha256'][:12]}`, manifest `{r['manifest_sha256'][:12]}`, "
             f"ensemble mode `{r['ensemble_mode']}`, threshold {r['threshold']['threshold']:.4f}. "
             f"95% CI: patient bootstrap, {r['bootstrap']['resamples']} resamples.", "",
             "## Stage 1 — benign vs malignant (test)", "",
             "| Model | Sensitivity | Specificity | ROC-AUC | PR-AUC |", "|---|---|---|---|---|"]
    for n, b in m["stage1"].items():
        lines.append(f"| {n} | " + " | ".join(
            _fmt(b[k], *_ci(ci, f"stage1.{n}.{k}")) for k in ("sensitivity", "specificity", "roc_auc", "pr_auc")) + " |")
    lines += ["", "## Stage 2 — benign subtype (benign test images)", "",
              "| Model | Macro-F1 | Accuracy | " + " | ".join(f"Recall {c}" for c in r["classes"]) + " |",
              "|---|---|---|" + "---|" * len(r["classes"])]
    for n, b in m["stage2"].items():
        lines.append(f"| {n} | {_fmt(b['macro_f1'], *_ci(ci, f'stage2.{n}.macro_f1'))} | {_fmt(b['accuracy'])} | "
                     + " | ".join(_fmt(b["per_class_recall"][c]) for c in r["classes"]) + " |")
    lines += ["", "## Cascade end-to-end (all binary test images)", "",
              f"Overall accuracy: {_fmt(m['cascade']['overall_accuracy'], *_ci(ci, 'cascade.overall_accuracy'))}", "",
              "| Class | n | Accuracy | Referral rate |", "|---|---|---|---|"]
    for lab, b in m["cascade"]["per_class"].items():
        lines.append(f"| {lab} | {b['n']} | {_fmt(b['accuracy'])} | {_fmt(b['referral_rate'])} |")
    lines += ["", "## Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    return "\n".join(lines)


def _ci(ci: dict, key: str):
    v = ci.get(key)
    return (v["low"], v["high"]) if v else (None, None)


if __name__ == "__main__":
    main()
