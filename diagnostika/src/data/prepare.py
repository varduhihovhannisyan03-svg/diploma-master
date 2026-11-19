"""Build DERM12345 manifests. Run:  python -m src.data.prepare --config config.yaml

Order of operations (each step is recorded in manifest_summary.json):
  1. read both metadata files, pick columns, validate schema and patient IDs (stop on any problem)
  2. locate every image on disk (missing images stop the run)
  3. exclude malignancy == indeterminate (saved separately, never trained or evaluated)
  4. remove TRAIN images whose patient also appears in the official TEST split
  5. exact (SHA-256) + near (pHash) duplicate groups; keep one image per group, dropping train copies
     whenever a test copy exists
  6. stratified, patient-grouped validation split from the remaining official train (seed from config)
  7. write six split manifests, all_records.csv, duplicates.csv, excluded lists, manifest_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import config_sha256, git_commit, load_config, resolve_path
from src.common.manifest import SUMMARY_FILE, manifest_sha256
from src.data.dedup import choose_keepers, group_duplicates, phash_int, sha256_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class DataError(RuntimeError):
    pass


def _pick_column(df: pd.DataFrame, candidates: list[str], what: str, file_name: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise DataError(f"{file_name}: no column for {what} among {candidates}. Found: {list(df.columns)}")


def load_metadata(path: Path, cfg: dict, official_split: str) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"Metadata file missing: {path}")
    d = cfg["data"]
    cols = d["columns"]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rec = pd.DataFrame({
        "image_id": df[_pick_column(df, cols["image_id"], "image id", path.name)].str.strip(),
        "patient_id": df[_pick_column(df, cols["patient_id"], "patient id", path.name)].str.strip(),
        "label_raw": df[_pick_column(df, cols["label"], "label", path.name)].str.strip(),
        "malignancy": df[_pick_column(df, cols["malignancy"], "malignancy", path.name)].str.strip().str.lower(),
    })
    split_col = next((c for c in cols["split"] if c in df.columns), None)
    if split_col is not None:
        values = set(df[split_col].str.strip().str.lower())
        if not values <= {official_split}:
            raise DataError(f"{path.name}: column '{split_col}' has values {sorted(values)}, "
                            f"expected only '{official_split}'")
    if (rec["image_id"] == "").any():
        raise DataError(f"{path.name}: empty image id")
    if rec["image_id"].duplicated().any():
        raise DataError(f"{path.name}: duplicated image ids")
    pattern = re.compile(d["patient_id_pattern"])
    bad = rec.loc[~rec["patient_id"].map(lambda v: bool(pattern.match(v))), "patient_id"]
    if len(bad):
        raise DataError(f"{path.name}: {len(bad)} patient ids do not match {d['patient_id_pattern']} "
                        f"(examples: {bad.head(5).tolist()}). Patient IDs cannot be reconciled.")
    unknown = set(rec["malignancy"]) - set(d["malignancy_values"])
    if unknown:
        raise DataError(f"{path.name}: unknown malignancy values {sorted(unknown)}")
    rec["official_split"] = official_split
    return rec


def index_images(raw: Path, image_dirs: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for sub in image_dirs:
        folder = raw / sub
        if not folder.exists():
            raise DataError(f"Image folder missing: {folder}")
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                key = f.stem.lower()
                if key in index:
                    ambiguous.add(key)
                index[key] = f
    if ambiguous:
        raise DataError(f"{len(ambiguous)} image stems occur more than once, e.g. {sorted(ambiguous)[:5]}")
    return index


def make_val_split(train_pool: pd.DataFrame, seed: int, n_folds: int) -> pd.Index:
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, va_pos = next(sgkf.split(np.zeros(len(train_pool)), train_pool["subtype"].to_numpy(),
                                train_pool["patient_id"].to_numpy()))
    return train_pool.index[va_pos]


def build(cfg: dict, raw: Path, out_dir: Path, reports_dir: Path) -> dict:
    d = cfg["data"]
    subtypes = list(d["subtype_labels"])
    other = d["other_benign_label"]
    expected = d["expected"]
    warnings: list[str] = []

    train_rec = load_metadata(raw / d["metadata_train"], cfg, "train")
    test_rec = load_metadata(raw / d["metadata_test"], cfg, "test")
    recs = pd.concat([train_rec, test_rec], ignore_index=True)
    if recs["image_id"].duplicated().any():
        raise DataError("An image id appears in both metadata files (or twice). Stop and check the Kaggle version.")

    lab = recs["label_raw"].str.lower()
    benign_mask = recs["malignancy"] == "benign"
    recs["binary"] = np.where(recs["malignancy"] == "malignant", 1,
                              np.where(benign_mask, 0, -1))
    recs["subtype"] = np.where(recs["malignancy"] == "malignant", "malignant",
                      np.where(recs["malignancy"] == "indeterminate", "indeterminate",
                      np.where(lab.isin(subtypes), lab, other)))

    index = index_images(raw, d["image_dirs"])
    located = recs["image_id"].str.lower().map(index)
    missing = int(located.isna().sum())
    if missing:
        examples = recs.loc[located.isna(), "image_id"].head(5).tolist()
        raise DataError(f"{missing} of {len(recs)} metadata rows have no image file, e.g. {examples}")
    recs["path"] = [p.relative_to(raw).as_posix() for p in located]
    recs["sha256"] = ""
    recs["phash"] = np.uint64(0)
    recs["group_id"] = -1
    recs["status"] = "kept"
    recs["reason"] = ""

    # 3. indeterminate
    ind = recs["malignancy"] == "indeterminate"
    recs.loc[ind, "status"] = "excluded_indeterminate"
    recs.loc[ind, "reason"] = "malignancy=indeterminate"

    # 4. patient overlap: train images whose patient is in the official test split
    test_patients = set(recs.loc[recs["official_split"] == "test", "patient_id"])
    overlap = (recs["official_split"] == "train") & recs["patient_id"].isin(test_patients) & ~ind
    recs.loc[overlap, "status"] = "excluded_patient_overlap"
    recs.loc[overlap, "reason"] = "train image of a patient that appears in official test"
    overlap_patients = sorted(set(recs.loc[overlap, "patient_id"]))

    # 5. duplicates among active records (SHA-256 exact, pHash near)
    active = recs.index[recs["status"] == "kept"]
    for i in active:
        recs.at[i, "sha256"] = sha256_file(raw / recs.at[i, "path"])
        recs.at[i, "phash"] = np.uint64(phash_int(raw / recs.at[i, "path"], cfg["dedup"]["phash_hash_size"]))
    act = recs.loc[active, ["image_id", "official_split", "sha256", "phash"]].copy()
    groups = group_duplicates(act["sha256"].tolist(),
                              act["phash"].to_numpy(dtype=np.uint64),
                              cfg["dedup"]["phash_hamming_threshold"])
    act["group"] = groups
    act = choose_keepers(act)
    recs.loc[active, "group_id"] = act["group"].to_numpy()
    dropped = act.index[~act["is_keeper"]]
    recs.loc[dropped, "status"] = "dropped_duplicate"
    recs.loc[dropped, "reason"] = "duplicate group member (train copy of a test image or extra copy)"
    dup_rows = build_duplicate_report(recs, act)

    # 6. validation split from official train only; official test stays as is
    kept_mask = recs["status"] == "kept"
    train_pool = recs[kept_mask & (recs["official_split"] == "train")]
    val_idx = make_val_split(train_pool, cfg["split"]["seed"], cfg["split"]["n_folds"])
    recs["partition"] = ""
    recs.loc[kept_mask & (recs["official_split"] == "train"), "partition"] = "train"
    recs.loc[kept_mask & (recs["official_split"] == "test"), "partition"] = "test"
    recs.loc[val_idx, "partition"] = "val"
    kept = recs[kept_mask].copy()

    _assert_split(kept, cfg)

    counts = _counts(recs, kept, expected, overlap_patients, warnings, dup_rows)
    write_manifests(kept, recs, out_dir, cfg)
    dup_rows.to_csv(out_dir / "duplicates.csv", index=False)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dup_rows.to_csv(reports_dir / "duplicates.csv", index=False)
    recs.loc[recs["status"] == "excluded_indeterminate"].to_csv(out_dir / "indeterminate_excluded.csv", index=False)
    recs.loc[recs["status"] == "excluded_patient_overlap"].to_csv(out_dir / "patient_overlap_excluded.csv", index=False)
    recs.drop(columns=["phash"]).assign(phash=recs["phash"].astype(str)).to_csv(out_dir / "all_records.csv", index=False)
    return {"counts": counts, "warnings": warnings}


def build_duplicate_report(recs: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    from src.data.dedup import hamming_to

    rows = []
    multi = act.groupby("group").filter(lambda g: len(g) > 1)
    for gid, sub in multi.groupby("group"):
        keeper = sub.loc[sub["is_keeper"]].index[0]
        keeper_sha = recs.at[keeper, "sha256"]
        keeper_hash = np.uint64(recs.at[keeper, "phash"])
        for idx in sub.index:
            h = int(hamming_to(np.array([recs.at[idx, "phash"]], dtype=np.uint64), int(keeper_hash))[0])
            rows.append({
                "group_id": int(gid),
                "image_id": recs.at[idx, "image_id"],
                "official_split": recs.at[idx, "official_split"],
                "patient_id": recs.at[idx, "patient_id"],
                "action": "kept" if idx == keeper else "dropped",
                "match_to_keeper": "keeper" if idx == keeper else ("exact" if recs.at[idx, "sha256"] == keeper_sha else "near"),
                "hamming_to_keeper": h,
            })
    return pd.DataFrame(rows, columns=["group_id", "image_id", "official_split", "patient_id", "action",
                                       "match_to_keeper", "hamming_to_keeper"])


def _assert_split(kept: pd.DataFrame, cfg: dict) -> None:
    subtypes = list(cfg["data"]["subtype_labels"]) + [cfg["data"]["other_benign_label"]]
    val = kept[kept["partition"] == "val"]
    train = kept[kept["partition"] == "train"]
    test = kept[kept["partition"] == "test"]
    if set(val["binary"]) != {0, 1}:
        raise DataError("Validation split lacks a binary class")
    missing = [c for c in subtypes + ["malignant"] if c not in set(val["subtype"])]
    if missing:
        raise DataError(f"Validation split lacks classes {missing}; change split.seed/n_folds and document it")
    tr_p, va_p, te_p = set(train["patient_id"]), set(val["patient_id"]), set(test["patient_id"])
    if (tr_p & va_p) or (tr_p & te_p) or (va_p & te_p):
        raise DataError("Patient leakage between train/val/test after split")
    if set(train["subtype"]) - set(subtypes + ["malignant"]):
        raise DataError("Unexpected subtype label in training split")


def _counts(recs, kept, expected, overlap_patients, warnings, dup_rows) -> dict:
    raw_train = int((recs["official_split"] == "train").sum())
    raw_test = int((recs["official_split"] == "test").sum())
    overlap_rows = recs[recs["status"] == "excluded_patient_overlap"]
    actual = {
        "total": int(len(recs)),
        "train": raw_train,
        "test": raw_test,
        "patients": int(recs["patient_id"].nunique()),
        "subclasses": int(recs["label_raw"].str.lower().nunique()),
        "indeterminate": int((recs["status"] == "excluded_indeterminate").sum()),
        "train_images_of_test_patients": int(len(overlap_rows)),
        "train_patients_of_test": int(len(overlap_patients)),
    }
    comparison = {}
    for key, exp in expected.items():
        act = actual.get(key)
        comparison[key] = {"expected": exp, "actual": act, "match": act == exp}
        if act != exp:
            warnings.append(f"{key}: expected {exp}, actual {act}")
    dup_dropped = int((recs["status"] == "dropped_duplicate").sum())
    return {
        "expected_vs_actual": comparison,
        "removed": {
            "indeterminate": actual["indeterminate"],
            "patient_overlap_images": int(len(overlap_rows)),
            "patient_overlap_patients": int(len(overlap_patients)),
            "duplicate_images_dropped": dup_dropped,
            "duplicate_groups": int(dup_rows["group_id"].nunique()),
        },
        "kept_by_partition": kept["partition"].value_counts().sort_index().to_dict(),
        "kept_stage1_by_partition": {p: {"0_benign": int((g["binary"] == 0).sum()), "1_malignant": int((g["binary"] == 1).sum())}
                                     for p, g in kept.groupby("partition")},
        "kept_subtype_by_partition": {p: g["subtype"].value_counts().sort_index().to_dict()
                                      for p, g in kept.groupby("partition")},
        "validation_patients": int(kept.loc[kept["partition"] == "val", "patient_id"].nunique()),
        "test_patients": int(kept.loc[kept["partition"] == "test", "patient_id"].nunique()),
        "train_patients": int(kept.loc[kept["partition"] == "train", "patient_id"].nunique()),
        "overlap_patient_ids": overlap_patients,
    }


def write_manifests(kept: pd.DataFrame, recs: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        sub = kept[kept["partition"] == part].sort_values("image_id")
        binary = pd.DataFrame({
            "image_id": sub["image_id"], "patient_id": sub["patient_id"], "path": sub["path"],
            "y_malignant": sub["binary"].astype(int), "subtype": sub["subtype"],
        })
        binary.to_csv(out_dir / f"binary_{part}.csv", index=False)
        benign = sub[sub["binary"] == 0]
        pd.DataFrame({
            "image_id": benign["image_id"], "patient_id": benign["patient_id"], "path": benign["path"],
            "subtype": benign["subtype"],
        }).to_csv(out_dir / f"benign_subtypes_{part}.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    raw = resolve_path(cfg["data"]["raw_dir"])
    out_dir = resolve_path(cfg["data"]["manifest_dir"])
    reports_dir = resolve_path(cfg["data"]["reports_dir"])
    if not raw.exists():
        raise DataError(f"Raw data folder missing: {raw}. Set data.raw_dir or env DIAG_RAW_DIR.")
    result = build(cfg, raw, out_dir, reports_dir)
    summary = {
        "dataset": "DERM12345",
        "config_sha256": config_sha256(cfg),
        "git_commit": git_commit(),
        "split_seed": cfg["split"]["seed"],
        "dedup": cfg["dedup"],
        **result["counts"],
        "warnings": result["warnings"],
    }
    summary["manifest_sha256"] = manifest_sha256(out_dir)
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest_sha256": summary["manifest_sha256"],
                      "expected_vs_actual": summary["expected_vs_actual"],
                      "removed": summary["removed"],
                      "warnings": summary["warnings"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
