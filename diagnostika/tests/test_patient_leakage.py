import inspect

import pandas as pd
import pytest

from src.common.config import load_config
from src.common.manifest import ManifestError, verify_manifest
from src.data import prepare


def _cfg():
    cfg = load_config()
    cfg["split"]["n_folds"] = 5
    return cfg


def test_full_pipeline_on_synthetic_data_removes_overlap_and_duplicates(synthetic_derm, tmp_path):
    cfg = _cfg()
    out = tmp_path / "manifests"
    reports = tmp_path / "reports"
    result = prepare.build(cfg, synthetic_derm, out, reports)
    removed = result["counts"]["removed"]
    assert removed["patient_overlap_images"] == 3
    assert removed["patient_overlap_patients"] == 1
    assert removed["indeterminate"] == 2
    assert removed["duplicate_images_dropped"] == 1

    binary = {p: pd.read_csv(out / f"binary_{p}.csv", dtype=str) for p in ("train", "val", "test")}
    assert "ISIC_OVERLAP_0" not in set(binary["train"]["image_id"])
    assert "ISIC_DUP_TRAIN" not in set(binary["train"]["image_id"])     # train copy of a test image dropped
    assert "ISIC_TE_1_0" in set(binary["test"]["image_id"])            # test copy kept
    assert not any(i.startswith("ISIC_IND") for p in binary.values() for i in p["image_id"])


def test_no_patient_appears_in_two_partitions(synthetic_derm, tmp_path):
    cfg = _cfg()
    out = tmp_path / "m"
    prepare.build(cfg, synthetic_derm, out, tmp_path / "r")
    sets = {p: set(pd.read_csv(out / f"binary_{p}.csv", dtype=str)["patient_id"]) for p in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_validation_contains_every_class(synthetic_derm, tmp_path):
    cfg = _cfg()
    out = tmp_path / "m"
    prepare.build(cfg, synthetic_derm, out, tmp_path / "r")
    val = pd.read_csv(out / "benign_subtypes_val.csv", dtype=str)
    binary_val = pd.read_csv(out / "binary_val.csv", dtype=str)
    assert set(val["subtype"]) == {"bdb", "cb", "db", "srjd", "other_benign"}
    assert set(binary_val["y_malignant"]) == {"0", "1"}


def test_manifest_hash_detects_tampering(synthetic_derm, tmp_path):
    cfg = _cfg()
    out = tmp_path / "m"
    result = prepare.build(cfg, synthetic_derm, out, tmp_path / "r")
    from src.common.manifest import manifest_sha256, SUMMARY_FILE
    import json
    (out / SUMMARY_FILE).write_text(json.dumps({"manifest_sha256": manifest_sha256(out)}), encoding="utf-8")
    verify_manifest(out)  # clean
    path = out / "binary_test.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        verify_manifest(out)


def test_training_scripts_never_open_test_manifests():
    from src.stage1_binary import train as s1
    from src.stage2_subtype import train as s2
    assert "binary_test" not in inspect.getsource(s1)
    assert "benign_subtypes_test" not in inspect.getsource(s2)
    from src.stage1_binary import threshold as thr
    assert "binary_test" not in inspect.getsource(thr)
