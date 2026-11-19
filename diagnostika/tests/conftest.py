from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _write_png(path: Path, seed: int, size: int = 24) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def synthetic_derm(tmp_path):
    """A tiny DERM12345-shaped dataset: metadata CSVs, image folders, planted problems.

    Planted: 3 train images of a test patient (overlap), 1 train image that is an exact copy of a
    test image (duplicate across splits), 2 indeterminate rows, and enough patients per class for a
    validation split that contains every class.
    """
    raw = tmp_path / "raw"
    classes = ["bdb", "cb", "db", "srjd", "other_benign", "malignant"]
    rows_train, rows_test = [], []
    seed = 0
    n_patients_per_class = 14
    for c in classes:
        for p in range(n_patients_per_class):
            pid = f"TR_{c}_{p}"
            for k in range(3):
                seed += 1
                image_id = f"ISIC_TR_{c}_{p}_{k}"
                _write_png(raw / "derm12345_train_part_1" / f"{image_id}.jpg", seed)
                rows_train.append({"isic_id": image_id, "patient_id": pid,
                                   "label": c if c != "malignant" else "melanoma",
                                   "malignancy": "malignant" if c == "malignant" else "benign",
                                   "split": "train"})
    # test patients
    for p in range(6):
        pid = f"TE_{p}"
        for k in range(3):
            seed += 1
            image_id = f"ISIC_TE_{p}_{k}"
            label = classes[(p + k) % 6]
            _write_png(raw / "derm12345_test" / f"{image_id}.jpg", seed)
            rows_test.append({"isic_id": image_id, "patient_id": pid,
                              "label": label if label != "malignant" else "melanoma",
                              "malignancy": "malignant" if label == "malignant" else "benign",
                              "split": "test"})
    # overlap: train images of test patient TE_0
    for k in range(3):
        seed += 1
        image_id = f"ISIC_OVERLAP_{k}"
        _write_png(raw / "derm12345_train_part_2" / f"{image_id}.jpg", seed)
        rows_train.append({"isic_id": image_id, "patient_id": "TE_0", "label": "bdb",
                           "malignancy": "benign", "split": "train"})
    # exact duplicate: a train copy of test image ISIC_TE_1_0 (same bytes), different patient
    src = raw / "derm12345_test" / "ISIC_TE_1_0.jpg"
    dup_dst = raw / "derm12345_train_part_2" / "ISIC_DUP_TRAIN.jpg"
    dup_dst.write_bytes(src.read_bytes())
    rows_train.append({"isic_id": "ISIC_DUP_TRAIN", "patient_id": "TR_dup", "label": "cb",
                       "malignancy": "benign", "split": "train"})
    # indeterminate rows
    for k in range(2):
        seed += 1
        image_id = f"ISIC_IND_{k}"
        _write_png(raw / "derm12345_train_part_2" / f"{image_id}.jpg", seed)
        rows_train.append({"isic_id": image_id, "patient_id": f"TR_ind_{k}", "label": "melanoma",
                           "malignancy": "indeterminate", "split": "train"})
    pd.DataFrame(rows_train).to_csv(raw / "derm12345_metadata_train.csv", index=False)
    pd.DataFrame(rows_test).to_csv(raw / "derm12345_metadata_test.csv", index=False)
    return raw
