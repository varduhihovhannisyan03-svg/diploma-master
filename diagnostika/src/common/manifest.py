"""Manifest hashing. Every training / calibration / evaluation step verifies the manifest hash first."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_FILES = (
    "binary_train.csv", "binary_val.csv", "binary_test.csv",
    "benign_subtypes_train.csv", "benign_subtypes_val.csv", "benign_subtypes_test.csv",
)
SUMMARY_FILE = "manifest_summary.json"


class ManifestError(RuntimeError):
    pass


def manifest_sha256(manifest_dir: str | Path) -> str:
    """SHA-256 over the six split manifests, in a fixed order, with name and length framing."""
    manifest_dir = Path(manifest_dir)
    h = hashlib.sha256()
    for name in MANIFEST_FILES:
        path = manifest_dir / name
        if not path.exists():
            raise ManifestError(f"Manifest file missing: {path}. Run: python -m src.data.prepare")
        data = path.read_bytes()
        h.update(name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
    return h.hexdigest()


def verify_manifest(manifest_dir: str | Path) -> dict:
    """Recompute the hash and compare it with manifest_summary.json. Raises on any mismatch."""
    manifest_dir = Path(manifest_dir)
    summary_path = manifest_dir / SUMMARY_FILE
    if not summary_path.exists():
        raise ManifestError(f"{summary_path} missing. Run: python -m src.data.prepare")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual = manifest_sha256(manifest_dir)
    if actual != summary.get("manifest_sha256"):
        raise ManifestError(
            "Manifest hash mismatch: files changed since prepare ran "
            f"(recorded {summary.get('manifest_sha256')}, actual {actual}). Re-run src.data.prepare.")
    return summary
