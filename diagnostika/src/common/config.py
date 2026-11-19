"""Load config.yaml, resolve paths against diagnostika/, and hash configs for run records."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]  # the diagnostika/ folder
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml. Environment variable DIAG_RAW_DIR overrides data.raw_dir (keeps paths out of git)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    raw_override = os.environ.get("DIAG_RAW_DIR")
    if raw_override:
        cfg = copy.deepcopy(cfg)
        cfg["data"]["raw_dir"] = raw_override
    return cfg


def resolve_path(value: str | Path) -> Path:
    """Absolute paths are kept; relative paths are resolved against diagnostika/."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def config_sha256(cfg: dict[str, Any]) -> str:
    """Canonical hash of the settings (sorted keys, no whitespace). Environment overrides are excluded."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_subtypes(cfg: dict[str, Any], include_other_benign: bool) -> list[str]:
    classes = list(cfg["data"]["subtype_labels"])
    if include_other_benign:
        classes.append(cfg["data"]["other_benign_label"])
    return classes
