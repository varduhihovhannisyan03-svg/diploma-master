"""Single-lesion inference: optional ROI crop -> 224x224 -> stage 1 ensemble -> stage 2 -> cascade decision."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.common.cli import select_device
from src.inference.cascade import decide
from src.inference.ensemble import ensemble_benign, renormalize_benign
from src.training.dataset import build_transforms
from src.training.weights_io import LoadedModels, load_models

ROI_KEYS = ("x", "y", "w", "h")


class ROIError(ValueError):
    pass


def validate_roi(roi: dict | None) -> dict | None:
    """Normalized rectangle: 0 <= x, y; w, h > 0; x + w <= 1; y + h <= 1."""
    if roi is None:
        return None
    if set(roi) != set(ROI_KEYS):
        raise ROIError(f"ROI needs keys {ROI_KEYS}")
    x, y, w, h = (float(roi[k]) for k in ROI_KEYS)
    if not all(np.isfinite([x, y, w, h])):
        raise ROIError("ROI values must be finite")
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
        raise ROIError("ROI must lie inside [0,1] x [0,1] with positive width and height")
    return {"x": x, "y": y, "w": min(w, 1 - x), "h": min(h, 1 - y)}


def crop_image(img: Image.Image, roi: dict | None) -> tuple[Image.Image, dict]:
    W, H = img.size
    if roi is None:
        return img, {"mode": "full", "image_width": W, "image_height": H, "roi": None,
                     "area_fraction": 1.0, "crop_pixels": [W, H]}
    x0 = int(round(roi["x"] * W))
    y0 = int(round(roi["y"] * H))
    x1 = max(x0 + 2, int(round((roi["x"] + roi["w"]) * W)))
    y1 = max(y0 + 2, int(round((roi["y"] + roi["h"]) * H)))
    x1, y1 = min(x1, W), min(y1, H)
    x0, y0 = min(x0, x1 - 1), min(y0, y1 - 1)
    crop = img.crop((x0, y0, x1, y1))
    return crop, {"mode": "roi", "image_width": W, "image_height": H, "roi": roi,
                  "area_fraction": float((x1 - x0) * (y1 - y0)) / float(W * H),
                  "crop_pixels": [x1 - x0, y1 - y0]}


class Predictor:
    def __init__(self, cfg: dict, device: torch.device | None = None, bundle: LoadedModels | None = None):
        self.cfg = cfg
        self.device = device or select_device(cfg["training"]["device"])
        self.bundle = bundle or load_models(cfg, self.device)
        t = cfg["training"]
        self.transform = build_transforms(t["image_size"], t["augment"], train=False)
        self.inf = cfg["inference"]

    def predict(self, img: Image.Image, roi: dict | None = None) -> dict:
        roi = validate_roi(roi)
        crop, crop_meta = crop_image(img.convert("RGB"), roi)
        x = self.transform(crop).unsqueeze(0).to(self.device)
        b = self.bundle
        with torch.inference_mode():
            p_mal = [float(torch.softmax(b.stage1[n](x), dim=1)[0, 1]) for n in b.models]
            benign = [torch.softmax(b.stage2[n](x), dim=1)[0].cpu().numpy() for n in b.models]
        per_model_benign = renormalize_benign(np.stack(benign))
        ensemble = ensemble_benign(per_model_benign, b.mode, b.weight_matrix)
        p_ens = float(np.mean(p_mal))
        dec = decide(p_ens, ensemble, per_model_benign, b.classes, b.threshold, self.inf["unclear_below"])
        warnings = []
        if crop_meta["mode"] == "roi" and crop_meta["area_fraction"] < self.inf["roi_min_area_fraction"]:
            pct = round(crop_meta["area_fraction"] * 100)
            warnings.append({
                "code": "roi_small",
                "message": (f"Ընտրված հատվածը ամբողջ պատկերի մոտ {pct}%-ն է։ Մոդելները վերապատրաստվել են "
                            "ամբողջ դերմատոսկոպիկ նկարների վրա, ուստի հատվածի վերլուծությունը դուրս է "
                            "վերապատրաստման տիրույթից։ Արդյունքը մեկնաբանեք զգուշությամբ։"),
            })
        per_model = []
        for i, name in enumerate(b.models):
            per_model.append({
                "model": name,
                "p_malignant": p_mal[i],
                "benign_probabilities": {c: float(v) for c, v in zip(b.classes, per_model_benign[i])},
                "top_benign": b.classes[int(np.argmax(per_model_benign[i]))],
            })
        return {
            "crop": crop_meta,
            "probabilities": dec["probabilities"],
            "top_class": dec["top_class"],
            "top_benign": dec["top_benign"],
            "per_model": per_model,
            "ensemble": {"mode": b.mode, "benign_probabilities": {c: float(v) for c, v in zip(b.classes, ensemble)},
                         "p_malignant": p_ens},
            "agreement": dec["agreement"],
            "n_models": dec["n_models"],
            "referral": dec["referral"],
            "subtype_for_information_only": dec["subtype_for_information_only"],
            "threshold": dec["threshold"],
            "unclear": dec["unclear"],
            "warnings": warnings,
        }
