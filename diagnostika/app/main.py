"""FastAPI service for the two-stage cascade. Run:  python -m app.main  (serves on config inference.host/port)

Images are processed in memory and never written to disk or logged.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from src.common.config import load_config
from src.inference.predictor import ROIError, Predictor
from src.training.weights_io import WeightsMissingError

STATIC = Path(__file__).resolve().parent / "static"
DISCLAIMER = ("Հետազոտական նախատիպ է, ոչ թե բժշկական ախտորոշիչ սարք։ Արդյունքը աջակցում է, "
              "բայց չի փոխարինում բժշկի դատողությանը։")
MAX_PIXELS = 60_000_000
ROI_FORM = ("roi_x", "roi_y", "roi_w", "roi_h")


def create_app(predictor, cfg: dict) -> FastAPI:
    app = FastAPI(title="diagnostika", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
    max_bytes = int(cfg["inference"]["max_upload_mb"] * 1024 * 1024)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models": predictor.bundle.models, "classes": predictor.bundle.classes,
                "ensemble_mode": predictor.bundle.mode, "disclaimer": DISCLAIMER}

    @app.post("/api/analyze")
    def analyze(file: UploadFile = File(...), roi_x: float | None = Form(None), roi_y: float | None = Form(None),
                roi_w: float | None = Form(None), roi_h: float | None = Form(None)):
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Ֆայլը մեծ է ({cfg['inference']['max_upload_mb']} ՄԲ-ից)")
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="Ֆայլը պատկեր չէ կամ չի կարող կարդացվել")
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Պատկերի չափը չափազանց մեծ է")
        img = img.convert("RGB")
        given = [v for v in (roi_x, roi_y, roi_w, roi_h) if v is not None]
        roi = None
        if given:
            if len(given) != 4:
                raise HTTPException(status_code=400, detail="ROI-ի բոլոր չորս արժեքները պետք է տրվեն")
            roi = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
        try:
            result = predictor.predict(img, roi)
        except ROIError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        result["disclaimer"] = DISCLAIMER
        return JSONResponse(result)

    return app


def main(argv=None) -> None:
    import uvicorn

    cfg = load_config()
    try:
        predictor = Predictor(cfg)
    except WeightsMissingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    app = create_app(predictor, cfg)
    uvicorn.run(app, host=cfg["inference"]["host"], port=cfg["inference"]["port"])


if __name__ == "__main__":
    main()
