import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from src.common.config import load_config
from src.inference.predictor import ROIError, validate_roi
from src.training.weights_io import WeightsMissingError, check_weights

CLASSES = ["bdb", "cb", "db", "srjd", "other_benign"]


class FakePredictor:
    """Stands in for the trained cascade; still validates the ROI like the real predictor."""

    class bundle:
        models = ["resnet50"]
        classes = CLASSES
        mode = "average"

    def predict(self, img, roi):
        roi = validate_roi(roi)
        return {"probabilities": {"malignant": 0.1, **{c: 0.18 for c in CLASSES}}, "top_class": "bdb",
                "referral": False, "unclear": False, "agreement": 1, "n_models": 1,
                "threshold": 0.5, "warnings": [], "crop": {"mode": "roi" if roi else "full", "roi": roi,
                                                           "area_fraction": 0.5 if roi else 1.0},
                "per_model": [], "subtype_for_information_only": False, "top_benign": "bdb",
                "ensemble": {}}


def _png_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 60)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    cfg = load_config()
    return TestClient(create_app(FakePredictor(), cfg))


def test_index_page_is_armenian_and_makes_no_external_requests(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert 'lang="hy"' in html
    assert "Վերլուծել ընտրված հատվածը" in html and "Վերլուծել ամբողջ պատկերը" in html
    assert "Հետազոտական նախատիպ" in html
    assert "http://" not in html and "https://" not in html and "//cdn" not in html


def test_analyze_full_image(client):
    res = client.post("/api/analyze", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["crop"]["mode"] == "full"
    assert "disclaimer" in body


def test_analyze_roi_passes_normalized_coordinates(client):
    res = client.post("/api/analyze", files={"file": ("x.png", _png_bytes(), "image/png")},
                      data={"roi_x": "0.1", "roi_y": "0.2", "roi_w": "0.5", "roi_h": "0.4"})
    assert res.status_code == 200
    assert res.json()["crop"]["roi"] == {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4}


def test_roi_outside_image_is_rejected(client):
    res = client.post("/api/analyze", files={"file": ("x.png", _png_bytes(), "image/png")},
                      data={"roi_x": "0.8", "roi_y": "0.2", "roi_w": "0.5", "roi_h": "0.4"})
    assert res.status_code == 400


def test_partial_roi_is_rejected(client):
    res = client.post("/api/analyze", files={"file": ("x.png", _png_bytes(), "image/png")},
                      data={"roi_x": "0.1"})
    assert res.status_code == 400


def test_non_image_is_rejected(client):
    res = client.post("/api/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_validate_roi_bounds():
    assert validate_roi(None) is None
    with pytest.raises(ROIError):
        validate_roi({"x": 0.9, "y": 0, "w": 0.2, "h": 0.2})
    with pytest.raises(ROIError):
        validate_roi({"x": 0, "y": 0, "w": 0, "h": 0.2})


def test_missing_weights_give_a_clear_error(tmp_path):
    cfg = load_config()
    cfg["stage1"]["weights_dir"] = str(tmp_path / "s1")
    cfg["stage2"]["weights_dir"] = str(tmp_path / "s2")
    cfg["inference"]["stage1_threshold_file"] = str(tmp_path / "threshold.json")
    cfg["inference"]["stage2_weights_file"] = str(tmp_path / "weights.json")
    with pytest.raises(WeightsMissingError) as err:
        check_weights(cfg)
    message = str(err.value)
    assert "weights/ is not in git" in message
    assert "threshold.json" in message and "resnet50.pt" in message
