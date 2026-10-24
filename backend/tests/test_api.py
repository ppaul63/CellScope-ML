"""Model API tests. Prediction tests need an exported model in artifacts/models/cervical-cytology/
(or ARTIFACTS_DIR pointing at a folder that has one)."""
import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

MODEL_ID = "cervical-cytology"
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parents[2] / "artifacts"))
needs_model = pytest.mark.skipif(not (ARTIFACTS / "models" / MODEL_ID / "model.onnx").exists(),
                                 reason="no exported model")
PREDICT = f"/api/models/{MODEL_ID}/predict"


def signed_in(c: TestClient) -> TestClient:
    from backend.tests.conftest import make_admin

    username, password = make_admin()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return c


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        yield signed_in(c)


def image_bytes(fmt="PNG", size=(320, 260)):
    buf = io.BytesIO()
    Image.new("RGB", size, (180, 120, 160)).save(buf, format=fmt)
    return buf.getvalue()


def test_models_listed_from_registry(client):
    models = client.get("/api/models").json()
    cervical = next(m for m in models if m["id"] == MODEL_ID)
    assert cervical["name"] == "Cervical Cytology" and cervical["category"] == "Oncology"
    assert {c["code"] for c in cervical["classes"]} == {"NILM", "ASCUS", "LSIL", "HSIL", "SCC"}
    assert client.get("/api/models/does-not-exist").status_code == 404


@needs_model
def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["models_ready"] >= 1


@needs_model
def test_model_detail_when_ready(client):
    info = client.get(f"/api/models/{MODEL_ID}").json()
    assert info["status"] == "ready" and info["arch"] and info["image_size"] == 224


@needs_model
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "TIFF"])
def test_predict_ok(client, fmt):
    r = client.post(PREDICT, files={"file": (f"x.{fmt.lower()}", image_bytes(fmt))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == MODEL_ID
    assert body["predicted_class"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["risk_level"] in {"Low", "Moderate", "High"}
    assert body["class_name"] != body["predicted_class"]  # details come from models.yaml
    assert body["heatmap_png_base64"]


@needs_model
def test_predict_without_heatmap(client):
    r = client.post(f"{PREDICT}?heatmap=false", files={"file": ("x.png", image_bytes())})
    assert r.status_code == 200 and r.json()["heatmap_png_base64"] is None


@needs_model
def test_rejects_non_image(client):
    r = client.post(PREDICT, files={"file": ("x.txt", b"hello world")})
    assert r.status_code == 400


@needs_model
def test_rejects_oversized(client, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main, "MAX_BYTES", 100)
    r = client.post(PREDICT, files={"file": ("x.png", image_bytes())})
    assert r.status_code == 413


def test_unknown_model_predict_is_404(client):
    assert client.post("/api/models/nope/predict", files={"file": ("x.png", image_bytes())}).status_code == 404


def test_untrained_model_is_listed_but_cannot_predict(monkeypatch, tmp_path):
    from backend.app.main import app

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    with TestClient(app) as c:
        signed_in(c)
        assert c.get("/api/health").json()["models_ready"] == 0
        assert c.get(f"/api/models/{MODEL_ID}").json()["status"] == "not_trained"
        r = c.post(PREDICT, files={"file": ("x.png", image_bytes())})
        assert r.status_code == 503 and "not been trained" in r.json()["detail"]


def test_model_endpoints_require_login():
    from backend.app.main import app

    with TestClient(app) as c:
        assert c.get("/api/models").status_code == 401
        assert c.post(PREDICT, files={"file": ("x.png", image_bytes())}).status_code == 401
