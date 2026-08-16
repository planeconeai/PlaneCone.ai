import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app import app, engine

client = TestClient(app)

def create_dummy_png_bytes():
    img = Image.new("L", (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_predict_no_files():
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NO_FILES"

def test_predict_model_unavailable_when_unloaded():
    # Store original state
    was_loaded = engine.loaded
    engine.loaded = False
    
    png_bytes = create_dummy_png_bytes()
    files = [("files", ("r_cc.png", png_bytes, "image/png"))]
    response = client.post("/predict", files=files)
    assert response.status_code == 503
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "MODEL_UNAVAILABLE"

    # Restore state
    engine.loaded = was_loaded

def test_predict_mocked_model_inference():
    # Mock loaded engine for test
    was_loaded = engine.loaded
    was_model = engine.model
    was_processor = engine.processor
    engine.loaded = True

    # Simple mock predict method
    def mock_predict(tensors, metadata):
        per_view = [
            {
                "view_position": m.get("view_position", "UNKNOWN"),
                "sop_instance_uid": m.get("sop_instance_uid"),
                "filename": m.get("filename"),
                "predictions": [
                    {"concept": "mass", "score": 0.45, "score_type": "zero_shot_text_alignment"},
                    {"concept": "normal", "score": 0.55, "score_type": "zero_shot_text_alignment"}
                ]
            }
            for m in metadata
        ]
        aggregate = [
            {"concept": "mass", "score": 0.45, "score_type": "zero_shot_text_alignment"},
            {"concept": "normal", "score": 0.55, "score_type": "zero_shot_text_alignment"}
        ]
        return per_view, aggregate, 0.12

    orig_predict = engine.predict
    engine.predict = mock_predict

    png_bytes = create_dummy_png_bytes()
    files = [
        ("files", ("r_cc.png", png_bytes, "image/png")),
        ("files", ("r_mlo.png", png_bytes, "image/png"))
    ]
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model"]["name"] == "Mammo-CLIP"
    assert data["model"]["clinical_diagnosis"] is False
    assert len(data["views"]) == 2
    assert "warnings" in data
    assert len(data["warnings"]) > 0

    # Restore engine
    engine.predict = orig_predict
    engine.loaded = was_loaded
    engine.model = was_model
    engine.processor = was_processor
