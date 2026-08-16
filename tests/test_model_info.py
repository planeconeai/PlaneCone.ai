import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Mammo-CLIP"
    assert data["clinical_diagnosis"] is False
    assert data["calibrated_probability"] is False
    assert "license" in data
    assert "CC BY-NC-SA 4.0" in data["license"]
