"""FastAPI endpoint unit test using TestClient."""

from fastapi.testclient import TestClient

from src.inference.api import app

client = TestClient(app)


def test_single_predict_endpoint():
    resp = client.post("/predict", json={"text": "1 2 3"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data and "confidence" in data


def test_batch_predict_endpoint():
    resp = client.post("/batch_predict", json={"texts": ["1 2", "3 4"]})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) == 2
