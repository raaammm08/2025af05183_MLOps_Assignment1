"""Integration and Unit tests for FastAPI Serving endpoints.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Verify landing endpoint returns service metadata and student identity."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["student"]["roll_number"] == "2025af05183"
    assert data["student"]["name"] == "Ram Narayan"


def test_health_check_endpoint():
    """Verify Kubernetes health probe returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["uptime_seconds"] >= 0.0


def test_predict_endpoint_valid_case():
    """Verify /predict returns valid prediction, confidence, and risk level."""
    payload = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert "prediction_label" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["disease_probability"] <= 1.0
    assert data["student_id"] == "2025af05183"
    assert "timestamp" in data


def test_predict_endpoint_invalid_payload():
    """Verify /predict rejects out-of-range or missing values with HTTP 422."""
    invalid_payload = {
        "age": -15.0,  # Invalid age
        "sex": 5.0,   # Invalid sex
        "cp": 1.0,
        "trestbps": 500.0,  # Out of range
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_batch_predict_endpoint():
    """Verify /predict/batch handles multiple records simultaneously."""
    payload = {
        "patients": [
            {
                "age": 45.0,
                "sex": 0.0,
                "cp": 2.0,
                "trestbps": 120.0,
                "chol": 210.0,
                "fbs": 0.0,
                "restecg": 0.0,
                "thalach": 170.0,
                "exang": 0.0,
                "oldpeak": 0.0,
                "slope": 1.0,
                "ca": 0.0,
                "thal": 3.0,
            },
            {
                "age": 67.0,
                "sex": 1.0,
                "cp": 4.0,
                "trestbps": 160.0,
                "chol": 286.0,
                "fbs": 0.0,
                "restecg": 2.0,
                "thalach": 108.0,
                "exang": 1.0,
                "oldpeak": 1.5,
                "slope": 2.0,
                "ca": 3.0,
                "thal": 3.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2


def test_prometheus_metrics_endpoint():
    """Verify /metrics returns Prometheus text scrape format with custom metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "api_requests_total" in content
    assert "api_request_duration_seconds" in content
