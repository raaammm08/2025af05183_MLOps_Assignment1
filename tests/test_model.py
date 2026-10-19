"""Unit tests for ML model loading, predictions, and performance validation.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.predict import get_predictor, MODEL_PATH
from src.data.preprocess import TEST_DATA_PATH, TARGET_COLUMN


def test_model_artifact_exists():
    """Verify champion model artifact file exists on disk."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_model_inference_structure():
    """Verify single inference returns expected schema and value domains."""
    predictor = get_predictor()
    sample = {
        "age": 58.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 125.0,
        "chol": 220.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 144.0,
        "exang": 0.0,
        "oldpeak": 0.4,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    }

    result = predictor.predict_single(sample)
    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert "prediction_label" in result
    assert 0.0 <= result["disease_probability"] <= 1.0
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["risk_level"] in ["Low", "Moderate", "High"]


def test_model_performance_on_test_set():
    """Verify that champion model achieves performance greater than benchmark threshold (>80%)."""
    assert os.path.exists(TEST_DATA_PATH), f"Test data not found at {TEST_DATA_PATH}"
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    predictor = get_predictor()
    batch_results = predictor.predict_batch(X_test)
    y_pred = [r["prediction"] for r in batch_results]
    y_prob = [r["disease_probability"] for r in batch_results]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    assert acc >= 0.80, f"Model accuracy ({acc:.4f}) dropped below 0.80 benchmark threshold!"
    assert auc >= 0.85, f"Model ROC-AUC ({auc:.4f}) dropped below 0.85 benchmark threshold!"


def test_risk_sensitivity():
    """Verify model discriminates between healthy and pathological clinical records."""
    predictor = get_predictor()

    healthy_record = {
        "age": 35.0,
        "sex": 0.0,
        "cp": 2.0,
        "trestbps": 115.0,
        "chol": 170.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 180.0,
        "exang": 0.0,
        "oldpeak": 0.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    }

    severe_record = {
        "age": 68.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 165.0,
        "chol": 310.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 105.0,
        "exang": 1.0,
        "oldpeak": 3.6,
        "slope": 2.0,
        "ca": 3.0,
        "thal": 7.0,
    }

    healthy_res = predictor.predict_single(healthy_record)
    severe_res = predictor.predict_single(severe_record)

    assert healthy_res["prediction"] == 0
    assert severe_res["prediction"] == 1
    assert healthy_res["disease_probability"] < severe_res["disease_probability"]
