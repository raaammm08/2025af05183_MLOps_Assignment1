"""Inference pipeline for Heart Disease Risk Prediction.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from src.utils.logger import get_logger

logger = get_logger("model_inference")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")


class HeartDiseasePredictor:
    """Production inference wrapper for serialized scikit-learn Pipeline."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model artifact not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded champion model from {self.model_path}")

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts heart disease presence and risk level for a single patient record."""
        df = pd.DataFrame([input_data])
        return self.predict_batch(df)[0]

    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predicts heart disease presence and risk level for a batch dataframe."""
        if self.model is None:
            self.load_model()

        probabilities = self.model.predict_proba(df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []
        for pred, prob in zip(predictions, probabilities):
            confidence = float(prob) if pred == 1 else float(1.0 - prob)

            if prob < 0.35:
                risk_level = "Low"
            elif prob < 0.65:
                risk_level = "Moderate"
            else:
                risk_level = "High"

            results.append({
                "prediction": int(pred),
                "prediction_label": "Presence of Heart Disease" if pred == 1 else "Absence of Heart Disease",
                "disease_probability": round(float(prob), 4),
                "confidence": round(confidence, 4),
                "risk_level": risk_level
            })

        return results


# Global singleton predictor instance
_predictor = None


def get_predictor() -> HeartDiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HeartDiseasePredictor()
    return _predictor
