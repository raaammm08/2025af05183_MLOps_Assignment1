"""Pydantic schemas for Heart Disease Risk Prediction API.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict


class PatientFeatures(BaseModel):
    """Input schema representing patient clinical features matching UCI Cleveland specifications."""

    age: float = Field(..., ge=1, le=120, description="Age in years", json_schema_extra={"example": 63.0})
    sex: float = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)", json_schema_extra={"example": 1.0})
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)", json_schema_extra={"example": 1.0})
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure in mm Hg", json_schema_extra={"example": 145.0})
    chol: float = Field(..., ge=80, le=650, description="Serum cholesterol in mg/dl", json_schema_extra={"example": 233.0})
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", json_schema_extra={"example": 1.0})
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0=normal, 1=ST-T abnormality, 2=LVH)", json_schema_extra={"example": 2.0})
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved (bpm)", json_schema_extra={"example": 150.0})
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)", json_schema_extra={"example": 0.0})
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 2.3})
    slope: float = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)", json_schema_extra={"example": 3.0})
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by flourosopy", json_schema_extra={"example": 0.0})
    thal: float = Field(..., description="Thallium heart scan (3=normal, 6=fixed defect, 7=reversible defect)", json_schema_extra={"example": 6.0})

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "thal": 6.0
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output schema for model prediction and risk assessment."""

    prediction: int = Field(..., description="Binary classification (0 = Absence, 1 = Presence)")
    prediction_label: str = Field(..., description="Human-readable prediction diagnosis")
    disease_probability: float = Field(..., description="Predicted probability of heart disease [0.0 - 1.0]")
    confidence: float = Field(..., description="Confidence score associated with predicted class [0.0 - 1.0]")
    risk_level: str = Field(..., description="Stratified risk category: Low (<35%), Moderate (35-65%), High (>65%)")
    model_version: str = Field(default="1.0.0", description="Deployed model version")
    student_id: str = Field(default="2025af05183", description="Student BITS ID")
    timestamp: str = Field(..., description="ISO 8601 inference timestamp")


class BatchPatientRequest(BaseModel):
    patients: List[PatientFeatures]


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
    course: str
    author: str
