"""FastAPI Production Model Serving Application.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import time
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PatientFeatures,
    PredictionResponse,
    BatchPatientRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from src.api.monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTIONS_COUNT,
    PREDICTION_CONFIDENCE,
    get_latest_metrics,
)
from src.models.predict import get_predictor
from src.utils.logger import get_logger

logger = get_logger("api_server")
START_TIME = time.time()

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API serving scikit-learn classifier for Heart Disease risk prediction.",
    version="1.0.0",
    contact={
        "name": "Ram Narayan",
        "email": "2025af05183@wilp.bits-pilani.ac.in",
        "institution": "BITS Pilani",
    },
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_and_log_requests(request: Request, call_next):
    """Middleware capturing request latency, metrics, and structured logs."""
    start_ts = time.time()
    method = request.method
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        status_code = 500
        logger.error(f"Unhandled exception processing {method} {endpoint}: {exc}")
        raise exc from None
    finally:
        duration = time.time() - start_ts
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        logger.info(f"{method} {endpoint} -> Status: {status_code} | Duration: {duration * 1000:.2f}ms")

    return response


@app.get("/", tags=["General"])
async def root():
    """Root landing endpoint with service metadata."""
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "student": {
            "name": "Ram Narayan",
            "roll_number": "2025af05183",
            "email": "2025af05183@wilp.bits-pilani.ac.in",
            "course": "AIMLCZG523 Machine Learning Operations",
            "assignment": "Assignment 01"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness & Readiness probe for Kubernetes orchestrator."""
    predictor = get_predictor()
    model_loaded = predictor.model is not None
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        course="AIMLCZG523 MLOps Assignment 01",
        author="Ram Narayan (2025af05183)"
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    content, content_type = get_latest_metrics()
    return Response(content=content, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_single_patient(patient: PatientFeatures):
    """Predicts heart disease risk for a single patient record."""
    predictor = get_predictor()
    try:
        data_dict = patient.model_dump()
        result = predictor.predict_single(data_dict)

        # Update custom business metrics
        PREDICTIONS_COUNT.labels(
            prediction_label=result["prediction_label"],
            risk_level=result["risk_level"]
        ).inc()
        PREDICTION_CONFIDENCE.observe(result["confidence"])

        response = PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            disease_probability=result["disease_probability"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            model_version="1.0.0",
            student_id="2025af05183",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return response
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict_batch_patients(batch: BatchPatientRequest):
    """Batch inference endpoint for multiple patient records."""
    predictor = get_predictor()
    try:
        import pandas as pd
        records = [p.model_dump() for p in batch.patients]
        df = pd.DataFrame(records)
        raw_results = predictor.predict_batch(df)

        now_str = datetime.now(timezone.utc).isoformat()
        formatted_preds = []
        for res in raw_results:
            PREDICTIONS_COUNT.labels(
                prediction_label=res["prediction_label"],
                risk_level=res["risk_level"]
            ).inc()
            PREDICTION_CONFIDENCE.observe(res["confidence"])

            formatted_preds.append(
                PredictionResponse(
                    prediction=res["prediction"],
                    prediction_label=res["prediction_label"],
                    disease_probability=res["disease_probability"],
                    confidence=res["confidence"],
                    risk_level=res["risk_level"],
                    model_version="1.0.0",
                    student_id="2025af05183",
                    timestamp=now_str
                )
            )

        return BatchPredictionResponse(
            total_records=len(formatted_preds),
            predictions=formatted_preds
        )
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
