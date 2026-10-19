"""Prometheus Monitoring and Observability metrics collector for FastAPI.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics Definitions
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total count of HTTP requests processed by endpoint and status code",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency histogram in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTIONS_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions segmented by label and risk level",
    ["prediction_label", "risk_level"],
)

PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)


def get_latest_metrics() -> tuple[bytes, str]:
    """Generates serialized metrics for Prometheus scraper endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
