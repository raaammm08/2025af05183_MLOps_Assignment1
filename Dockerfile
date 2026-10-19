# Multi-stage production Dockerfile for Heart Disease MLOps Serving API
# Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
# BITS Pilani - AIMLCZG523 MLOps Assignment 01

# ==========================================
# Stage 1: Build Environment & Dependencies
# ==========================================
FROM python:3.9-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Production Distroless / Slim Image
# ==========================================
FROM python:3.9-slim AS runner

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy installed python wheels from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Copy application code and artifacts
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/

# Grant permissions to unprivileged user
RUN chown -R appuser:appgroup /app /home/appuser

USER appuser

EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
