# Video Walkthrough Script: End-to-End MLOps Pipeline Demonstration

**Student Name:** Ram Narayan  
**Student ID / Roll:** 2025af05183  
**Email:** 2025af05183@wilp.bits-pilani.ac.in  
**Course:** AIMLCZG523 Machine Learning Operations (MLOps) - Assignment 01  
**Project:** Cloud-Ready Monitored Heart Disease Risk Classifier  

---

## Video Overview & Demonstration Outline (Target Duration: ~3 to 5 Minutes)

| Timestamp | Phase / Section | Screen Display & Action | Narration Keypoints |
|---|---|---|---|
| **0:00 - 0:35** | **Introduction & Architecture Overview** | Display `reports/screenshots/08_architecture_pipeline_diagram.png` | *"Hello, I am Ram Narayan (Student ID: 2025af05183) presenting Assignment 1 for MLOps AIMLCZG523. Today I will demonstrate our production-grade, end-to-end Machine Learning Operations pipeline for Heart Disease risk prediction using the UCI Cleveland dataset."* |
| **0:35 - 1:15** | **Data Acquisition & EDA** | Open `notebooks/01_eda_and_feature_engineering.ipynb` and show `reports/screenshots/01_eda_distributions.png`, `02_eda_correlation_heatmap.png`, and `03_eda_class_balance.png` | *"We implemented automated data acquisition from the UCI Repository. Our EDA evaluated 14 clinical features across 303 patient records, analyzing continuous biomarkers and categorical correlations with heart disease presence."* |
| **1:15 - 2:00** | **Feature Engineering & MLflow Tracking** | Terminal showing `python src/models/train.py`, followed by MLflow UI at `http://localhost:5000` or `reports/screenshots/07_mlflow_runs_dashboard.png` | *"We implemented a ColumnTransformer pipeline handling imputation, scaling, and one-hot encoding without data leakage. We tuned Logistic Regression, Random Forest, and Gradient Boosting under 5-Fold Stratified CV. Logistic Regression emerged as champion with 86.89% test accuracy and a 0.9632 ROC-AUC."* |
| **2:00 - 2:40** | **Testing & CI/CD Pipeline** | Terminal running `pytest -v --cov=src tests/`, followed by GitHub Actions workflow YAML display | *"Our test suite achieves 93% code coverage across data processing, model inference sanity (>80% accuracy threshold), and API endpoints. The GitHub Actions CI/CD automatically lints, tests, trains, packages, and smoke-tests the container on every push."* |
| **2:40 - 3:30** | **Docker Serving & FastAPI Swagger UI** | Terminal showing container start, then opening `http://localhost:8000/docs` | *"We containerized the service using a multi-stage, non-root Dockerfile. Testing the `/predict` endpoint live with high-risk and low-risk clinical inputs demonstrates real-time predictions, probabilities, and stratified risk tiers."* |
| **3:30 - 4:20** | **Kubernetes Orchestration & Monitoring** | Terminal showing `kubectl get pods,svc,hpa -n mlops-heart-disease` and Prometheus/Grafana dashboard | *"We authored declarative Kubernetes manifests and a production Helm chart featuring rolling updates, resource limits, liveness probes, and HPA. Prometheus scrapes `/metrics` and Grafana displays real-time request rates, latency percentiles, and prediction drift."* |
| **4:20 - 4:45** | **Conclusion & Wrap-Up** | Display final deliverables list and contact information | *"All artifacts, reproducible code, serialized models, Kubernetes manifests, and our comprehensive 10-page report are packaged under Roll Number 2025af05183. Thank you!"* |

---

## Live Demonstration Terminal Commands

Execute the following commands in sequence during your video capture:

```bash
# 1. Environment & Test Suite Verification
pytest -v --cov=src tests/

# 2. Automated Training & Experiment Tracking
python src/data/download_data.py
python src/data/eda.py
python src/models/train.py

# 3. Launch Local Model Serving API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 4. In a separate tab: Test API Endpoints
# Check Health Status
curl -s http://localhost:8000/health | jq .

# Send Sample Prediction Request (Low Risk Patient)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45.0, "sex": 0.0, "cp": 2.0, "trestbps": 120.0, "chol": 210.0,
    "fbs": 0.0, "restecg": 0.0, "thalach": 170.0, "exang": 0.0,
    "oldpeak": 0.0, "slope": 1.0, "ca": 0.0, "thal": 3.0
  }' | jq .

# Check Prometheus Scrape Endpoint
curl -s http://localhost:8000/metrics | grep api_requests_total

# 5. Build and Run Container
docker build -t heart-disease-api:2025af05183 .
docker run -d -p 8000:8000 heart-disease-api:2025af05183

# 6. Deploy to Kubernetes
kubectl apply -f k8s/
kubectl get pods,svc,hpa -n mlops-heart-disease
```
