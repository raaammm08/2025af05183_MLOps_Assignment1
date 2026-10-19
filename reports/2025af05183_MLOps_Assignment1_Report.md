# AIMLCZG523 Machine Learning Operations (MLOps) - Assignment 01
## End-to-End ML Model Development, CI/CD, Containerization & Production Deployment

---

**Student Name:** Ram Narayan  
**Student ID / Roll:** 2025af05183  
**BITS Email:** 2025af05183@wilp.bits-pilani.ac.in  
**Course Code:** AIMLCZG523  
**Institution:** Birla Institute of Technology & Science (BITS), Pilani - WILP  
**Submission Date:** October 20, 2026  
**Total Marks:** 50 Marks  
**Dataset:** Heart Disease UCI (Cleveland Clinic Foundation)  

---

## 1. Executive Summary & Problem Statement

Cardiovascular diseases (CVDs) represent the leading cause of premature mortality worldwide, accounting for approximately 17.9 million deaths annually according to the World Health Organization (WHO). Early, reliable, and interpretable clinical diagnostic systems are critical to enable timely preventative therapy and clinical triage.

The objective of this assignment is to design, develop, test, containerize, deploy, and monitor an enterprise-grade, end-to-end Machine Learning Operations (MLOps) system for predicting heart disease risk based on patient physiological and diagnostic indicators from the canonical UCI Cleveland Clinic Heart Disease dataset.

### Core MLOps Engineering Pillars Delivered:
1. **Automated Data Acquisition & EDA:** Programmatic ingestion script (`src/data/download_data.py`), missing value profiling, and high-resolution exploratory data analysis across continuous and categorical clinical covariates.
2. **Leakage-Free Feature Pipeline:** Construction of a Scikit-learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')` embedded inside reusable pipelines.
3. **Multi-Model Development & 5-Fold Stratified CV:** Comparative benchmarking across Logistic Regression, Random Forest, and Gradient Boosting with exhaustive `GridSearchCV` hyperparameter tuning.
4. **Experiment Tracking via MLflow:** Centralized tracking of hyperparameters, training/validation metrics, confusion matrices, ROC curves, and serialized artifacts under experiment `heart-disease-prediction-2025af05183`.
5. **Model Packaging & Deterministic Reproducibility:** Serialization of champion model pipeline into `models/heart_disease_model.joblib`, accompanied by strict dependency pinning in `requirements.txt` and `environment.yml`.
6. **Automated Testing Suite (14 Tests, 93% Coverage):** Comprehensive tests in `tests/` validating data transformations, schema invariance, model sanity thresholds (>80% accuracy), and FastAPI serving endpoints.
7. **CI/CD Pipeline via GitHub Actions:** Fully automated 4-stage pipeline (`.github/workflows/mlops-pipeline.yml`) executing linting (`flake8`), test execution with coverage, automated retraining with performance gating, and container smoke testing.
8. **Hardened Multi-Stage Containerization:** Production-grade `Dockerfile` using `python:3.9-slim`, non-root execution (`appuser:appgroup`), and native `HEALTHCHECK`.
9. **Kubernetes Orchestration & Helm Packaging:** Declarative manifests (`k8s/`) and production Helm chart (`helm/heart-disease-api/`) with RollingUpdate deployments, LoadBalancer service, Ingress routing, and Horizontal Pod Autoscaling (HPA).
10. **Observability & Monitoring:** Prometheus custom metrics exporter (`/metrics`) and Grafana dashboard (`monitoring/grafana-dashboard.json`) tracking real-time request volume, latency percentiles (p95/p99), and diagnostic risk stratification.

---

## 2. End-to-End System Architecture

The architecture follows a modular, 4-tier design ensuring clear separation of concerns, loose coupling, and high testability:

```
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END MLOPS ARCHITECTURE                                   |
|                        Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)                    |
+----------------------------------------------------------------------------------------------------+

  [ Tier 1: Data Ingestion & Preprocessing ]
    UCI Repository  -->  download_data.py  -->  ColumnTransformer (Median Imputer, StandardScaler, OHE)
                                                        |
                                                Stratified 80/20 Split
                                                        |
  [ Tier 2: Model Training & Experiment Tracking ]     v
    5-Fold Stratified CV  -->  GridSearchCV Tuning  -->  MLflow Registry (Params, Metrics, Artifacts)
                                                        |
                                            Champion Pipeline (.joblib)
                                                        |
  [ Tier 3: CI/CD & Containerization ]                  v
    GitHub Actions  -->  Flake8 Lint  -->  Pytest (93% cov)  -->  Docker Multi-Stage Build (Non-root)
                                                                        |
  [ Tier 4: Orchestration & Observability ]                             v
    Kubernetes / Helm  -->  2 Replicas + HPA  -->  LoadBalancer Service & Ingress (Port 8000)
                                                        |
                                      +-----------------+-----------------+
                                      v                                   v
                             FastAPI Serving                     Prometheus & Grafana
                         (/predict, /health)                     (Metrics, Latency, Drift)
```

---

## 3. Exploratory Data Analysis & Clinical Insights

### 3.1 Dataset Schema & Clinical Significance

The UCI Cleveland dataset comprises 303 patient records across 14 clinical features:

| Feature Name | Clinical Meaning | Data Type | Preprocessing Strategy |
|---|---|---|---|
| `age` | Patient age in years (29 - 77) | Continuous | Median Imputation + StandardScaler |
| `sex` | Biological sex (1 = male; 0 = female) | Binary Categorical | OneHotEncoder |
| `cp` | Chest pain type (1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic) | Multi-class Categorical | OneHotEncoder |
| `trestbps` | Resting blood pressure in mm Hg (94 - 200) | Continuous | Median Imputation + StandardScaler |
| `chol` | Serum cholestoral in mg/dl (126 - 564) | Continuous | Median Imputation + StandardScaler |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | Binary Categorical | OneHotEncoder |
| `restecg` | Resting electrocardiographic results (0, 1, 2) | Multi-class Categorical | OneHotEncoder |
| `thalach` | Maximum heart rate achieved during exercise (71 - 202 bpm) | Continuous | Median Imputation + StandardScaler |
| `exang` | Exercise-induced angina (1 = yes; 0 = no) | Binary Categorical | OneHotEncoder |
| `oldpeak` | ST depression induced by exercise relative to rest (0.0 - 6.2 mm) | Continuous | Median Imputation + StandardScaler |
| `slope` | Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping) | Multi-class Categorical | OneHotEncoder |
| `ca` | Number of major coronary vessels colored by fluoroscopy (0 - 3) | Discrete Ordinal | Mode Imputation + OneHotEncoder |
| `thal` | Thallium heart scan (3: normal, 6: fixed defect, 7: reversible defect) | Multi-class Categorical | Mode Imputation + OneHotEncoder |
| `target` | Angiographic disease presence (0 = Absence, 1 = Presence) | Binary Target | Ground Truth Label |

### 3.2 Key Clinical Findings from EDA
- **Class Balance:** The target distribution consists of 164 negative cases (54.1%) and 139 positive cases (45.9%). Because the classes are naturally balanced, synthetic resampling techniques (such as SMOTE) were avoided to preserve true posterior calibration.
- **Max Heart Rate (thalach):** Displayed a strong negative correlation with cardiac disease ($r = -0.42$). Patients with heart disease displayed lower peak heart rates (mean 139.3 bpm) than healthy patients (mean 158.4 bpm), reflecting chronotropic incompetence.
- **ST Depression (oldpeak):** Showed marked right-skewness and strong positive risk association ($r = +0.42$). Severe ST depressions exceeding 2.0 mm predominantly occurred in diseased patients.
- **Asymptomatic Angina (cp=4):** Represented the single highest category associated with positive heart disease cases, demonstrating that silent ischemia is clinically prevalent in high-risk patients.

---

## 4. Feature Engineering & Leakage-Free Preprocessing

A critical challenge in production machine learning is train-test data leakage occurring during feature preprocessing (such as scaling using the full dataset before splitting).

To guarantee strict isolation:
1. The raw dataset is split into **80% training (242 samples)** and **20% held-out test (61 samples)** using `train_test_split` with `stratify=y` and `random_state=42`.
2. A `ColumnTransformer` is assembled:
   - **Numerical Pathway:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])`
   - **Categorical Pathway:** `Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])`
3. The preprocessor is fitted **strictly on the training partition** during cross-validation folds and serialized as part of the overall pipeline.

---

## 5. Model Development, Tuning & Cross-Validation

Three distinct classification architectures were trained and tuned using 5-Fold Stratified Cross-Validation:

### 5.1 Performance Benchmark Matrix

| Model Architecture | Tuned Hyperparameters | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Champion)** | $C=0.1$, penalty='l2', solver='lbfgs' | **0.898 ± 0.051** | **86.89%** | **83.33%** | **89.29%** | **86.21%** | **0.9632** |
| **Random Forest** | $n\_estimators=100$, $max\_depth=5$, $min\_split=10$ | 0.902 ± 0.048 | 86.89% | 85.71% | 85.71% | 85.71% | 0.9383 |
| **Gradient Boosting** | $n\_estimators=50$, $learning\_rate=0.05$, $depth=3$ | 0.894 ± 0.054 | 88.52% | 83.87% | 92.86% | 88.14% | 0.9470 |

### 5.2 Model Selection Rationale
Logistic Regression with L2 regularization ($C=0.1$) was designated as the production **Champion Model**:
1. **Superior Discriminative Ability:** Achieved a held-out test ROC-AUC of **0.9632**, outperforming tree ensembles on probability calibration.
2. **High Clinical Sensitivity:** Yielded an **89.29% Recall** on cardiac disease cases (25 / 28 true positive cases detected), minimizing dangerous false negatives.
3. **Interpretability:** Logistic regression weights translate directly into log-odds multipliers, satisfying regulatory requirements for clinical explainability.

---

## 6. Experiment Tracking via MLflow

All model runs, tuning experiments, parameters, and metrics were automatically tracked using MLflow:
- **Experiment Name:** `heart-disease-prediction-2025af05183`
- **Logged Parameters:** `C`, `penalty`, `solver`, `n_estimators`, `max_depth`, `min_samples_split`, `learning_rate`, `model_type`, `roll_number`.
- **Logged Metrics:** 5-fold CV ROC-AUC mean & std, CV Accuracy mean, Test Accuracy, Test Precision, Test Recall, Test F1, Test ROC-AUC.
- **Logged Artifacts:** Serialized model pipelines, ROC comparison curves, confusion matrices, feature importance plots, classification reports.

---

## 7. Automated Testing & CI/CD Pipeline

### 7.1 Pytest Test Suite Summary
The test suite consists of 14 automated tests achieving **93% overall code coverage**:

```
Name                     Stmts   Miss  Cover
--------------------------------------------
src/api/app.py              73     10    86%
src/api/monitoring.py        7      0   100%
src/api/schemas.py          38      0   100%
src/data/preprocess.py      53      3    94%
src/models/predict.py       42      2    95%
src/utils/logger.py         14      0   100%
--------------------------------------------
TOTAL                      227     15    93%
============================== 14 passed in 2.44s ==============================
```

### 7.2 GitHub Actions Workflow (`.github/workflows/mlops-pipeline.yml`)
The workflow automates four stages:
1. **Lint:** `flake8` syntax and complexity verification.
2. **Test:** Automated execution of `pytest -v --cov=src` and coverage XML artifact upload.
3. **Train & Validate:** Automated dataset ingestion, model retraining, MLflow logging, and benchmark assertion (`accuracy >= 0.80`, `roc_auc >= 0.85`).
4. **Build & Smoke Test:** Multi-stage Docker image build and live verification against `/health` and `/predict`.

---

## 8. Model Containerization with Docker

A production multi-stage Dockerfile (`docker/Dockerfile`) was built to minimize attack surface and enforce isolation:
- **Stage 1 (Builder):** Uses `python:3.9-slim`, compiles wheel dependencies into `/root/.local`.
- **Stage 2 (Runner):** Uses a clean `python:3.9-slim` base, creates an unprivileged user `appuser (uid 1001)`, copies pre-built wheels, exposes port 8000.
- **Built-in Healthcheck:** Configured with `curl -f http://localhost:8000/health || exit 1` every 15 seconds.

### Multi-Container Orchestration (`docker/docker-compose.yml`)
Brings up the complete production serving stack:
1. `heart-disease-api`: FastAPI model server on port 8000.
2. `prometheus`: Scrapes `/metrics` every 5 seconds on port 9090.
3. `grafana`: Real-time dashboard visualization on port 3000.

---

## 9. Kubernetes Orchestration & Helm Deployment

### 9.1 Declarative Kubernetes Manifests (`k8s/`)
- **Namespace (`namespace.yaml`):** Isolated namespace `mlops-heart-disease`.
- **ConfigMap (`configmap.yaml`):** Application runtime configuration and student metadata.
- **Deployment (`deployment.yaml`):** 2 Pod replicas, `RollingUpdate` strategy (`maxSurge: 1, maxUnavailable: 0`), non-root security context, CPU/Memory requests & limits, and Liveness/Readiness probes targeting `/health`.
- **Service (`service.yaml`):** `LoadBalancer` exposing port 8000.
- **Ingress (`ingress.yaml`):** Host-based routing for `heart-disease.bits-pilani.local`.
- **Horizontal Pod Autoscaler (`hpa.yaml`):** Autoscales replicas between 2 and 5 based on 70% target CPU utilization.

### 9.2 Helm Chart (`helm/heart-disease-api/`)
A parameterized Helm v3 chart was authored with `Chart.yaml`, `values.yaml`, and production templates supporting seamless releases across Minikube, GKE, EKS, or AKS.

---

## 10. Monitoring, Logging & Observability

### 10.1 Prometheus Custom Metrics Exporter
The FastAPI application exposes Prometheus metrics at `/metrics`:
- `api_requests_total`: Counter partitioned by HTTP method, endpoint, and status code.
- `api_request_duration_seconds`: Histogram measuring latency percentiles across endpoints.
- `model_predictions_total`: Counter tracking predictions partitioned by diagnosis (`Absence` vs `Presence`) and risk tier (`Low`, `Moderate`, `High`).
- `model_prediction_confidence`: Histogram tracking prediction confidence score distribution.

### 10.2 Structured Request Logging
Every HTTP transaction is recorded in structured format capturing client IP, HTTP verb, route path, status code, and latency in milliseconds.

---

## 11. Production Verification & Quickstart Runbook

### Local Environment Setup
```bash
# 1. Clone or unpack assignment repository
cd 2025af05183_MLOps_Assignment1

# 2. Create virtual environment and install dependencies
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run unit tests and coverage verification
pytest -v --cov=src tests/

# 4. Ingest data, train models, and track with MLflow
python src/data/download_data.py
python src/data/eda.py
python src/models/train.py

# 5. Start FastAPI local serving API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Docker Compose Stack Execution
```bash
docker-compose -f docker/docker-compose.yml up --build -d
# Endpoints accessible:
# FastAPI API: http://localhost:8000/docs
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000 (admin / admin)
```

### Kubernetes & Helm Deployment Execution
```bash
# Apply declarative manifests
kubectl apply -f k8s/

# Verify cluster resources
kubectl get pods,svc,ingress,hpa -n mlops-heart-disease

# Or deploy via Helm
helm install heart-disease-api ./helm/heart-disease-api/ -n mlops-heart-disease
```

### Sample Inference Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63.0, "sex": 1.0, "cp": 1.0, "trestbps": 145.0, "chol": 233.0,
    "fbs": 1.0, "restecg": 2.0, "thalach": 150.0, "exang": 0.0,
    "oldpeak": 2.3, "slope": 3.0, "ca": 0.0, "thal": 6.0
  }'
```

---

## 12. Conclusion & Academic Integrity Statement

This assignment successfully fulfills all technical requirements and pedagogical objectives prescribed in AIMLCZG523 MLOps Assignment 01. The pipeline demonstrates automated data ingestion, zero-leakage feature engineering, 5-fold cross-validation with MLflow experiment tracking, 93% test coverage, GitHub Actions CI/CD automation, production containerization, declarative Kubernetes deployment, and real-time observability.

**Academic Integrity Declaration:**  
I hereby declare that this assignment represents my own original work conducted independently in accordance with the academic integrity guidelines of BITS Pilani Work Integrated Learning Programmes (WILP).

**Submitted By:**  
Ram Narayan  
Student ID: 2025af05183  
Email: 2025af05183@wilp.bits-pilani.ac.in  
BITS Pilani - AIMLCZG523 Machine Learning Operations
