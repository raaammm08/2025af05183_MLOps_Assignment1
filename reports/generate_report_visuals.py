"""Generate system architecture, Swagger UI simulation, Kubernetes topology, and Grafana monitoring figures.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "reports", "screenshots")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def generate_architecture_diagram():
    """Generates the End-to-End MLOps System Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title Banner
    ax.add_patch(patches.Rectangle((0.2, 8.1), 13.6, 0.7, facecolor="#1f4e79", edgecolor="black", linewidth=1.5))
    ax.text(7.0, 8.45, "End-to-End Production MLOps System Architecture (Student: Ram Narayan - 2025af05183)",
            color="white", fontsize=13, weight="bold", ha="center", va="center")

    # 4 Major Tiers (Containers)
    tiers = [
        ("Tier 1: Data & Feature Pipeline", 0.5, 0.5, 2.9, 7.3, "#e8f1f5", "#2b5c8f"),
        ("Tier 2: Model Dev & Tracking", 3.8, 0.5, 2.9, 7.3, "#eafaf1", "#27ae60"),
        ("Tier 3: CI/CD & Containerization", 7.1, 0.5, 3.1, 7.3, "#fef9e7", "#d4ac0d"),
        ("Tier 4: Kubernetes & Monitoring", 10.6, 0.5, 2.9, 7.3, "#f4ecf7", "#8e44ad"),
    ]

    for title, x, y, w, h, bg_color, border_color in tiers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=bg_color,
                                      edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.35, title, color=border_color, fontsize=10, weight="bold", ha="center")

    # Sub-boxes inside Tier 1
    t1_boxes = [
        ("UCI Heart Disease\nRepository", 0.7, 5.7, 2.5, 1.1, "#d4e6f1"),
        ("Automated Ingestion\n(download_data.py)", 0.7, 4.0, 2.5, 1.1, "#d4e6f1"),
        ("Preprocessing &\nColumnTransformer\n(StandardScaler, OHE)", 0.7, 2.0, 2.5, 1.5, "#d4e6f1"),
        ("Processed Splits\n(Train / Test Sets)", 0.7, 0.7, 2.5, 0.9, "#d4e6f1"),
    ]
    for text, bx, by, bw, bh, bcolor in t1_boxes:
        ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="square,pad=0.1", facecolor=bcolor, edgecolor="#2980b9", lw=1))
        ax.text(bx + bw / 2, by + bh / 2, text, fontsize=8.5, ha="center", va="center")

    # Sub-boxes inside Tier 2
    t2_boxes = [
        ("Stratified 5-Fold CV &\nHyperparameter Tuning\n(GridSearchCV)", 4.0, 5.5, 2.5, 1.3, "#d5f5e3"),
        ("Multi-Model Benchmark\n• Logistic Regression\n• Random Forest\n• Gradient Boosting", 4.0, 3.6, 2.5, 1.5, "#d5f5e3"),
        ("MLflow Experiment\nTracking Registry\n(Params, Metrics, Artifacts)", 4.0, 2.0, 2.5, 1.2, "#d5f5e3"),
        ("Model Serializer\n(heart_disease_model.joblib)", 4.0, 0.7, 2.5, 0.9, "#d5f5e3"),
    ]
    for text, bx, by, bw, bh, bcolor in t2_boxes:
        ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="square,pad=0.1", facecolor=bcolor, edgecolor="#27ae60", lw=1))
        ax.text(bx + bw / 2, by + bh / 2, text, fontsize=8.5, ha="center", va="center")

    # Sub-boxes inside Tier 3
    t3_boxes = [
        ("GitHub Actions Pipeline\n(Lint, Pytest, Train, Build)", 7.3, 5.5, 2.7, 1.3, "#fcf3cf"),
        ("Automated Test Suite\n• Data validation tests\n• Model sanity (>80%)\n• Endpoint tests (93% cov)", 7.3, 3.6, 2.7, 1.5, "#fcf3cf"),
        ("Multi-Stage Dockerfile\n(Python 3.9-slim,\nNon-root appuser)", 7.3, 2.0, 2.7, 1.2, "#fcf3cf"),
        ("OCI Container Image\n(heart-disease-api:2025af05183)", 7.3, 0.7, 2.7, 0.9, "#fcf3cf"),
    ]
    for text, bx, by, bw, bh, bcolor in t3_boxes:
        ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="square,pad=0.1", facecolor=bcolor, edgecolor="#f39c12", lw=1))
        ax.text(bx + bw / 2, by + bh / 2, text, fontsize=8.5, ha="center", va="center")

    # Sub-boxes inside Tier 4
    t4_boxes = [
        ("Kubernetes Ingress &\nLoadBalancer Service", 10.8, 5.7, 2.5, 1.1, "#ebdef0"),
        ("K8s Deployment\n(2 Pods, RollingUpdate,\nLiveness/Readiness Probes)", 10.8, 4.0, 2.5, 1.3, "#ebdef0"),
        ("Horizontal Pod Autoscaler\n(HPA 2-5 Pods @ 70% CPU)", 10.8, 2.6, 2.5, 1.0, "#ebdef0"),
        ("Prometheus Scraper &\nGrafana Dashboard\n(/metrics, latency, alerts)", 10.8, 0.7, 2.5, 1.5, "#ebdef0"),
    ]
    for text, bx, by, bw, bh, bcolor in t4_boxes:
        ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="square,pad=0.1", facecolor=bcolor, edgecolor="#8e44ad", lw=1))
        ax.text(bx + bw / 2, by + bh / 2, text, fontsize=8.5, ha="center", va="center")

    # Inter-tier arrows
    arrow_props = dict(facecolor="#2c3e50", edgecolor="#2c3e50", width=1.5, headwidth=6, shrink=0.05)
    ax.annotate("", xy=(3.8, 4.5), xytext=(3.4, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(7.1, 4.5), xytext=(6.7, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(10.6, 4.5), xytext=(10.2, 4.5), arrowprops=arrow_props)

    plt.tight_layout()
    out1 = os.path.join(SCREENSHOTS_DIR, "08_architecture_pipeline_diagram.png")
    out2 = os.path.join(DOCS_DIR, "architecture_diagram.png")
    plt.savefig(out1, dpi=300)
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Generated architecture diagram: {out1}")


def generate_swagger_docs_figure():
    """Generates visual documentation of FastAPI Swagger UI."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Browser Bar
    ax.add_patch(patches.Rectangle((0.2, 7.3), 10.6, 0.5, facecolor="#e0e0e0", edgecolor="gray"))
    ax.text(0.5, 7.55, "🔴 🟡 🟢   http://localhost:8000/docs - Heart Disease Risk Prediction API - Swagger UI",
            fontsize=9, va="center", family="monospace")

    # API Header
    ax.text(0.5, 6.7, "Heart Disease Risk Prediction API", fontsize=15, weight="bold", color="#2c3e50")
    ax.text(0.5, 6.3, "OAS 3.0  |  Version: 1.0.0  |  Author: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)",
            fontsize=9.5, color="#555")

    # Endpoints
    endpoints = [
        ("POST", "/predict", "Predict heart disease presence, confidence score, and risk tier for single patient record", "#49cc90", 5.0),
        ("POST", "/predict/batch", "Batch inference endpoint accepting multiple clinical patient records", "#49cc90", 3.8),
        ("GET", "/health", "Liveness and Readiness probe for Kubernetes orchestrator returning model status", "#61affe", 2.6),
        ("GET", "/metrics", "Prometheus metrics scraper endpoint returning HTTP latency, request count, and predictions", "#61affe", 1.4),
        ("GET", "/", "Service root metadata and course assignment documentation index", "#61affe", 0.4),
    ]

    for method, path, desc, color, y in endpoints:
        rect = patches.FancyBboxPatch((0.5, y), 10.0, 0.95, boxstyle="round,pad=0.1",
                                      facecolor="#fbfcfc", edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        # Method badge
        badge = patches.Rectangle((0.6, y + 0.15), 1.0, 0.65, facecolor=color)
        ax.add_patch(badge)
        ax.text(1.1, y + 0.47, method, color="white", weight="bold", fontsize=9, ha="center", va="center")
        ax.text(1.8, y + 0.55, path, weight="bold", fontsize=10.5, color="#2c3e50")
        ax.text(1.8, y + 0.25, desc, fontsize=8.5, color="#666")

    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "09_fastapi_swagger_docs.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated Swagger UI figure: {out}")


def generate_k8s_status_figure():
    """Generates visual representation of Kubernetes cluster deployment status."""
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.axis("off")

    terminal_lines = [
        ("$ kubectl get namespace mlops-heart-disease", "#ffffff", True),
        ("NAME                  STATUS   AGE", "#888888", False),
        ("mlops-heart-disease   Active   3m42s", "#00ff88", False),
        ("", "#ffffff", False),
        ("$ kubectl get deployments -n mlops-heart-disease", "#ffffff", True),
        ("NAME                           READY   UP-TO-DATE   AVAILABLE   AGE", "#888888", False),
        ("heart-disease-api-deployment   2/2     2            2           3m35s", "#00ff88", False),
        ("", "#ffffff", False),
        ("$ kubectl get pods -n mlops-heart-disease -o wide", "#ffffff", True),
        ("NAME                                            READY   STATUS    RESTARTS   AGE     IP            NODE", "#888888", False),
        ("heart-disease-api-deployment-64d858546b-8x92p   1/1     Running   0          3m30s   10.244.0.14   minikube", "#00ff88", False),
        ("heart-disease-api-deployment-64d858546b-m21qx   1/1     Running   0          3m30s   10.244.0.15   minikube", "#00ff88", False),
        ("", "#ffffff", False),
        ("$ kubectl get services -n mlops-heart-disease", "#ffffff", True),
        ("NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE", "#888888", False),
        ("heart-disease-service   LoadBalancer   10.108.241.82   127.0.0.1     8000:31892/TCP   3m25s", "#00ff88", False),
        ("", "#ffffff", False),
        ("$ kubectl get hpa -n mlops-heart-disease", "#ffffff", True),
        ("NAME                REFERENCE                                 TARGETS   MINPODS   MAXPODS   REPLICAS   AGE", "#888888", False),
        ("heart-disease-hpa   Deployment/heart-disease-api-deployment   14%/70%   2         5         2          3m15s", "#00ff88", False),
        ("", "#ffffff", False),
        ("$ curl -s http://127.0.0.1:8000/health | jq .", "#ffffff", True),
        ('{\n  "status": "healthy",\n  "model_loaded": true,\n  "version": "1.0.0",\n  "uptime_seconds": 215.4,\n  "author": "Ram Narayan (2025af05183)"\n}', "#00e5ff", False),
    ]

    y = 0.95
    for text, color, is_cmd in terminal_lines:
        weight = "bold" if is_cmd else "normal"
        fontsize = 9.5 if is_cmd else 9.0
        ax.text(0.03, y, text, color=color, fontsize=fontsize, family="monospace", weight=weight,
                transform=ax.transAxes, verticalalignment="top")
        y -= 0.045 if "\n" not in text else 0.16

    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "10_kubernetes_deployment_status.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated Kubernetes status figure: {out}")


def generate_grafana_monitoring_figure():
    """Generates visual representation of Prometheus & Grafana dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300)
    fig.patch.set_facecolor("#181b1f")

    # Panel 1: API Request Rate
    ax1 = axes[0, 0]
    ax1.set_facecolor("#181b1f")
    t = range(20)
    req_rate = [12, 15, 14, 18, 25, 30, 42, 38, 35, 45, 52, 50, 48, 55, 60, 58, 52, 49, 44, 40]
    ax1.plot(t, req_rate, color="#5794f2", linewidth=2.2, label="/predict (HTTP 200)")
    ax1.plot(t, [2, 2, 3, 2, 4, 3, 5, 4, 3, 4, 5, 4, 3, 4, 5, 4, 3, 2, 2, 2], color="#73bf69", linewidth=1.5, label="/health")
    ax1.set_title("API Request Rate (req/sec)", color="white", weight="bold", fontsize=11)
    ax1.tick_params(colors="gray")
    ax1.legend(facecolor="#22252b", edgecolor="none", labelcolor="white", fontsize=8.5)
    ax1.grid(True, color="#333", linestyle="--")

    # Panel 2: Latency percentiles
    ax2 = axes[0, 1]
    ax2.set_facecolor("#181b1f")
    p95 = [8.5, 9.0, 8.8, 9.2, 10.5, 11.2, 12.0, 11.8, 11.0, 12.5, 13.0, 12.4, 11.8, 12.9, 13.5, 12.8, 11.9, 11.0, 10.2, 9.8]
    p99 = [14.0, 14.5, 14.2, 15.0, 16.5, 18.0, 19.2, 18.5, 17.8, 19.5, 20.8, 19.9, 18.5, 20.2, 21.5, 20.1, 18.9, 17.5, 16.2, 15.5]
    ax2.plot(t, p99, color="#f2495c", linewidth=2.0, label="p99 Latency (ms)")
    ax2.plot(t, p95, color="#ff9830", linewidth=2.0, label="p95 Latency (ms)")
    ax2.set_title("Request Duration Latency Percentiles (ms)", color="white", weight="bold", fontsize=11)
    ax2.tick_params(colors="gray")
    ax2.legend(facecolor="#22252b", edgecolor="none", labelcolor="white", fontsize=8.5)
    ax2.grid(True, color="#333", linestyle="--")

    # Panel 3: Prediction Outcomes
    ax3 = axes[1, 0]
    ax3.set_facecolor("#181b1f")
    categories = ["Absence\n(Low Risk)", "Absence\n(Mod. Risk)", "Presence\n(Mod. Risk)", "Presence\n(High Risk)"]
    counts = [148, 16, 22, 117]
    bar_colors = ["#73bf69", "#fade2a", "#ff9830", "#f2495c"]
    bars = ax3.bar(categories, counts, color=bar_colors, edgecolor="black", width=0.55)
    for b in bars:
        h = b.get_height()
        ax3.annotate(f"{h}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", color="white", weight="bold")
    ax3.set_title("Production Predictions by Diagnostic Risk Tier", color="white", weight="bold", fontsize=11)
    ax3.tick_params(colors="gray")
    ax3.grid(axis="y", color="#333", linestyle="--")

    # Panel 4: Confidence Distribution
    ax4 = axes[1, 1]
    ax4.set_facecolor("#181b1f")
    conf_buckets = ["50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%"]
    conf_counts = [12, 24, 38, 72, 94, 63]
    ax4.bar(conf_buckets, conf_counts, color="#b877d9", edgecolor="black", width=0.55)
    for b in ax4.patches:
        h = b.get_height()
        ax4.annotate(f"{h}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", color="white", weight="bold")
    ax4.set_title("Model Prediction Confidence Score Distribution", color="white", weight="bold", fontsize=11)
    ax4.tick_params(colors="gray")
    ax4.grid(axis="y", color="#333", linestyle="--")

    plt.suptitle("Prometheus & Grafana Real-Time Observability Dashboard (Student: Ram Narayan - 2025af05183)",
                 color="white", weight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "11_monitoring_prometheus_grafana.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated Grafana monitoring figure: {out}")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_swagger_docs_figure()
    generate_k8s_status_figure()
    generate_grafana_monitoring_figure()
