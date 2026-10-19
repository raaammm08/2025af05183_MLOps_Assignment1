"""Generate visual summary of MLflow experiment runs for reports and documentation.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics_summary.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "reports", "screenshots", "07_mlflow_runs_dashboard.png")


def generate_mlflow_dashboard():
    with open(METRICS_PATH) as f:
        data = json.load(f)

    models = ["Logistic_Regression", "Random_Forest", "Gradient_Boosting"]
    metrics_list = ["test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, (ax_table, ax_bar) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: Table of MLflow Runs
    table_data = []
    headers = ["Run Name", "Model Type", "Key Tuned Parameters", "CV ROC-AUC", "Test Accuracy", "Test Recall", "Test ROC-AUC"]

    for m in models:
        m_data = data[m]
        p_str = ", ".join([f"{k}={v}" for k, v in list(m_data["best_params"].items())[:3]])
        met = m_data["metrics"]
        table_data.append([
            f"{m}_Tuning_2025af05183",
            m.replace("_", " "),
            p_str,
            f"{met['cv_roc_auc_mean']:.3f} ± {met['cv_roc_auc_std']:.3f}",
            f"{met['test_accuracy']:.3f}",
            f"{met['test_recall']:.3f}",
            f"{met['test_roc_auc']:.3f}"
        ])

    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        else:
            if i % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            if i == 1: # Logistic Regression (champion)
                cell.set_edgecolor("#28a745")
                cell.set_linewidth(1.5)

    ax_table.set_title("MLflow Experiment Tracking: Run Comparison Table (Experiment: heart-disease-prediction-2025af05183)",
                       fontweight="bold", fontsize=12, pad=10)

    # Bottom: Bar chart comparing metrics across models
    x = range(len(metric_labels))
    width = 0.25
    palette = ["#2b5c8f", "#2ca02c", "#d95f02"]

    for idx, m in enumerate(models):
        values = [data[m]["metrics"][k] for k in metrics_list]
        pos = [i + (idx - 1) * width for i in x]
        bars = ax_bar.bar(pos, values, width, label=m.replace("_", " "), color=palette[idx], edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontweight="bold", fontsize=10)
    ax_bar.set_ylim(0.70, 1.05)
    ax_bar.set_ylabel("Metric Score", fontweight="bold")
    ax_bar.set_title("Model Performance Benchmark Across Evaluation Metrics (Held-Out Test Set)", fontweight="bold", fontsize=12)
    ax_bar.legend(loc="lower right", frameon=True)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"MLflow dashboard graphic generated at: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_mlflow_dashboard()
