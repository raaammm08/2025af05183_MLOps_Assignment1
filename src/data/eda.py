"""Exploratory Data Analysis (EDA) and Visualization Generator.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger("eda_generator")

# Set visualization styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_disease_raw.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "screenshots")


def run_eda(input_path: str = RAW_PATH, output_dir: str = FIGURES_DIR):
    """Executes comprehensive EDA and generates high-resolution figures."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Prepare target column: 0=Absence (num=0), 1=Presence (num>=1)
    df["target"] = (df["num"] > 0).astype(int)
    target_labels = {0: "Absence (No Disease)", 1: "Presence (Disease)"}

    # Summary statistics
    logger.info("Computing summary statistics...")
    summary = df.describe().round(2)
    missing_info = df.isnull().sum()
    logger.info(f"Missing values count:\n{missing_info[missing_info > 0]}")

    # ==========================================
    # Figure 1: Class Balance
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = df["target"].value_counts().sort_index()
    palette = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(["Absence (0)", "Presence (1)"], counts.values, color=palette, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{h} ({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("UCI Heart Disease - Class Balance Distribution\n(Target: Presence vs. Absence)", fontweight="bold", pad=15)
    ax.set_ylabel("Patient Count", fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    class_fig_path = os.path.join(output_dir, "03_eda_class_balance.png")
    plt.tight_layout()
    plt.savefig(class_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved class balance plot: {class_fig_path}")

    # ==========================================
    # Figure 2: Numerical Feature Distributions (Histograms & KDE)
    # ==========================================
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    titles = {
        "age": "Age Distribution (years)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Serum Cholesterol (mg/dl)",
        "thalach": "Max Heart Rate Achieved (bpm)",
        "oldpeak": "ST Depression (Exercise vs Rest)"
    }

    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.histplot(data=df, x=col, hue="target", kde=True, palette={0: "#2b5c8f", 1: "#d95f02"},
                     ax=ax, bins=18, element="step", common_norm=False, alpha=0.4)
        ax.set_title(titles[col], fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Use the 6th subplot for resting BP vs Chol by target scatter
    ax_scatter = axes[5]
    sns.scatterplot(data=df, x="trestbps", y="chol", hue="target", palette={0: "#2b5c8f", 1: "#d95f02"},
                    ax=ax_scatter, alpha=0.8, s=40)
    ax_scatter.set_title("Resting BP vs. Serum Cholesterol", fontweight="bold")
    ax_scatter.set_xlabel("trestbps (mm Hg)")
    ax_scatter.set_ylabel("chol (mg/dl)")
    ax_scatter.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Clinical Feature Distributions Stratified by Heart Disease Status", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    dist_fig_path = os.path.join(output_dir, "01_eda_distributions.png")
    plt.savefig(dist_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distributions plot: {dist_fig_path}")

    # ==========================================
    # Figure 3: Correlation Heatmap
    # ==========================================
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    corr_df = df.copy()
    # Fill median for correlation computation
    corr_df = corr_df.fillna(corr_df.median())
    corr_matrix = corr_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Clinical Feature Correlation Matrix (Pearson r)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_fig_path = os.path.join(output_dir, "02_eda_correlation_heatmap.png")
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap: {corr_fig_path}")

    # ==========================================
    # Figure 4: Key Categorical Clinical Predictors
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axes = axes.flatten()

    cat_mappings = [
        ("cp", "Chest Pain Type", {1: "Typ. Angina", 2: "Atyp. Angina", 3: "Non-Anginal", 4: "Asymptomatic"}),
        ("sex", "Patient Sex", {0: "Female", 1: "Male"}),
        ("exang", "Exercise-Induced Angina", {0: "No", 1: "Yes"}),
        ("slope", "ST Segment Slope", {1: "Upsloping", 2: "Flat", 3: "Downsloping"})
    ]

    for idx, (col, title, label_map) in enumerate(cat_mappings):
        ax = axes[idx]
        temp_df = df.copy()
        temp_df[col] = temp_df[col].map(label_map)
        sns.countplot(data=temp_df, x=col, hue="target", palette={0: "#2b5c8f", 1: "#d95f02"}, ax=ax, edgecolor="black", linewidth=0.8)
        ax.set_title(f"Heart Disease by {title}", fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.legend(title="Heart Disease", labels=["Absence", "Presence"])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Heart Disease Risk vs. Key Categorical Biomarkers", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    cat_fig_path = os.path.join(output_dir, "04_eda_categorical_relationships.png")
    plt.savefig(cat_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved categorical relationships plot: {cat_fig_path}")

    logger.info("EDA completed successfully. All figures generated.")


if __name__ == "__main__":
    run_eda()
