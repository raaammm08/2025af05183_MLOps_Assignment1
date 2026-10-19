"""Model Training, Hyperparameter Tuning, Evaluation, and MLflow Experiment Tracking.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import mlflow
import mlflow.sklearn

from src.data.preprocess import prepare_datasets, ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils.logger import get_logger

logger = get_logger("model_training")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "screenshots")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics_summary.json")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")


def plot_roc_curves(models_dict, X_test, y_test, output_path):
    """Plots and saves ROC curves for all models."""
    plt.figure(figsize=(8, 6), dpi=300)
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.7)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC comparison curve to {output_path}")


def plot_confusion_matrices(models_dict, X_test, y_test, output_path):
    """Generates side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 5), dpi=300)
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["No Disease (0)", "Disease (1)"],
            yticklabels=["No Disease (0)", "Disease (1)"],
        )
        ax.set_title(f"Confusion Matrix: {name}", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to {output_path}")


def plot_feature_importance(rf_pipeline, output_path):
    """Extracts and plots Random Forest feature importances after one-hot encoding."""
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf = rf_pipeline.named_steps["classifier"]

    # Retrieve feature names after transformation
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + encoded_cat_names

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:12]  # Top 12 features

    plt.figure(figsize=(10, 6), dpi=300)
    palette = sns.color_palette("viridis", len(indices))
    plt.barh(range(len(indices)), importances[indices][::-1], color=palette, align="center", edgecolor="black")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]], fontweight="bold")
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontweight="bold")
    plt.title("Random Forest - Top Predictive Clinical Features", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


def train_and_evaluate():
    """Executes full training pipeline with hyperparameter tuning and MLflow logging."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_datasets()

    # 2. Setup MLflow
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    experiment_name = "heart-disease-prediction-2025af05183"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking initialized: Experiment '{experiment_name}' at {MLFLOW_TRACKING_URI}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"]
    }

    # Model 2: Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__criterion": ["gini", "entropy"]
    }

    # Model 3: Gradient Boosting
    gb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    gb_param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5]
    }

    candidates = [
        ("Logistic_Regression", lr_pipeline, lr_param_grid),
        ("Random_Forest", rf_pipeline, rf_param_grid),
        ("Gradient_Boosting", gb_pipeline, gb_param_grid)
    ]

    best_models = {}
    metrics_summary = {}

    for name, pipeline, param_grid in candidates:
        logger.info(f"--- Training and Tuning: {name} ---")
        with mlflow.start_run(run_name=f"{name}_Tuning_2025af05183"):
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True
            )
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_models[name] = best_estimator

            # Log Hyperparameters
            clean_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
            mlflow.log_params(clean_params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("roll_number", "2025af05183")
            mlflow.log_param("student_name", "Ram Narayan")

            # 5-Fold Cross Validation scores on training data
            cv_roc_scores = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring="roc_auc")
            cv_acc_scores = cross_val_score(best_estimator, X_train, y_train, cv=cv, scoring="accuracy")

            # Evaluation on Held-Out Test Set
            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            metrics = {
                "cv_roc_auc_mean": float(cv_roc_scores.mean()),
                "cv_roc_auc_std": float(cv_roc_scores.std()),
                "cv_accuracy_mean": float(cv_acc_scores.mean()),
                "test_accuracy": float(acc),
                "test_precision": float(prec),
                "test_recall": float(rec),
                "test_f1_score": float(f1),
                "test_roc_auc": float(auc)
            }
            mlflow.log_metrics(metrics)
            metrics_summary[name] = {
                "best_params": clean_params,
                "metrics": metrics,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            logger.info(
                f"[{name}] Test Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
            )

            # Log model with MLflow
            mlflow.sklearn.log_model(best_estimator, artifact_path=f"model_{name}")

    # Generate Visualizations for all models
    roc_plot_path = os.path.join(REPORTS_DIR, "04_model_roc_curves.png")
    plot_roc_curves(best_models, X_test, y_test, roc_plot_path)

    cm_plot_path = os.path.join(REPORTS_DIR, "05_confusion_matrices.png")
    plot_confusion_matrices(best_models, X_test, y_test, cm_plot_path)

    rf_feat_path = os.path.join(REPORTS_DIR, "06_feature_importance.png")
    plot_feature_importance(best_models["Random_Forest"], rf_feat_path)

    # Select Best Model based on Test ROC-AUC & F1-score
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]["metrics"]["test_roc_auc"])
    champion_model = best_models[best_model_name]
    logger.info(f"Champion Model selected: {best_model_name} (ROC-AUC: {metrics_summary[best_model_name]['metrics']['test_roc_auc']:.4f})")

    # Save final model for production deployment
    final_model_path = os.path.join(MODELS_DIR, "heart_disease_model.joblib")
    joblib.dump(champion_model, final_model_path)
    logger.info(f"Champion model pipeline saved to {final_model_path}")

    # Save metrics summary
    metrics_summary["champion_model"] = best_model_name
    metrics_summary["author"] = {
        "name": "Ram Narayan",
        "roll_number": "2025af05183",
        "email": "2025af05183@wilp.bits-pilani.ac.in",
        "course": "AIMLCZG523 Machine Learning Operations"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Metrics summary saved to {METRICS_PATH}")

    return champion_model, metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
