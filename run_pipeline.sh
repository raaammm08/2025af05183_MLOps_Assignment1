#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script: End-to-End MLOps Pipeline
# Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
# BITS Pilani - AIMLCZG523 MLOps Assignment 01
# ==============================================================================

set -e

# Detect python binary
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    PYTEST_CMD="venv/bin/pytest"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    PYTEST_CMD="pytest"
else
    PYTHON_CMD="python"
    PYTEST_CMD="pytest"
fi

echo "========================================================================"
echo " Starting MLOps Pipeline Execution: Heart Disease Risk Classifier"
echo " Author: Ram Narayan | Roll: 2025af05183 | BITS Pilani"
echo " Python Binary: ${PYTHON_CMD}"
echo "========================================================================"

# Set environment
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp

# 1. Automated Data Acquisition
echo ""
echo "[Step 1/6] Ingesting UCI Heart Disease Dataset..."
$PYTHON_CMD src/data/download_data.py

# 2. Exploratory Data Analysis & Visualizations
echo ""
echo "[Step 2/6] Running Exploratory Data Analysis..."
$PYTHON_CMD src/data/eda.py

# 3. Model Training, Tuning & MLflow Experiment Tracking
echo ""
echo "[Step 3/6] Training Models, Evaluating 5-Fold CV, and Logging to MLflow..."
$PYTHON_CMD src/models/train.py
$PYTHON_CMD src/models/generate_mlflow_dashboard.py

# 4. Generate System Architecture & Documentation Figures
echo ""
echo "[Step 4/6] Generating System Figures and Diagrams..."
$PYTHON_CMD reports/generate_report_visuals.py

# 5. Execute Comprehensive Automated Test Suite
echo ""
echo "[Step 5/6] Running Automated Unit Tests with Coverage..."
$PYTEST_CMD -v --cov=src --cov-report=term-missing tests/

# 6. Generate 10-Page Publication-Quality PDF Report
echo ""
echo "[Step 6/6] Compiling 10-Page Final PDF Report..."
$PYTHON_CMD reports/build_10_page_pdf.py

echo ""
echo "========================================================================"
echo " MLOps Pipeline Completed Successfully!"
echo " All deliverables, tests, models, and 10-page report generated."
echo " Output PDF: reports/2025af05183_MLOps_Assignment1_Report.pdf"
echo "========================================================================"
