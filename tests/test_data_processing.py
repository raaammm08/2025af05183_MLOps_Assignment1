"""Unit tests for Data Processing and Feature Engineering pipeline.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.preprocess import (
    clean_raw_data,
    build_preprocessor,
    prepare_datasets,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RAW_DATA_PATH,
)


def test_raw_data_file_exists():
    """Verify raw dataset exists and has content."""
    assert os.path.exists(RAW_DATA_PATH), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) > 0, "Raw dataset is empty"
    assert "num" in df.columns or TARGET_COLUMN in df.columns


def test_clean_raw_data():
    """Verify raw data cleaning produces binary target without leakage."""
    sample_raw = pd.DataFrame({
        "age": [55.0, 60.0, 45.0],
        "sex": [1.0, 0.0, 1.0],
        "cp": [2.0, 4.0, 1.0],
        "trestbps": [130.0, 140.0, 120.0],
        "chol": [250.0, 290.0, 210.0],
        "fbs": [0.0, 1.0, 0.0],
        "restecg": [0.0, 2.0, 0.0],
        "thalach": [160.0, 130.0, 175.0],
        "exang": [0.0, 1.0, 0.0],
        "oldpeak": [0.5, 2.8, 0.0],
        "slope": [1.0, 2.0, 1.0],
        "ca": [0.0, 2.0, 0.0],
        "thal": [3.0, 7.0, 3.0],
        "num": [0, 3, 0],
    })

    cleaned = clean_raw_data(sample_raw)
    assert TARGET_COLUMN in cleaned.columns
    assert "num" not in cleaned.columns
    assert list(cleaned[TARGET_COLUMN].values) == [0, 1, 0]


def test_build_preprocessor():
    """Verify preprocessor properly imputes and transforms feature columns."""
    preprocessor = build_preprocessor()
    sample_df = pd.DataFrame({
        "age": [55.0, np.nan, 45.0],
        "sex": [1.0, 0.0, 1.0],
        "cp": [2.0, 4.0, 1.0],
        "trestbps": [130.0, 140.0, 120.0],
        "chol": [250.0, 290.0, 210.0],
        "fbs": [0.0, 1.0, 0.0],
        "restecg": [0.0, 2.0, 0.0],
        "thalach": [160.0, 130.0, 175.0],
        "exang": [0.0, 1.0, 0.0],
        "oldpeak": [0.5, 2.8, 0.0],
        "slope": [1.0, 2.0, 1.0],
        "ca": [0.0, np.nan, 0.0],
        "thal": [3.0, 7.0, np.nan],
    })

    transformed = preprocessor.fit_transform(sample_df)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Preprocessed array must not contain NaNs"
    assert transformed.shape[0] == 3


def test_dataset_stratified_split():
    """Verify stratified splitting maintains proportion and creates non-empty splits."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_datasets()
    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Check stratification balance
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05, "Train and test sets should have similar class ratios"
