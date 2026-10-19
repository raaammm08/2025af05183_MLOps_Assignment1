"""Data Preprocessing and Feature Engineering Pipeline.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger("data_preprocessor")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_disease_raw.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_disease_cleaned.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

NUMERICAL_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "target"


def build_preprocessor() -> ColumnTransformer:
    """Creates a reusable scikit-learn ColumnTransformer for numerical and categorical features."""
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def clean_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw UCI Cleveland dataframe into binary classification dataset."""
    df = raw_df.copy()
    if "num" in df.columns:
        df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])
    elif TARGET_COLUMN not in df.columns:
        raise ValueError(f"Neither 'num' nor '{TARGET_COLUMN}' found in dataframe.")
    return df


def prepare_datasets(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Loads raw dataset, cleans it, performs stratified splitting, and saves artifacts."""
    logger.info(f"Loading raw dataset from {raw_path}")
    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_raw_data(raw_df)

    # Save cleaned data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Cleaned dataset saved to {PROCESSED_DATA_PATH} ({len(cleaned_df)} rows)")

    X = cleaned_df[ALL_FEATURES]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Stratified Split completed: Train size={len(X_train)}, Test size={len(X_test)}")

    # Save train and test sets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Train/Test sets saved to {TRAIN_DATA_PATH} and {TEST_DATA_PATH}")

    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    prepare_datasets()
