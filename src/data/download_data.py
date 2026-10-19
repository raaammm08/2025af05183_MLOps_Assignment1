"""Data acquisition script for UCI Heart Disease Dataset.
Student: Ram Narayan (2025af05183@wilp.bits-pilani.ac.in)
BITS Pilani - AIMLCZG523 MLOps Assignment 01
"""

import os
import sys
import io
import urllib.request
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_acquisition")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "heart_disease_raw.csv")

FEATURE_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male; 0 = female
    "cp",        # chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic
    "trestbps",  # resting blood pressure (mm Hg on admission)
    "chol",      # serum cholestoral in mg/dl
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",   # resting electrocardiographic results (0=normal, 1=ST-T wave abnormality, 2=LVH)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes; 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)
    "ca",        # number of major vessels (0-3) colored by flourosopy
    "thal",      # 3 = normal; 6 = fixed defect; 7 = reversable defect
    "num"        # diagnosis of heart disease (angiographic status: 0=absence, 1,2,3,4=presence)
]


def download_dataset(output_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Downloads the Cleveland Heart Disease dataset from UCI repository and saves to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Initiating download from UCI Repository: {UCI_URL}")

    try:
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), names=FEATURE_NAMES, na_values="?")
        logger.info(f"Downloaded {len(df)} records successfully from UCI repository.")
    except Exception as e:
        logger.warning(f"Could not download directly from network ({e}). Checking local cache or backup source.")
        if os.path.exists(output_path):
            logger.info(f"Reading existing cached dataset from {output_path}")
            df = pd.read_csv(output_path)
            return df
        else:
            raise RuntimeError(f"Unable to fetch dataset and no local cache found: {e}")

    df.to_csv(output_path, index=False)
    logger.info(f"Raw dataset successfully saved to: {output_path}")
    return df


if __name__ == "__main__":
    download_dataset()
