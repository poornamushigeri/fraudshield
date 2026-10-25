"""Turn raw application records into model input, using the same cleaning code as training."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import clean

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def load_model(version: str = "v1"):
    model = joblib.load(MODEL_DIR / f"hgb_{version}.joblib")
    meta = json.loads((MODEL_DIR / f"hgb_{version}.json").read_text())
    return model, meta


def prepare(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Apply the training-time cleaning and return exactly the model's input columns, in training order."""
    raw = raw.copy()
    raw["device_fraud_count"] = 0  # constant column that clean() drops; not a model input
    return clean(raw)[features]