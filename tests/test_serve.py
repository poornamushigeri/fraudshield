from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import clean, load_raw
from src.model import model_features
from src.serve import load_model, prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Base.csv"
MODEL = ROOT / "models" / "hgb_v1.joblib"
pytestmark = pytest.mark.skipif(not (DATA.exists() and MODEL.exists()), reason="data or model not available")


def test_serving_input_matches_training_input_and_predictions():
    raw = load_raw(DATA)
    raw = raw[raw["month"] <= 5].sample(2000, random_state=1)  # the test months are never used here
    model, meta = load_model("v1")

    offline = model_features(clean(raw))
    served_input = raw.drop(columns=["fraud_bool", "month", "customer_age", "days_since_request", "device_fraud_count"])
    served = prepare(served_input, meta["features"])

    assert list(served.columns) == meta["features"] == list(offline.columns)
    pd.testing.assert_frame_equal(served, offline[meta["features"]], check_dtype=False)
    np.testing.assert_allclose(model.predict_proba(served)[:, 1], model.predict_proba(offline)[:, 1])