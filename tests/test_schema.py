from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.data import MISSING_CODE_COLS, clean, load_raw
from src.model import model_features
from src.schemas import Application, to_frame
from src.serve import load_model, prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Base.csv"
MODEL = ROOT / "models" / "hgb_v1.joblib"
pytestmark = pytest.mark.skipif(not (DATA.exists() and MODEL.exists()), reason="data or model not available")

NOT_SENT = ["fraud_bool", "month", "customer_age", "days_since_request", "device_fraud_count"]


def to_record(row: dict) -> dict:
    """Dataset row -> API record: the dataset's unknown codes become null."""
    r = {k: v for k, v in row.items() if k not in NOT_SENT}
    for c in MISSING_CODE_COLS:
        if r[c] == -1:
            r[c] = None
    if r["intended_balcon_amount"] < 0:
        r["intended_balcon_amount"] = None
    return r


@pytest.fixture(scope="module")
def train_rows():
    raw = load_raw(DATA)
    return raw[raw["month"] <= 5].sample(500, random_state=2)  # never the test months


@pytest.fixture()
def record(train_rows):
    return to_record(train_rows.to_dict("records")[0])


def test_api_path_reproduces_offline_predictions(train_rows):
    model, meta = load_model("v1")
    offline = model.predict_proba(model_features(clean(train_rows)))[:, 1]
    apps = [Application(**to_record(r)) for r in train_rows.to_dict("records")]
    served = prepare(to_frame(apps), meta["features"])
    np.testing.assert_allclose(model.predict_proba(served)[:, 1], offline)


def test_rejects_fields_the_model_does_not_use(record):
    record["customer_age"] = 40
    with pytest.raises(ValidationError):
        Application(**record)


@pytest.mark.parametrize("field,value", [
    ("payment_type", "ZZ"), ("income", 5.0), ("email_is_free", 2),
    ("velocity_24h", -3.0), ("device_os", "beos"),
])
def test_rejects_invalid_values(record, field, value):
    record[field] = value
    with pytest.raises(ValidationError):
        Application(**record)


def test_unknown_must_be_stated_as_null(record):
    del record["prev_address_months_count"]
    with pytest.raises(ValidationError):
        Application(**record)