from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data import LABEL, clean, load_raw, split_by_month
from src.model import make_hgb, model_features

DATA = Path(__file__).resolve().parents[1] / "data" / "Base.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def df():
    return clean(load_raw(DATA))


def test_forbidden_columns_are_not_features(df):
    cols = set(model_features(df).columns)
    assert not cols & {"fraud_bool", "month", "customer_age", "days_since_request", "device_fraud_count"}


def test_split_months_do_not_overlap(df):
    train, valid, test = split_by_month(df)
    assert train["month"].max() < valid["month"].min() <= valid["month"].max() < test["month"].min()
    assert len(train) + len(valid) + len(test) == len(df)


def test_missing_codes_are_gone(df):
    assert (df["prev_address_months_count"] == -1).sum() == 0
    assert (df["intended_balcon_amount"].dropna() < 0).sum() == 0


def test_saved_pipeline_matches_in_memory(df, tmp_path):
    sample = df.sample(60_000, random_state=0)
    X, y = model_features(sample), sample[LABEL]
    model = make_hgb(X).fit(X, y)
    joblib.dump(model, tmp_path / "m.joblib")
    reloaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], reloaded.predict_proba(X)[:, 1])