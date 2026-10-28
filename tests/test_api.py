from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api import app, band
from src.data import clean, load_raw
from src.db import Prediction, make_engine
from src.model import model_features
from src.serve import load_model
from tests.test_schema import to_record

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Base.csv"
MODEL = ROOT / "models" / "hgb_v1.joblib"
BANDS = ROOT / "models" / "bands_v1.json"
pytestmark = pytest.mark.skipif(not (DATA.exists() and MODEL.exists() and BANDS.exists()),
                                reason="data, model or bands not available")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample():
    raw = load_raw(DATA)
    return raw[raw["month"] <= 5].sample(20, random_state=3)  # never the test months


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1" and body["database"] == "ok"


def test_score_matches_offline_prediction(client, sample):
    model, _ = load_model("v1")
    offline = model.predict_proba(model_features(clean(sample)))[:, 1]
    for rec, expected in zip(sample.to_dict("records"), offline):
        r = client.post("/score", json=to_record(rec))
        assert r.status_code == 200
        body = r.json()
        assert body["risk_score"] == pytest.approx(expected, abs=1e-5)
        assert body["model_version"] == "v1"
        assert body["review_recommended"] == (body["risk_band"] == "high")
        assert len(body["reasons"]) <= 3


def test_rejects_bad_input(client, sample):
    rec = to_record(sample.to_dict("records")[0])
    assert client.post("/score", json={**rec, "payment_type": "ZZ"}).status_code == 422
    assert client.post("/score", json={**rec, "customer_age": 40}).status_code == 422


def test_band_cutoffs():
    c = {"high": 0.04, "medium": 0.02}
    assert band(0.05, c) == "high"
    assert band(0.04, c) == "high"
    assert band(0.03, c) == "medium"
    assert band(0.01, c) == "low"


def test_reasons_only_for_medium_and_high(client):
    raw = load_raw(DATA)
    train_months = raw[raw["month"] <= 5]  # never the test months
    sample = (train_months[train_months["fraud_bool"] == 1].sample(40, random_state=7).to_dict("records")
              + train_months.sample(40, random_state=8).to_dict("records"))
    bands = set()
    for rec in sample:
        body = client.post("/score", json=to_record(rec)).json()
        bands.add(body["risk_band"])
        if body["risk_band"] == "low":
            assert body["reasons"] == []
        else:
            assert len(body["reasons"]) <= 3
    assert "low" in bands and bands != {"low"}  # the test saw both kinds of case


def test_every_scored_application_is_stored(client, sample):
    rec = to_record(sample.to_dict("records")[0])
    body = client.post("/score", json=rec).json()
    assert isinstance(body["prediction_id"], int)
    with Session(make_engine()) as s:     # same throwaway test database the API used
        row = s.get(Prediction, body["prediction_id"])
        assert row.model_version == "v1"
        assert row.risk_score == pytest.approx(body["risk_score"])
        assert row.risk_band == body["risk_band"]
        assert row.application == rec          # exactly what was sent, nulls included
        assert row.outcome is None


def test_rejected_requests_are_not_stored(client, sample):
    with Session(make_engine()) as s:
        before = s.query(Prediction).count()
    rec = to_record(sample.to_dict("records")[0])
    assert client.post("/score", json={**rec, "payment_type": "ZZ"}).status_code == 422
    with Session(make_engine()) as s:
        assert s.query(Prediction).count() == before

def test_an_outage_costs_one_slow_request_not_every_request(client, sample, monkeypatch):
    calls = []

    def failing_save(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("database down")

    monkeypatch.setattr("src.api.save_prediction", failing_save)
    monkeypatch.setattr(app.state, "store_down_until", 0.0)   # restored to its old value after the test
    rec = to_record(sample.to_dict("records")[0])

    first = client.post("/score", json=rec)
    second = client.post("/score", json=rec)
    for r in (first, second):
        assert r.status_code == 200                      # the analyst still gets a score
        assert r.json()["prediction_id"] is None         # and the missing record is visible
    assert len(calls) == 1                               # the second request did not try to save

    monkeypatch.setattr(app.state, "store_down_until", 0.0)   # the pause is over
    client.post("/score", json=rec)
    assert len(calls) == 2                               # saving is tried again