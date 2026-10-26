import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db import (ModelVersion, Prediction, init_db, make_engine, record_outcome, register_model,
                    save_prediction)

META = {"version": "v1", "trained_at_utc": "2026-10-25T09:58:32+00:00", "train_months": "0-5", "git_commit": "dba9b1f"}
CUTOFFS = {"high": 0.0391, "medium": 0.0194}
APP = {"income": 0.6, "prev_address_months_count": None, "payment_type": "AD"}
REASONS = [{"feature": "prev_address_months_count_missing", "value": 1, "effect": 0.31}]


@pytest.fixture()
def engine(tmp_path):
    e = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    init_db(e)
    register_model(e, META, CUTOFFS)
    return e


def save(engine, **over):
    kw = dict(model_version="v1", risk_score=0.05, risk_band="high", review_recommended=True,
              application=APP, reasons=REASONS)
    kw.update(over)
    return save_prediction(engine, **kw)


def test_prediction_round_trip(engine):
    pid = save(engine)
    with Session(engine) as s:
        row = s.get(Prediction, pid)
        assert row.application == APP and row.reasons == REASONS
        assert row.outcome is None and row.created_at is not None


def test_registering_a_model_twice_does_not_duplicate_it(engine):
    register_model(engine, META, CUTOFFS)
    with Session(engine) as s:
        assert s.query(ModelVersion).count() == 1


def test_database_rejects_invalid_rows(engine):
    with pytest.raises(IntegrityError):
        save(engine, risk_band="urgent")
    with pytest.raises(IntegrityError):
        save(engine, model_version="v9")   # unregistered model version


def test_outcome_can_be_added_later_but_only_as_0_or_1(engine):
    pid = save(engine)
    record_outcome(engine, pid, 1)
    with Session(engine) as s:
        row = s.get(Prediction, pid)
        assert row.outcome == 1 and row.outcome_at is not None
    with pytest.raises(IntegrityError):
        record_outcome(engine, pid, 2)