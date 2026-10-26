"""Storage for every prediction the API makes.

PostgreSQL in the running system. The same models also work on SQLite, which the tests use so they
run without Docker. Predictions are append-only: the only column updated later is the outcome,
filled in when the real label becomes known (needed for accuracy monitoring).
"""
import os
from datetime import datetime

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index,
                        Integer, SmallInteger, String, create_engine, event, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# Local development credentials only. Anything shared or deployed needs a real secret in DATABASE_URL.
DEFAULT_URL = "postgresql+psycopg://fraudshield:fraudshield_dev@localhost:5433/fraudshield"

JsonType = JSON().with_variant(JSONB(), "postgresql")
IdType = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    trained_at_utc: Mapped[str] = mapped_column(String(40))
    train_months: Mapped[str] = mapped_column(String(16))
    git_commit: Mapped[str] = mapped_column(String(40))
    cutoff_high: Mapped[float] = mapped_column(Float)
    cutoff_medium: Mapped[float] = mapped_column(Float)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("risk_band IN ('high', 'medium', 'low')", name="ck_predictions_band"),
        CheckConstraint("outcome IN (0, 1)", name="ck_predictions_outcome"),  # NULL (unknown) is allowed
        Index("ix_predictions_version_band", "model_version", "risk_band"),
    )

    id: Mapped[int] = mapped_column(IdType, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(8))
    review_recommended: Mapped[bool] = mapped_column(Boolean)
    application: Mapped[dict] = mapped_column(JsonType)   # the request as received (synthetic data here)
    reasons: Mapped[list] = mapped_column(JsonType)
    outcome: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)  # 1 = fraud, 0 = not fraud, NULL = unknown
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def make_engine(url: str | None = None):
    engine = create_engine(url or os.environ.get("DATABASE_URL", DEFAULT_URL), pool_pre_ping=True)
    if engine.dialect.name == "sqlite":  # SQLite ignores foreign keys unless told otherwise
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def register_model(engine, meta: dict, cutoffs: dict) -> None:
    """Record the model version once, with the cutoffs the API uses for it."""
    with Session(engine) as s:
        if s.get(ModelVersion, meta["version"]) is None:
            s.add(ModelVersion(
                version=meta["version"], trained_at_utc=meta["trained_at_utc"],
                train_months=meta["train_months"], git_commit=meta["git_commit"],
                cutoff_high=cutoffs["high"], cutoff_medium=cutoffs["medium"],
            ))
            s.commit()


def save_prediction(engine, *, model_version: str, risk_score: float, risk_band: str,
                    review_recommended: bool, application: dict, reasons: list) -> int:
    with Session(engine) as s:
        row = Prediction(model_version=model_version, risk_score=risk_score, risk_band=risk_band,
                         review_recommended=review_recommended, application=application, reasons=reasons)
        s.add(row)
        s.commit()
        return row.id


def record_outcome(engine, prediction_id: int, outcome: int) -> None:
    with Session(engine) as s:
        row = s.get(Prediction, prediction_id)
        if row is None:
            raise KeyError(prediction_id)
        row.outcome = outcome
        row.outcome_at = func.now()
        s.commit()