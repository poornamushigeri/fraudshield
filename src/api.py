"""FraudShield scoring API.   Run:  uvicorn src.api:app --reload"""
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text

from src.db import init_db, make_engine, register_model, save_prediction
from src.explain import Explainer
from src.schemas import Application, to_frame
from src.serve import MODEL_DIR, load_model, prepare

log = logging.getLogger("fraudshield")
import time

VERSION = "v1"
STORE_RETRY_SECONDS = 30   # after a failed save, skip saving for this long
NOTE = ("risk_score ranks applications and is not a calibrated probability of fraud. "
        "Reasons are returned only for medium and high bands; they show which inputs pushed "
        "this score up, not what caused it. "
        "Decision support only: a person makes the final call.")


def band(score: float, cutoffs: dict) -> str:
    if score >= cutoffs["high"]:
        return "high"
    if score >= cutoffs["medium"]:
        return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, meta = load_model(VERSION)
    cutoffs = json.loads((MODEL_DIR / f"bands_{VERSION}.json").read_text())["cutoffs"]
    engine = make_engine()
    init_db(engine)                       # fails loudly at startup if the database is unreachable
    register_model(engine, meta, cutoffs)
    app.state.model, app.state.meta, app.state.cutoffs = model, meta, cutoffs
    app.state.explainer = Explainer.load(model, VERSION)
    app.state.engine = engine
    app.state.store_down_until = 0.0
    yield
    engine.dispose()


app = FastAPI(
    title="FraudShield",
    description="Account-opening fraud risk scoring on benchmark data. Not for real decisions.",
    lifespan=lifespan,
)


class Reason(BaseModel):
    feature: str
    value: float | str | None
    effect: float


class ScoreResponse(BaseModel):
    model_version: str
    prediction_id: int | None
    risk_score: float
    risk_band: Literal["high", "medium", "low"]
    review_recommended: bool
    reasons: list[Reason]
    note: str


@app.get("/health")
def health():
    m = app.state.meta
    try:
        with app.state.engine.connect() as c:
            c.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "unavailable"
    return {"status": "ok" if database == "ok" else "degraded", "database": database,
            "model_version": m["version"], "trained_at_utc": m["trained_at_utc"],
            "train_months": m["train_months"]}


@app.post("/score", response_model=ScoreResponse)
def score(application: Application):
    X = prepare(to_frame([application]), app.state.meta["features"])
    s = round(float(app.state.model.predict_proba(X)[0, 1]), 5)
    b = band(s, app.state.cutoffs)
    # Reasons only matter when there is something to look at; skipping them for low risk
    # also skips the SHAP computation, which makes the common case faster.
    reasons = app.state.explainer.top_reasons(X)[0] if b != "low" else []

    # If storing fails, the analyst still gets the score, but the failure is logged and
    # prediction_id is null so a missing record is visible, not silent.
    # After a failure we skip saving for a short while: an outage then costs one slow
    # request, not every request.
    prediction_id = None
    if time.monotonic() >= app.state.store_down_until:
        try:
            prediction_id = save_prediction(
                app.state.engine, model_version=app.state.meta["version"], risk_score=s, risk_band=b,
                review_recommended=(b == "high"), application=application.model_dump(), reasons=reasons)
        except Exception:
            app.state.store_down_until = time.monotonic() + STORE_RETRY_SECONDS
            log.exception("could not store prediction; pausing storage for %s s", STORE_RETRY_SECONDS)
    else:
        log.warning("prediction not stored: database marked unavailable")

    return ScoreResponse(
        model_version=app.state.meta["version"], prediction_id=prediction_id, risk_score=s,
        risk_band=b, review_recommended=(b == "high"), reasons=reasons, note=NOTE)