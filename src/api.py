"""FraudShield scoring API.   Run:  uvicorn src.api:app --reload"""
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from src.explain import Explainer
from src.schemas import Application, to_frame
from src.serve import MODEL_DIR, load_model, prepare

VERSION = "v1"
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
    app.state.model, app.state.meta = model, meta
    app.state.cutoffs = json.loads((MODEL_DIR / f"bands_{VERSION}.json").read_text())["cutoffs"]
    app.state.explainer = Explainer.load(model, VERSION)
    yield


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
    risk_score: float
    risk_band: Literal["high", "medium", "low"]
    review_recommended: bool
    reasons: list[Reason]
    note: str


@app.get("/health")
def health():
    m = app.state.meta
    return {"status": "ok", "model_version": m["version"],
            "trained_at_utc": m["trained_at_utc"], "train_months": m["train_months"]}


@app.post("/score", response_model=ScoreResponse)
def score(application: Application):
    X = prepare(to_frame([application]), app.state.meta["features"])
    s = float(app.state.model.predict_proba(X)[0, 1])
    b = band(s, app.state.cutoffs)
    # Reasons only matter when there is something to look at. Skipping them for low risk
    # also skips the SHAP computation, which makes the common case faster.
    reasons = app.state.explainer.top_reasons(X)[0] if b != "low" else []
    return ScoreResponse(
        model_version=app.state.meta["version"],
        risk_score=round(s, 5),
        risk_band=b,
        review_recommended=(b == "high"),
        reasons=reasons,
        note=NOTE,
    )