"""Train the final model on months 0-5 and save it.   Usage: python -m src.train"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.data import LABEL, SPLIT_COL, clean, load_raw
from src.model import HGB_PARAMS, make_hgb, model_features
from src.tracking import log_run, setup

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
TRAIN_MAX_MONTH = 5  # months 0-5; the test months (6-7) are never used for fitting


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(version: str = "v1") -> None:
    df = clean(load_raw(ROOT / "data" / "Base.csv"))
    train = df[df[SPLIT_COL] <= TRAIN_MAX_MONTH]
    assert train[SPLIT_COL].max() <= TRAIN_MAX_MONTH

    X, y = model_features(train), train[LABEL]
    model = make_hgb(X).fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"hgb_{version}.joblib")
    meta = {
        "version": version,
        "model": "HistGradientBoostingClassifier",
        "hyperparameters": HGB_PARAMS,
        "train_months": f"0-{TRAIN_MAX_MONTH}",
        "train_rows": int(len(train)),
        "train_fraud_rate": float(y.mean()),
        "features": list(X.columns),
        "excluded": ["customer_age", "days_since_request", "month", "fraud_bool"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    (MODEL_DIR / f"hgb_{version}.json").write_text(json.dumps(meta, indent=2))

    setup()
    log_run(
        f"final_hgb_{version}",
        params={"model": meta["model"], "train_months": meta["train_months"],
                "n_features": len(X.columns), "git_commit": meta["git_commit"], **HGB_PARAMS},
        metrics={"train_rows": float(len(train)), "train_fraud_rate": meta["train_fraud_rate"]},
    )
    print(f"saved models/hgb_{version}.joblib  |  rows: {len(train)}  |  features: {len(X.columns)}  |  fraud rate: {y.mean():.4%}")


if __name__ == "__main__":
    main()