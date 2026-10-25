"""Risk-band cutoffs from out-of-fold scores on months 0-5 (the test months are never used).
Usage: python -m src.bands"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import LABEL, SPLIT_COL, clean, load_raw
from src.metrics import recall_at_budget
from src.model import make_hgb, model_features

ROOT = Path(__file__).resolve().parents[1]


def main(version: str = "v1") -> None:
    df = clean(load_raw(ROOT / "data" / "Base.csv"))
    train = df[df[SPLIT_COL] <= 5]
    assert train[SPLIT_COL].max() <= 5
    X, y = model_features(train), train[LABEL].to_numpy()

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
        m = make_hgb(X.iloc[tr]).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]

    cutoffs = {"high": float(np.quantile(oof, 0.95)), "medium": float(np.quantile(oof, 0.90))}
    out = {
        "version": version,
        "method": "5-fold out-of-fold scores on months 0-5 (random folds)",
        "rows": int(len(y)),
        "cutoffs": cutoffs,
        "oof_recall_top5pct": recall_at_budget(y, oof, 0.05),
        "oof_recall_top10pct": recall_at_budget(y, oof, 0.10),
        "share_above_high": float((oof >= cutoffs["high"]).mean()),
        "share_above_medium": float((oof >= cutoffs["medium"]).mean()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ROOT / "models" / f"bands_{version}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()