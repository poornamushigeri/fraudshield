"""Evaluate on the test months (6-7) ONCE.
Usage:  python -m src.evaluate_test --dry-run   (code check on month 5, writes nothing)
        python -m src.evaluate_test             (the real, single evaluation)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from src.data import LABEL, SPLIT_COL, clean, load_raw
from src.metrics import evaluate, recall_at_fpr
from src.model import model_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "test_results.json"
BUDGET = 0.05
B = 300


def main(version: str = "v1", dry_run: bool = False) -> None:
    if RESULTS.exists() and not dry_run:
        sys.exit(f"{RESULTS.name} already exists: the test set has already been evaluated. "
                 "Re-running breaks the protocol. Add an amendment explaining why, then delete the file on purpose.")

    df = clean(load_raw(ROOT / "data" / "Base.csv"))
    test = df[df[SPLIT_COL] == 5] if dry_run else df[df[SPLIT_COL] >= 6]
    model = joblib.load(ROOT / "models" / f"hgb_{version}.joblib")
    score = model.predict_proba(model_features(test))[:, 1]
    y = test[LABEL].to_numpy()

    out = {
        "model_version": version,
        "evaluated_months": "5 (DRY RUN, in-sample, ignore the numbers)" if dry_run else "6-7",
        "rows": int(len(y)),
        "fraud_rate": float(y.mean()),
        "mean_predicted": float(score.mean()),
        "metrics": evaluate(y, score),
        "by_month": {},
    }
    months = test[SPLIT_COL].to_numpy()
    for m in sorted(set(months)):
        mask = months == m
        out["by_month"][int(m)] = {"fraud_rate": float(y[mask].mean()), **evaluate(y[mask], score[mask])}

    rng = np.random.default_rng(42)
    pr, rc = [], []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        pr.append(average_precision_score(y[idx], score[idx]))
        rc.append(recall_at_fpr(y[idx], score[idx], 0.05))
    out["bootstrap_95"] = {
        "pr_auc": np.percentile(pr, [2.5, 97.5]).round(4).tolist(),
        "recall_at_fpr_5pct": np.percentile(rc, [2.5, 97.5]).round(4).tolist(),
    }

    older = (test["customer_age"] >= 50).to_numpy()
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-score)[: int(round(BUDGET * len(y)))]] = True
    groups = {}
    for name, mask in [("under_50", ~older), ("50_plus", older)]:
        neg, pos = mask & (y == 0), mask & (y == 1)
        groups[name] = {
            "rows": int(mask.sum()),
            "fraud_rate": float(y[mask].mean()),
            "flagged_share": float(flagged[mask].mean()),
            "false_positive_rate": float(flagged[neg].sum() / neg.sum()),
            "recall": float(flagged[pos].sum() / pos.sum()),
            "precision": float(y[flagged & mask].mean()),
        }
    out["fairness_at_5pct_budget"] = groups

    print(json.dumps(out, indent=2))
    if dry_run:
        print("\nDRY RUN: nothing was written. These numbers are in-sample and must not be used.")
        return
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)