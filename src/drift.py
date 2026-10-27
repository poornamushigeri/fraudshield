"""Input and score drift by month. Uses features and model scores only, never labels.
Usage: python -m src.drift"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LABEL, SPLIT_COL, clean, load_raw
from src.model import model_features
from src.serve import load_model

ROOT = Path(__file__).resolve().parents[1]
REF_MAX_MONTH = 5     # reference = the training months (0-5)
EPS = 1e-4            # smoothing so empty bins do not break the log
MAX_DISCRETE = 12     # columns with this many distinct values or fewer are treated as categories


def psi_from_dists(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _labels(s: pd.Series) -> pd.Series:
    return (s.round(6) if pd.api.types.is_numeric_dtype(s) else s).astype(str)  # missing becomes its own "nan" label


def dists_categorical(ref: pd.Series, cur: pd.Series):
    r, c = _labels(ref), _labels(cur)
    cats = pd.Index(pd.concat([r, c]).unique())   # includes categories that only appear in the new month
    return (r.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(),
            c.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy())


def dists_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10):
    """Bins are deciles of the reference; missing values are their own bin."""
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]

    def dist(s: pd.Series):
        v = s.dropna().to_numpy()
        counts = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(edges) - 1).astype(float)
        counts = np.append(counts, float(s.isna().sum()))
        return counts / counts.sum()

    return dist(ref), dist(cur)


def feature_psi(ref: pd.Series, cur: pd.Series) -> float:
    discrete = (not pd.api.types.is_numeric_dtype(ref)) or ref.nunique() <= MAX_DISCRETE
    e, a = dists_categorical(ref, cur) if discrete else dists_numeric(ref, cur)
    return psi_from_dists(e, a)


def main(version: str = "v1") -> None:
    df = clean(load_raw(ROOT / "data" / "Base.csv"))
    X = model_features(df)
    assert LABEL not in X.columns                 # labels are never used here
    months = sorted(df[SPLIT_COL].unique())
    in_ref = (df[SPLIT_COL] <= REF_MAX_MONTH).to_numpy()
    by_month = {m: (df[SPLIT_COL] == m).to_numpy() for m in months}

    table = {}
    for col in X.columns:
        ref = X.loc[in_ref, col]
        table[col] = {f"month_{m}": feature_psi(ref, X.loc[by_month[m], col]) for m in months}
    psi = pd.DataFrame(table).T
    psi["max_6_7"] = psi[["month_6", "month_7"]].max(axis=1)
    psi = psi.sort_values("max_6_7", ascending=False)
    psi.round(4).to_csv(ROOT / "reports" / "drift_psi.csv")

    cols = ["month_5", "month_6", "month_7"]
    print("Feature PSI against months 0-5 (largest drift in months 6-7 first):")
    print(psi[cols].head(12).round(3).to_string())
    print()
    print("features with PSI above 0.10 in month 6:", int((psi["month_6"] > 0.10).sum()),
          "| month 7:", int((psi["month_7"] > 0.10).sum()),
          "| above 0.25 in month 7:", int((psi["month_7"] > 0.25).sum()),
          "| of", len(psi), "features")

    # Model scores on the unseen months: the share of applications above the cutoffs should be about 5% and 10%
    model, _ = load_model(version)
    cut = json.loads((ROOT / "models" / f"bands_{version}.json").read_text())["cutoffs"]
    print("\nShare of applications above the risk-band cutoffs (expected about 5% and 10%):")
    for m in (6, 7):
        s = model.predict_proba(X[by_month[m]])[:, 1]
        print(f"month {m}: above high {np.mean(s >= cut['high']):.2%} | above medium {np.mean(s >= cut['medium']):.2%} "
              f"| mean score {s.mean():.4f}")
    print("\nwrote reports/drift_psi.csv")


if __name__ == "__main__":
    main()