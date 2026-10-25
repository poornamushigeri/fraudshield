"""Reason codes: which inputs pushed a score up the most (SHAP values, model-agnostic).

Why not TreeExplainer: it mishandled this model's native categorical splits; the additivity test
(contributions + baseline = model score) failed. This explainer only calls the model's prediction
function, so the values add up exactly.

Values are in log-odds, relative to a background sample of 100 applications from months 0-5.
They describe how the model reached its score; they do not show that a feature causes fraud.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
N_PERMUTATIONS = 10


class Explainer:
    def __init__(self, model, background: np.ndarray):
        self.pre = model.named_steps["pre"]
        self.clf = model.named_steps["clf"]
        self.background = background
        self.names = [n.split("__", 1)[-1] for n in self.pre.get_feature_names_out()]

    @classmethod
    def load(cls, model, version: str = "v1"):
        return cls(model, np.load(ROOT / "models" / f"background_{version}.npy"))

    def contributions(self, X: pd.DataFrame):
        """Return (SHAP values per feature, baseline per row, the model's raw score per row)."""
        Z = self.pre.transform(X)
        masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
        ex = shap.explainers.Permutation(self.clf.decision_function, masker,
                                         feature_names=self.names, seed=0)  # fixed seed: same input, same reasons
        res = ex(Z, max_evals=2 * N_PERMUTATIONS * len(self.names) + 1)
        raw = self.clf.decision_function(Z)
        return pd.DataFrame(res.values, columns=self.names, index=X.index), np.asarray(res.base_values), raw

    def top_reasons(self, X: pd.DataFrame, k: int = 3):
        contrib, _, _ = self.contributions(X)
        out = []
        for i in range(len(X)):
            row = contrib.iloc[i]
            top = row[row > 0].sort_values(ascending=False).head(k)
            reasons = []
            for feat, effect in top.items():
                v = X.iloc[i][feat]
                if isinstance(v, (float, np.floating)) and np.isnan(v):
                    v = None
                elif isinstance(v, np.generic):
                    v = v.item()
                reasons.append({"feature": feat, "value": v, "effect": round(float(effect), 4)})
            out.append(reasons)
        return out