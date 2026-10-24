import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def _ranked_labels(y, score, seed=0):
    """Labels ordered from highest to lowest score; ties broken randomly."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))
    return y[order]


def recall_at_fpr(y, score, fpr, seed=0):
    """Share of frauds caught when at most `fpr` of honest applications are flagged."""
    ys = _ranked_labels(y, score, seed)
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
    i = np.searchsorted(fp, fpr * n_neg, side="right") - 1
    return float(tp[i] / n_pos) if i >= 0 else 0.0


def recall_at_budget(y, score, budget, seed=0):
    """Share of frauds caught when only the top `budget` share of applications is reviewed."""
    ys = _ranked_labels(y, score, seed)
    k = int(round(budget * len(ys)))
    return float(ys[:k].sum() / ys.sum())


def evaluate(y, score, seed=0):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "recall_at_fpr_5pct": recall_at_fpr(y, score, 0.05, seed),
        "recall_at_fpr_1pct": recall_at_fpr(y, score, 0.01, seed),
        "recall_at_top_1pct": recall_at_budget(y, score, 0.01, seed),
        "recall_at_top_5pct": recall_at_budget(y, score, 0.05, seed),
        "brier": float(brier_score_loss(y, score)),
    }