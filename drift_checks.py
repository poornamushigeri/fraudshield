"""Two quick drift checks. Features only for the monthly means; the ablations train on months 0-4
and score the validation month (5). The test months are never used for labels.
Usage: python drift_checks.py"""
from src.data import LABEL, clean, load_raw, split_by_month
from src.metrics import evaluate
from src.model import make_hgb, model_features
from src.tracking import log_run, setup

df = clean(load_raw("data/Base.csv"))

print("Monthly means of the features that drift most:")
print(df.groupby("month")[["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w"]].mean().round(1))

train, valid, _ = split_by_month(df)
setup()


def run(name, drop):
    Xtr = model_features(train).drop(columns=drop)
    Xva = model_features(valid).drop(columns=drop)
    s = make_hgb(Xtr).fit(Xtr, train[LABEL]).predict_proba(Xva)[:, 1]
    m = evaluate(valid[LABEL], s)
    print(name, {k: round(v, 4) for k, v in m.items() if k in ("pr_auc", "recall_at_fpr_5pct", "recall_at_top_5pct")})
    log_run(name, params={"model": "HGB", "dropped": ", ".join(drop) or "none", "train_months": "0-4",
                          "eval_split": "validation (month 5)", "purpose": "drift analysis"},
            metrics={f"valid_{k}": v for k, v in m.items()})


velocity = ["velocity_6h", "velocity_24h", "velocity_4w"]
windows = velocity + ["zip_count_4w", "date_of_birth_distinct_emails_4w", "bank_branch_count_8w"]
print("\nAblations (validation month):")
run("drift_ablation_none", [])
run("drift_ablation_no_velocity", velocity)
run("drift_ablation_no_windows", windows)