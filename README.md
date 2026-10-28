
![CI](https://github.com/poornamushigeri/fraudshield/actions/workflows/ci.yml/badge.svg)
# FraudShield

Cost-aware account-opening fraud risk scoring.

- Problem: predict whether an online bank account application is fraudulent
- Dataset: Bank Account Fraud (BAF) base variant, NeurIPS 2022 (non-commercial use; data is not stored in this repo)

## Folder layout

- `data/`: raw dataset files (ignored by Git)
- `notebooks/`: exploration and audit notebooks
- `src/`: reusable project code
- `reports/`: data audit, evaluation protocol, validation report

## Setup

```bash
python -m venv .venv
# Mac/Linux: source .venv/bin/activate
# Windows:   .venv\Scripts\activate
pip install -r requirements.txt
```

## Data and attribution
Dataset: Bank Account Fraud (BAF) suite, Base variant. Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro (2022), "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022 Datasets and Benchmarks. License: CC BY-NC-ND 4.0. The data is synthetic; the authors state it should not be used to build real-world fraud detection. This project is a methodology exercise on benchmark data, not a deployable fraud system.

## Results (validation month, month 5)

| Model | PR-AUC | Recall at 5% FPR | Recall in top 5% |
|---|---|---|---|
| Dummy | 0.012 | 0.045 | 0.045 |
| Red-flag rules (no age) | 0.024 | 0.152 | 0.150 |
| Logistic regression (no age) | 0.159 | 0.510 | 0.489 |
| Gradient-boosted trees (no age) | 0.184 | 0.530 | 0.506 |

95% bootstrap intervals and the model-selection reasoning are in `reports/model_selection.md`.
The test months (6-7) are evaluated once; results go in `reports/test_results.json`.


 ## How to run
    pip install -r requirements.txt
    python -m src.build_dataset        # clean data
    python -m src.train                # only for a NEW model version; v1 is frozen, retraining must save as v2
    python -m src.bands                # risk-band cutoffs
    python -m src.background           # background sample for reason codes
    docker compose up -d --build       # PostgreSQL + the API at http://127.0.0.1:8000/docs
    python -m pytest -q

The trained model files live in `models/` (not in Git) and are mounted into the API container.