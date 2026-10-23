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