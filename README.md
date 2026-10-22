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
