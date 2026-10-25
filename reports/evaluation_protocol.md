# Evaluation Protocol

Written before any model was trained. Changes after this point must be recorded in the "Amendments" section at the bottom, with a reason.

## 1. Task
Score online account-opening applications so analysts can review the riskiest ones first. Label: `fraud_bool` (1 = fraud).

## 2. Data and split
- Dataset: BAF Base variant (1,000,000 rows, months 0-7). Synthetic, derived from real bank data; non-commercial license; not stored in the repo.
- Time-based split:
  - Train: months 0-4 (675,666 rows, fraud rate 0.9975%)
  - Validation: month 5 (119,323 rows, 1.1825%)
  - Test: months 6-7 (205,011 rows, 1.4038%)
- The test set is evaluated once, at the end, after the final model and threshold are fixed. If it is used for any tuning decision, that must be recorded as an amendment.
- Not used as features: `fraud_bool` (label), `month` (split variable).

## 3. Preprocessing rules
- Any negative value in `intended_balcon_amount`, or -1 in `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, and `device_distinct_emails_8w`, is treated as "unknown": converted to NaN, with a was-missing flag added.
- `days_since_request` is excluded from the main model because its measurement time is unclear, and is tested as a sensitivity run.
- `device_fraud_count` is dropped (constant).
- All fitting (imputation, encoding, scaling) happens on the training set only, inside a pipeline.


## 4. Models to compare
1. Dummy classifier (predicts the base rate)
2. Simple hand-written rules baseline
3. Logistic regression
4. One gradient-boosted tree model

The final model is chosen on validation results, with the reason written down.

## 5. Metrics
Primary:
- PR-AUC (average precision)
- Recall at a fixed false-positive rate (5%, also reported at 1%)
- Recall at a fixed review budget (top 1% and top 5% of applications by score)

Secondary:
- Calibration: Brier score and reliability plot
- Bootstrap 95% confidence intervals on the primary metrics

Not used as headline metrics: accuracy and ROC-AUC (misleading when fraud is about 1% of rows).

## 6. Threshold selection
- Chosen on the validation month only, expressed as a review budget (share of applications flagged), because the fraud rate shifts across months.
- Cost analysis uses hypothetical parameters: loss per missed fraud, cost per manual review, and daily review capacity. Results are shown across a range of loss-to-review-cost ratios (for example 10:1, 50:1, 200:1) as a sensitivity analysis. No real-world savings are claimed.

## 7. Drift and fairness checks
- Drift: per-feature Population Stability Index (PSI) per month, and metrics on months 6-7 compared with validation. Input drift is reported separately from performance change.
- Fairness: recall and false-positive rate compared across customer_age bands. Whether age is used as a model feature is decided and justified in the validation report.
- Feature ablation: the final model is also trained without (a) `customer_age`, and (b) `days_since_request`, and (c) `credit_risk_score`, one at a time. Results are compared on validation. A large drop means the model leans heavily on that column; the dependence is reported and the column's availability at application time is re-checked.

## 8. Known limitations
- Synthetic data; results do not prove real-world performance.
- Categorical columns are anonymised codes, which limits interpretation.
- Only one validation month is available for threshold choice.

## Amendments
- Datasheet review (before any model was trained): intended_balcon_amount negatives are missing values per the BAF datasheet, so it is handled like the other missing-code columns. days_since_request is excluded from the main model because its measurement time is unclear, and is tested as a sensitivity run.
- Before any test-set evaluation: the final model (gradient-boosted trees, no age, no days_since_request) will be retrained on months 0-5 with the hyperparameters chosen on validation, then evaluated once on months 6-7. The operating point stays a review budget of 5%.
- Before any test-set evaluation: the final model (gradient-boosted trees, no customer_age, no days_since_request) is retrained on months 0-5 with the hyperparameters chosen on validation, then evaluated once on months 6-7. The operating point is a review budget of 5% of applications. The test results file is written once; re-running requires a new amendment explaining why.
- Test-set evaluation: executed once; the results file is `reports/test_results.json`, committed in <commit hash from Step 2>. It was produced by the final model v1 (trained on months 0-5). No model, feature, hyperparameter or threshold decision was changed after the results were produced. The evaluation script refuses to run again if the file exists.