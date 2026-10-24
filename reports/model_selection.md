# Model selection

Chosen on the validation month (month 5) only. The test months (6-7) have not been used.

## Candidates (validation, 95% bootstrap intervals, 300 resamples)
| Model | PR-AUC | Recall at 5% FPR |
|---|---|---|
| Dummy (prior) | 0.012 | 0.045 |
| Red-flag rules (no age) | 0.024 (0.022-0.026) | 0.158 (0.139-0.177) |
| Logistic regression (no age) | 0.161 (0.140-0.184) | 0.510 (0.485-0.537) |
| Gradient-boosted trees (no age) | 0.186 (0.163-0.209) | 0.532 (0.506-0.557) |

Paired difference, boosted trees minus logistic regression: PR-AUC +0.025 (0.016 to 0.037); recall at 5% FPR +0.022 (0.005 to 0.039).

## Decision
Gradient-boosted trees, without customer_age and without days_since_request.

## Reasons
- Reliably better than logistic regression on validation, though the gain is modest.
- Removing credit_risk_score changed PR-AUC by about 0.003, so the model does not simply reproduce the bank's own score.
- Adding days_since_request gave no meaningful gain (PR-AUC 0.186 vs 0.184), so there is no sign of leakage; it stays excluded because its measurement time is unclear.
- Age gave no accuracy gain in either model. Excluding it still leaves a disparity (honest applicants aged 50+ were flagged about 2.3 times as often in the logistic regression at a 5% review budget), which is reported rather than hidden.

## Limits
- Intervals cover sampling variation in one validation month only, not fitting randomness or month-to-month shift.
- Synthetic benchmark data; no claim of real-world performance.