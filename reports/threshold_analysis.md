# Threshold and cost analysis

Model: gradient-boosted trees, no age, no days_since_request. Validation month (month 5) only. All costs are hypothetical, in units where one manual review = 1.

## Setup
- Policy: review the top X% of applications by score (a review budget, not a score cutoff).
- Unknown parameter: the cost of a missed fraud as a multiple R of the cost of a review. Tested R = 10, 50, 200.

## Results
| R | Cost-minimising review share | Fraud caught | Precision | Cost reduction vs no review |
|---|---|---|---|---|
| 10 | 1.8% | 32% | 21% | 17% |
| 50 | 14.6% | 75% | 6% | 50% |
| 200 | 29.2% (edge of searched range) | 89% | 3.6% | 77% |

At a fixed capacity of 5% of applications: 51% of fraud caught, precision 12%; cost reduction 8% (R=10), 42% (R=50), 49% (R=200).

## Findings
- The cost-minimising review share depends strongly on R, so no single threshold is "optimal" without a justified cost assumption.
- At R=10 reviewing more than about 7% of applications costs more than reviewing nobody.
- The cost curve is flat around its minimum for R=50 and R=200, so the exact budget is not critical.

## Calibration
Scores under-predict fraud in the top deciles (top decile: 5.7% predicted vs 7.8% observed) and overall (0.9% vs 1.18%), consistent with the rising fraud rate. Scores are used for ranking only and are not presented as probabilities.

## Decision
Operating point: review the top 5% of applications by score. Sensitivity reported at 2% and 10%.
Limitations: costs are assumptions, one validation month, synthetic data, no claim of real-world savings.