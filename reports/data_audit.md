# Data Audit: BAF Base Variant

Audit of the Bank Account Fraud (BAF) base dataset, done before any modelling. Source notebook: `notebooks/01_first_look.ipynb`.

## 1. Overview
- 1,000,000 account-opening applications, 32 columns, months 0-7.
- Label: `fraud_bool` (1 = fraud). Fraud count: 11,029 (1.10%).
- No NaN cells in the raw file (but see section 3: missing values are hidden as codes).
- Column types: 9 float, 18 integer, 5 text/categorical.
- Source and license: Kaggle, BAF dataset suite (NeurIPS 2022). Non-commercial license, so the data is not stored in this repository. Synthetic data derived from a real bank's data.

## 2. Class imbalance
Fraud is rare (1.10%). A model that predicts "not fraud" for every application would be about 98.9% accurate and useless, so accuracy is not used as a metric. See `evaluation_protocol.md`.

## 3. Missing values hidden as codes
-1 appears to be a "value unknown" code in five columns:

| Column | Rows with -1 | Share of rows |
|---|---|---|
| prev_address_months_count | 712,920 | 71.3% |
| bank_months_count | 253,635 | 25.4% |
| current_address_months_count | 4,254 | 0.4% |
| session_length_in_minutes | 2,015 | 0.2% |
| device_distinct_emails_8w | 359 | 0.04% |

Evidence for treating -1 as a code: in each column it is a single repeated value, not a spread of numbers.

The fact that a value is missing is itself informative:

| Column | Fraud rate when -1 | Fraud rate otherwise |
|---|---|---|
| prev_address_months_count | 1.42% | 0.31% |
| bank_months_count | 1.63% | 0.92% |

Decision: convert -1 to NaN and add a "was missing" flag for each of the five columns.

Verification status: TODO - confirm in the BAF datasheet/paper that -1 means "unknown", and note the wording here.

Association is not cause: missing information may be linked to other applicant traits.

## 4. Negative values that are not missing codes

- `intended_balcon_amount`: 742,523 negative values, almost all distinct (range about -15.5 to near 0). Continuous values, left unchanged. Fraud rate is 1.31% when negative vs 0.50% otherwise.
- `credit_risk_score`: 14,445 negative values across 164 distinct values (-170 to -1). Treated as real scores, left unchanged.
- `velocity_6h`: 44 negative values, all different (minimum about -171). The velocity columns have medians in the thousands, so they are evidently transformed or scaled rather than raw counts, and a negative value is not necessarily invalid. Left unchanged. TODO: confirm in the datasheet what the velocity columns represent.


## 5. Constant and special columns
- `device_fraud_count` has a single unique value in this dataset, so it carries no information and is dropped.
- `month` is the time index. It is used only for splitting, never as a feature.
- Six columns are binary flags (2 unique values): keep_alive_session, foreign_request, phone_home_valid, has_other_cards, email_is_free, phone_mobile_valid.
- `source`: INTERNET 992,952 vs TELEAPP 7,048 (0.7%), so the second channel is very small.
- `customer_age` is rounded to the decade (10 to 90), and the datasheet notes that age and income were categorised and noise added before the data was generated. The "10" band (2.1% of rows) is not explained in the datasheet and is probably an artefact of that process, so it is not interpreted as minors. Fraud rate rises steadily with age, from 0.35% (band 10) to 5.26% (band 90).
- Age groups used for fairness analysis (the same threshold the dataset authors use):

| Group | Rows | Frauds | Fraud rate |
|---|---|---|---|
| Under 50 | 816,987 | 6,744 | 0.83% |
| 50 and over | 183,013 | 4,285 | 2.34% |

  The 50+ group is 18.3% of applications but 38.9% of frauds, with a fraud rate about 2.8 times higher. The datasheet describes the gap as roughly twice, noting that values vary across the suite. Bands 80 and 90 are too small to analyse separately.

- `customer_age` is grouped into decade bands (10 to 90). Fraud rate rises steadily with age, from 0.35% (band 10) to 2.00% (band 50) to 5.26% (band 90). The 50-and-over group has 183,013 applications. Bands 80 and 90 are too small (1,297 and 76 rows) for separate analysis, so fairness checks will use 50+ versus under 50. TODO: confirm what the "10" band means.
- `proposed_credit_limit` takes a small set of values: mostly 200, 500, 1,000 and 1,500, with small clusters beside them (190, 210, 490, 510, 990, 1,900, 2,000, 2,100). TODO: confirm in the datasheet why.

## 6. Categorical columns
| Column | Values and counts |
|---|---|
| payment_type | AA 258,249; AB 370,554; AC 252,071; AD 118,837; AE 289 |
| employment_status | CA 730,252; CB 138,288; CC 37,758; CD 26,522; CE 22,693; CF 44,034; CG 453 |
| housing_status | BA 169,675; BB 260,965; BC 372,143; BD 26,161; BE 169,135; BF 1,669; BG 252 |
| device_os | other 342,728; linux 332,712; windows 263,506; macintosh 53,826; x11 7,228 |

The category codes are anonymised, so no business meaning can be given to them. This limits explanations and is listed as a limitation. Several categories are very small (for example AE, CG, BG), so estimates for them will be noisy.

## 7. Behaviour over time
| Month | Applications | Frauds | Fraud rate |
|---|---|---|---|
| 0 | 132,440 | 1,500 | 1.13% |
| 1 | 127,620 | 1,198 | 0.94% |
| 2 | 136,979 | 1,198 | 0.87% |
| 3 | 150,936 | 1,392 | 0.92% |
| 4 | 127,691 | 1,452 | 1.14% |
| 5 | 119,323 | 1,411 | 1.18% |
| 6 | 108,168 | 1,450 | 1.34% |
| 7 | 96,843 | 1,428 | 1.47% |

Fraud counts stay near 1,200-1,500 per month, but application volume falls in later months, so the fraud rate rises from 0.87% (month 2) to 1.47% (month 7). This is the first evidence of a shift over time and is the reason for the time-based split. Later months are therefore harder to compare with earlier ones.

## 8. Leakage review
For each column: would the bank know this at the moment the application arrives?

| Column | Known at application time? | Notes |
|---|---|---|
| days_since_request | Unconfirmed in datasheet | Excluded from the main model. Sensitivity run: adding it left validation PR-AUC at 0.186 vs 0.184 and recall at 5% FPR at 0.528 vs 0.530, so no sign of leakage and no benefit. Stays excluded. |
| velocity_6h / velocity_24h / velocity_4w | TODO | Rolling counts; check how they are computed |
| credit_risk_score | TODO | |
| proposed_credit_limit | TODO | |
| intended_balcon_amount | TODO | |
| device_fraud_count | n/a | Constant, dropped |

Finding: TODO after reading the datasheet. Any column judged to be available only after the decision is removed and the reason is written here.

## 9. Decisions summary
- Convert -1 to NaN plus flag in five columns.
- Drop `device_fraud_count`.
- Exclude `fraud_bool` and `month` from features.
- Leave `intended_balcon_amount`, `credit_risk_score`, `velocity_6h` unchanged.
- Split by month as defined in `evaluation_protocol.md`.

## 10. Limitations
- Synthetic data: results do not prove real-world performance.
- Anonymised categories limit interpretation.
- Missing-value codes inferred from patterns, pending datasheet confirmation.