1.first test run happened accidentally; the guard stopped a second run, and I disclosed it
2.Return reasons only when there’s something to investigate; showing weak reasons for low-risk cases invites false suspicion, and skipping them makes the common case faster too.
3.The features with the biggest drift were ones the model barely used, which is why performance held. I checked by ablation instead of assuming it.

“The container reproduces the offline model’s scores to the API’s rounding precision, with the same pinned library versions.”
“I tested the failure mode on purpose: with the database down, the analyst still gets a score, the missing record is visible, and it recovers on its own.”
“CI runs on every push. Data-dependent tests skip on the CI machine because the dataset is licensed and can’t be in the repo, and I know exactly which ones skip.”