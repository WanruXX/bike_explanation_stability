# BE0 Review: Correlated-Feature Explanation Stability

Review date: 2026-10-07. This note interprets the existing completed run; it does
not replace its report, regenerate explanations, or change its settings.

## Evidence And Protocol

Sources: `manifest.json`, `report.json`, `summary.txt`, and the per-model JSON
records in this directory. There are 24 selected models from 48 candidate fits:
four families, three seeds, and two feature representations. TRAIN contains
8,645 rows from 2011, validation 4,358 rows from the first half of 2012, and TEST
4,376 rows from the second half of 2012. Preprocessing and models fit TRAIN only;
candidate selection uses validation RMSE without refitting on validation.

Individual/group permutation importance uses 512 held-out rows and 10 repeats.
SHAP was enabled: 32 explained rows, 64 TRAIN background rows, three permutation
cycles. Dataset SHA256:
`e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

## Main Finding

The run supports investigating how correlated features receive different credit
across similarly performing models. It does not demonstrate that grouping is
generally more reliable, faithful, or causal.

TRAIN correlation between temperature and apparent temperature is 0.9920.
For the full-feature, seed-42 selected models:

| Model | Validation RMSE | TEST RMSE | temp permutation MSE increase | atemp permutation MSE increase | Joint temperature MSE increase |
|---|---:|---:|---:|---:|---:|
| Random forest | 122.558 | 141.110 | 1,989.624 | 3,752.047 | 9,962.566 |
| Histogram gradient boosting | 120.663 | 130.194 | 3,593.606 | 1,094.177 | 7,424.815 |

The preferred temperature feature reverses despite close validation performance.
This localized attribution ambiguity is more informative than claiming that
explanations are broadly unstable.

Removing atemp changes seed-42 validation RMSE from 122.558 to 122.789 for forest
and from 120.663 to 120.981 for boosting. Shared information is consistent with
these observations, but correlation and ablation do not identify causal effects.

## Agreement And Performance Limits

- Only forest and boosting satisfy the within-5%-of-best validation criterion.
  Ridge and EBM have validation RMSE around 146-147 versus 121-123; their
  explanation comparisons should remain secondary, not near-optimal evidence.
- The six near-optimal cross-family comparisons have individual rank
  correlations approximately 0.92-0.96 and group rank correlation 1. All four
  group rankings agree, but twelve-feature and four-group correlations are not
  directly calibrated against one another.
- Group magnitude is not invariant: joint temperature reliance is about 9,963
  versus 7,425 in the seed-42 full-feature comparison.
- Forest/boosting predictions disagree by about 30-31 rentals/hour RMSE on
  TEST. Similar validation error is not equal predictions or equal TEST error.
- Removing atemp changes near-optimal model predictions by approximately
  12.9-15.9 rentals/hour RMSE, despite small validation-score differences.
- Boosting's three seeds give identical TEST predictions in each representation;
  ridge also repeats predictions. These are not independent model replications.
- Mean demand rises from 143.794 in TRAIN to 215.162 in validation and 254.091
  in TEST. The stronger full-feature models underpredict TEST by about 86-88
  rentals/hour. Temporal shift is an important competing influence, not yet
  isolated or corrected by this study.
- Marginal permutation can create unrealistic feature combinations. Joint
  permutation preserves dependence within a group, not between groups. Repeat
  SD measures Monte Carlo variation, not temporal sampling uncertainty.

## SHAP Sampling Defect

The notebook's training cell delegates to `run_bike_experiment`, which calls
`explain_bike_model` in `bike_explanations.py`. `explanation_indices` returns a
random 512-row cohort sorted chronologically. BE0 then used
`indices[:cfg.shap_rows]` for SHAP: the earliest 32 rows of that sorted cohort,
not a random 32-row subset covering the six-month TEST period. Saved SHAP row IDs
run from 13024 through 13210 in the representative seed-42 records.

This is a backend selection defect, not a wrong notebook configuration value.
The values still explain the selected early observations, but their averages
must not represent the full TEST period. Reconstruction errors around 1e-12
verify additivity, not faithfulness, representativeness, or adequate Monte Carlo
convergence. Forest's mean absolute temp/atemp SHAP values are 17.336/41.933;
boosting's are 12.706/12.843 on this narrow cohort. These are descriptive only.

The corrected code randomly selects the SHAP subset before sorting it, keeps
the same cohort across representations/models, and records timestamps and its
sampling policy. It does not retroactively correct BE0 artifacts.

## Next Experiment: BE1

Question: How representation-dependent is attribution within the temperature
group, and what information does grouping preserve or conceal?

The notebook's BE1 configuration compares full features, temp without atemp,
and atemp without temp; all other predictors remain unchanged. It uses forest
and boosting with candidate index 1 fixed from BE0 validation selections, seed
42, and the original chronological cohorts. This is six fits, not a new
hyperparameter search or independent seed-robustness study. BE1 increases the
permutation cohort/repeats to 1,024/20 and SHAP rows/background/cycles to
64/128/10. These budgets are declared settings, not evidence of convergence.

Outputs include raw individual temperature reliance, absolute allocation
shares, joint reliance, signed per-row summed SHAP group contributions,
cancellation ratios, and local group contribution disagreement. Summing
raw-feature SHAP contributions is not recomputing a grouped Shapley game.

BE0 is now inspected. BE1 on the same TEST period is an exploratory follow-up,
not untouched confirmatory evidence. Do not tune budgets or model choices until
the desired explanation appears. Background/budget sensitivity, paired
temporal-block uncertainty, rolling development windows, and controlled
synthetic mechanisms remain subsequent work; this run does not implement them.

BE0 files remain frozen. Write BE1 into a new guarded run directory. This
analysis lives under Git-ignored results; explicitly archive it when preparing
a public research repository.