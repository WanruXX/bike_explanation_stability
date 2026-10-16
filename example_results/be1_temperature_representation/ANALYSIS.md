# BE1 Review: Temperature Representation And Attribution

Review date: 2026-10-07. Sources are the completed `manifest.json`, `report.json`,
`summary.txt`, and six per-model records in this directory. This analysis does
not modify their values or regenerate their explanations.

## Question And Answer

Question: How representation-dependent is attribution within a correlated
feature group, and what does grouping preserve or conceal?

BE1 gives a descriptive answer: individual credit varies with model and
representation while the overall temperature signal remains relatively stable
within each model family. Grouping conceals allocation differences and local
effect disagreements. It does not make explanations invariant, faithful, or
causal. This is pilot evidence, not a general theorem or independent confirmation.

## Run Integrity

- Six fits: forest and histogram boosting, three representations, seed 42.
- Candidate index 1 is fixed from BE0 validation selections; no new search.
- All other predictors, row IDs, and chronological splits remain unchanged.
- Permutation: 1,024 rows, 20 paired repeats. SHAP: 64 explained rows, 128 TRAIN
  background rows, ten permutation cycles. All models share cohort IDs.
- Corrected SHAP cohort covers July through December with 13, 13, 12, 10, 7, 9
  rows respectively. It is random, not a chronological prefix.
- Maximum SHAP reconstruction error is approximately 1.08e-12. This establishes
  additivity, not adequate estimator convergence or explanation faithfulness.
- Dataset SHA256 is
  `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

## Predictive Information Is Largely Retained

| Model | Representation | Validation RMSE | TEST RMSE |
|---|---|---:|---:|
| Forest | temp + atemp | 122.558 | 141.110 |
| Forest | temp without atemp | 122.789 | 140.425 |
| Forest | atemp without temp | 122.071 | 140.880 |
| Boosting | temp + atemp | 120.663 | 130.194 |
| Boosting | temp without atemp | 120.981 | 130.886 |
| Boosting | atemp without temp | 121.700 | 131.046 |

Scores barely change within each family. All six qualify under the within-5%
validation flag, which is relative to this small within-representation pool.
This does not establish statistical equivalence or equality of predictions.

## Allocation Depends On Model And Representation

With both features present, temp receives 25.7% of the absolute temperature
permutation-importance total in forest versus 75.7% in boosting. Its share of
mean absolute temperature SHAP contributions is 39.2% versus 58.8%.
Thus the two estimands both show allocation differences, but not the same split.
Permutation measures loss reliance; SHAP decomposes predictions against a
marginal reference distribution. Agreement is not required between them.

A surviving feature receives 100% under a one-feature representation by
definition. That percentage alone is not evidence of stability or true credit;
the nontrivial evidence is retained prediction quality and joint magnitude.

| Model | Representation | Joint permutation MSE increase | Mean absolute summed temperature SHAP |
|---|---|---:|---:|
| Forest | temp + atemp | 9,874.57 | 33.85 |
| Forest | temp without atemp | 10,436.40 | 34.89 |
| Forest | atemp without temp | 9,387.78 | 32.79 |
| Boosting | temp + atemp | 7,095.87 | 21.56 |
| Boosting | temp without atemp | 6,426.40 | 23.15 |
| Boosting | atemp without temp | 6,628.54 | 22.55 |

Mean absolute summed SHAP magnitude changes by roughly 3% for forest and at
most 7.4% for boosting relative to full features. Joint permutation changes are
larger, approaching 9.4% for boosting. Stability is approximate and estimand-specific.

## What Grouping Conceals

All four-group rank correlations equal 1, but ranking alone hides magnitude
differences. Full-feature forest versus boosting has local temperature-group
SHAP MAE 14.37 and RMSE 19.19 rentals/hour. Their joint SHAP magnitudes are 33.85
versus 21.56, so grouping is not invariant across model families.

Within a family, representation changes yield local group SHAP MAE from 3.25
to 7.25 and RMSE from 4.60 to 8.75. Forest full-to-temp-only MAE is 6.70;
boosting's is 4.48. Similar aggregate magnitudes do not imply the same explanation
for the same observation. Contributions must be summed before absolute values;
this is not a recomputed grouped Shapley game.

## Limits

There is only one fitted seed per family in BE1. The TEST period was already
inspected in BE0, so this follow-up is exploratory. The models underpredict
TEST by roughly 86-88 rentals/hour, leaving temporal shift as a competing
influence. One city and two years do not establish general validity.

The SHAP background and estimator randomness have not been varied. Ten cycles
are a setting, not a convergence result. Background changes also change the
empirical reference game, unlike pure estimator RNG variation. The 64-row
cohort is sparse; its local comparisons should not represent every hour equally.
Permutation-repeat SD is not temporal sampling uncertainty.

## Next: BE2 Robustness Checks

Keep BE1 models and explained row IDs fixed. Independently vary SHAP permutation
seeds, cycle budgets, TRAIN background samples, and background size. Report these
sources separately; do not pool reference sensitivity with Monte Carlo noise.
Use matched settings across models and representations when comparing them.

Add paired non-overlapping calendar-block percentile intervals with one-day and
seven-day blocks for TEST prediction differences, averaged per-row permutation
loss changes, and baseline local SHAP differences. These are exploratory,
conditional on fitted models, fixed perturbation orders/reference, and observed
cohorts. They do not include training uncertainty or guarantee stationarity.
Sparse SHAP-cohort intervals are not substitutes for dense temporal validation.

Use in-memory BE1 models where available. If rebuilding is necessary, it must
be explicit, use the saved fixed settings and original TRAIN only, and reproduce
saved TEST predictions before sensitivity analysis. No TEST-driven selection.

The new test code is not evidence until executed and reviewed. A controlled
synthetic correlation mechanism and rolling development windows remain later
extensions, rather than additional model families in this robustness stage.
Keep BE0 and BE1 files frozen; write BE2 into a fresh guarded directory.

This analysis is under Git-ignored results. Archive it explicitly for publication.