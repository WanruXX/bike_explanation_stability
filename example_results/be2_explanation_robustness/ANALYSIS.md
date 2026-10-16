# BE2 Review: Robust Comparisons, Reference-Dependent Attributions

Review date: 2026-10-07. This review interprets completed BE2 artifacts and
preserves BE0, BE1, and BE2 numerical outputs unchanged. It does not rerun models
or explanations.

## Bottom Line And Stopping Decision

BE2 supports a bounded empirical conclusion: temperature representations retain
similar predictive performance while redistributing individual attribution;
grouping approximately preserves aggregate temperature magnitude but conceals
local differences. These comparative findings survive the tested matched
explanation settings. Absolute SHAP contributions and some local signs remain
reference-dependent.

No additional broad diagnostic stage is required before writing up this
exploratory case study or starting planned repetitions. There is no identified
blocking failure in the reviewed completed artifacts. This is not a certificate
that every possible implementation or statistical issue has been excluded.
Additional experiments are needed only to support broader claims, such as
training-sample robustness, reference-size convergence, generalization to other
time periods, or replication of a published method's experiments.

Freeze this run rather than repeatedly changing TEST-driven settings. Decide the
scope and criteria of any repetitions before executing them. The same inspected
TEST period remains exploratory, regardless of new run labels.

## Evidence And Run Integrity

Sources: [manifest.json](manifest.json), [report.json](report.json),
[summary.txt](summary.txt), 42 per-case SHAP files, and six per-row permutation
files. BE1's settings and baseline explanations are retained.

- Six models: forest and histogram boosting, three representations, fitted seed
  42. All were explicitly reconstructed from saved settings on original TRAIN,
  not selected again or refitted on validation. Completion follows the pipeline's
  guards for agreement with saved TEST predictions and source data/cohorts.
- All recorded package versions match BE1, including NumPy 2.1.3, scikit-learn
  1.6.1, and SHAP 0.49.1.
- Seven matched cases: saved baseline; two additional TRAIN background seeds;
  two additional SHAP RNG seeds; 30 rather than ten cycles; 256 rather than 128
  background rows. Each new case changes one baseline control.
- Explained rows remain fixed: 64 SHAP rows and 1,024 permutation rows, with
  20 paired permutation repeats. Full-TEST prediction intervals use 4,376 rows.
- Percentile intervals use 1,000 block-bootstrap draws, seed 2027, and nominal
  95% coverage. One-day/seven-day occupied block counts are respectively
  184/27 for predictions, 183/27 for permutation, and 53/24 for sparse SHAP.
- Dataset SHA256:
  `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

Successful reconstruction is not another model replication. The six source
functions remain the seed-42 functions. SHAP additivity checks establish
reconstruction, not explanation faithfulness or estimator convergence. A saved
unit-test execution log was not used as evidence in this review.

## Numerical Sensitivity Versus Reference Sensitivity

The table reports temperature-group local contribution MAE against each model's
saved baseline, in rentals/hour, pooled descriptively across the six models and
tested settings. These are ranges, not sampling intervals or independent trials.

| Changed control | Minimum MAE | Maximum MAE |
|---|---:|---:|
| SHAP RNG seed, fixed background | 0.995 | 2.151 |
| Cycles: ten to thirty | 0.791 | 1.717 |
| Background size: 128 to 256, fixed seed | 1.056 | 1.628 |
| Background seed, fixed size 128 | 5.081 | 14.887 |

Approximation sensitivity is modest relative to most observed comparisons, but
not zero. Ten-to-thirty cycles is one budget comparison, not proof of convergence.
Only one seed was tested at background size 256; its small difference cannot
establish adequate reference size generally.

Changing backgrounds changes the empirical reference distribution and hence the
attribution game, not merely numerical noise. It has the largest observed effect.
For full-feature models, temperature magnitudes change from 33.85 to as much as
41.01 for forest and from 21.56 to 24.73 for boosting under alternate 128-row
backgrounds.

Additional descriptive calculations from saved full-feature temperature-group
vectors separate signed offsets from centered pattern changes. Centered MAE is
the mean absolute value of each contribution difference after subtracting its
cohort mean; it does not replace the original attribution or prove shape equality.

| Full-feature model | Background seed | Signed mean shift | Centered MAE | Sign flips out of 64 |
|---|---:|---:|---:|---:|
| Forest | 2718 | 14.659 | 3.234 | 5 |
| Forest | 3141 | 8.715 | 1.289 | 4 |
| Boosting | 2718 | 8.291 | 2.916 | 6 |
| Boosting | 3141 | 5.511 | 1.671 | 3 |

In these four comparisons every temperature contribution moves in the positive
direction, so MAE equals the signed mean shift. Much of the change is an offset,
but residual observation-level variation remains. Local signs near zero are not
robust to the tested reference changes. This calculation concerns two
full-feature models, not all groups or all representations.

## Comparative Findings Survive Matched Settings

Background effects can be shared across compared arms. Therefore, large
within-model reference shifts do not automatically invalidate comparisons made
under the same reference settings.

| Temperature-group local comparison | Minimum MAE across cases | Maximum MAE across cases |
|---|---:|---:|
| Forest: full vs temp without atemp | 6.116 | 6.699 |
| Forest: full vs atemp without temp | 2.904 | 3.565 |
| Forest: temp vs atemp representations | 6.889 | 7.544 |
| Boosting: full vs temp without atemp | 4.223 | 4.779 |
| Boosting: full vs atemp without temp | 3.513 | 3.991 |
| Boosting: temp vs atemp representations | 5.191 | 5.888 |
| Forest vs boosting: full | 14.086 | 17.434 |
| Forest vs boosting: temp representation | 13.551 | 16.473 |
| Forest vs boosting: atemp representation | 11.520 | 15.281 |

With both features present, temp's share of absolute individual temperature SHAP
magnitude is 38.9-43.9% for forest versus 58.1-62.3% for boosting across all seven
settings. These descriptive ranges do not overlap. They support model-dependent
allocation under the tested controls, not a confidence interval for true credit.

Relative to full features under each matched setting, aggregate temperature
SHAP magnitude changes by at most approximately 4.6% for forest and 7.8% for
boosting. This is approximate preservation, not invariance. Taking an absolute
value after summing contributions is important: individual absolute magnitudes
cannot simply be added to obtain joint magnitude. A one-feature representation's
100% allocation share remains tautological.

## What The Block Intervals Support

Selected seven-day paired percentile intervals are shown below. Differences
are second arm minus first; SHAP MAE is unsigned local disagreement.

| Comparison and measure | Estimate | Conditional 95% interval |
|---|---:|---:|
| Forest full to temp: TEST RMSE difference | -0.684 | [-1.804, 0.240] |
| Boosting full to temp: TEST RMSE difference | 0.692 | [-0.343, 1.722] |
| Full forest to full boosting: TEST RMSE difference | -10.916 | [-13.549, -8.097] |
| Full forest to full boosting: temperature SHAP MAE | 14.366 | [10.707, 17.953] |
| Full forest to full boosting: absolute group SHAP magnitude difference | -12.292 | [-15.834, -8.676] |
| Boosting full to temp: absolute group SHAP magnitude difference | 1.590 | [0.311, 2.790] |

All six within-family representation RMSE-difference intervals include zero
under both block lengths. This supports small, uncertain risk differences, not
formal statistical equivalence. Forest full-to-temp MAE does improve by about
1.09, with seven-day interval [-1.97, -0.32], so not every predictive metric is
unchanged.

Grouping does not remove systematic magnitude changes: boosting full-to-temp
has a small positive SHAP-magnitude difference and a negative joint permutation
MSE-increase difference of -669.47, with seven-day interval [-1119.26, -226.58].
These are different estimands, not contradictory results. Forest full-to-temp
permutation difference includes zero with seven-day blocks but not one-day
blocks; avoid strong claims based on that boundary result.

Cross-family comparisons have clearer risk and attribution differences under
both block lengths. Boosting is more accurate on this TEST period, and forest
and boosting predictions differ. Do not describe these as equally accurate
models explaining identical functions.

Intervals condition on fitted functions, saved cohorts, fixed SHAP reference,
and fixed averaged perturbation losses. They do not include fitting uncertainty,
reference-sampling uncertainty, or permutation-design uncertainty. Sparse SHAP
rows are not dense day-level observations. Temporal shift, nonstationarity, and
multiple comparisons limit inferential claims. An MAE interval above zero is
not by itself a calibrated test against explanation-estimator noise. Marginal
SHAP masking and joint permutation can break between-group dependence; these
results do not establish that the perturbed inputs are realistic or causal.

## Relationship To The Papers

### GAM Group-Importance Paper

Source checked: Bosschieter et al.,
[The Most Important Features in Generalized Additive Models Might Be Groups of Features](https://arxiv.org/html/2506.19937v1),
especially Sections 2.3, 3, and 5. This review refers to that version, not an
unverified later revision.

The paper defines group importance by averaging the absolute value of the sum
of a group's centered GAM shape-function contributions. For observations in a
set S and group G, its core form is

$$
I_G = \frac{1}{|S|}\sum_{x\in S}\left|\sum_{j\in G} f_j(x_j)\right|.
$$

The paper evaluates its definition over training observations and studies
synthetic additive/conflicting signals, correlation regimes, computation costs,
and neuroscience/healthcare applications. It argues that individual feature
analysis can miss useful joint signals, and that group importance can account
for cancellation. It does not establish that grouped explanations must be
invariant across model families, retrained representations, or SHAP references.

What we actually used:

- BE0's EBM panel directly implements the core functional form through
  `additive_group_importance` in [bike_explanations.py](../../bike_explanations.py): obtain centered
  additive terms with `eval_terms`, sum within groups, then average absolute
  values. Interactions are disabled. We average over a held-out subset, not the
  paper's training-data average, and do not reproduce its full experiments.
- EBM did not meet BE0's near-optimal validation criterion and is absent from
  BE1/BE2. Thus the main robustness conclusions do not directly validate the
  paper's GAM estimator.
- BE1/BE2 use joint marginal permutation and grouped sums of raw-feature
  marginal SHAP. SHAP magnitude has the analogous form with attribution values
  instead of GAM shape functions, but the estimand and reference dependence are
  different. Summing feature SHAP is not recomputing a grouped Shapley game.
- Joint permutation is an established loss-reliance measure, not this paper's
  new GAM importance definition. Our allocation shares are not the paper's
  relative total-importance or explained-variation measure.

Our results are qualitatively consistent with the motivation for examining
groups, while adding a practical qualification: aggregate preservation can
coexist with concealed allocation/local differences and reference dependence.
This is a related empirical sensitivity study, not a replication, a general
extension theorem, a refutation, or an established novel contribution.

### Effector Paper

Source checked: Gkolemis et al.,
[Effector: A Python package for regional explanations](https://arxiv.org/abs/2404.02629).
Effector concerns global and regional feature effects, partitioning input space
to expose heterogeneity that a single global effect can hide.

Effector was a Bike-Sharing example and public-code exploration anchor. Its
package, regional partitioning, and PDP/ALE effect estimators have not been used
in BE0, BE1, or BE2. Sharing the dataset is not reproducing its protocol or
validating its methods. Our fixed-row local SHAP comparisons are not regional
feature-effect analysis. Reproducing Effector would be a separate scope, not a
missing validity gate for the present attribution study.

## Repetitions And Claim Boundaries

For the current bounded case-study conclusion, stop diagnostics here and write
up the evidence with the stated limitations. There is no need to add models,
ALE/PDP, causal estimators, or a synthetic mechanism merely to make this run
look more complete.

If pursuing stronger claims, the most useful planned extensions are repeated
background samples at larger sizes, and genuine fitting/data-period variation.
BE0 already showed identical histogram-boosting predictions across three seeds;
changing a seed on a deterministic procedure is not an independent replication.
Choose repetitions that actually vary fitted functions or temporal development
samples, retain predeclared settings, and keep reference changes separate from
fitting changes. They remain follow-up experiments, not prerequisites for the
present descriptive conclusion.

## Reportable Conclusion

In this bike-demand case study, temperature representations retain similar
predictive performance but redistribute individual attribution. Grouping
approximately preserves aggregate temperature magnitude while concealing local
differences. Comparative findings persist across the tested matched explanation
settings, although absolute SHAP contributions and some local signs depend on
the reference background. The evidence is exploratory and conditional on the
studied fitted models and cohorts, not proof of causal faithfulness, universal
stability, or a reproduction of the cited papers.

This review lives under Git-ignored results. Explicit archival is necessary
when preparing the public repository. No new experiment was executed to create
this document; the offset diagnostics use arithmetic on existing saved vectors.