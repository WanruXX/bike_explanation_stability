# Bike Explanation Stability

An exploratory study of representation-dependent attribution in retrospective
hourly bike-demand regression using the public UCI Bike Sharing dataset.

This independent study was motivated by Explainable AI coursework. It compares
individual and grouped feature attribution across predictive models, feature
representations, and explanation settings.

**Research question:** How representation-dependent is attribution within a
correlated feature group, and what information does grouping preserve or conceal?

## Usage

Run these commands from the repository root. Use the same Python
environment for installation, tests, and experiments.

```bash
python -m pip install -r requirements-shap.txt
python main.py --help
python main.py test
```

The finalized BE1/BE2 workflow requires SHAP. Base `requirements.txt` alone is
for non-SHAP exploration. Tests use small synthetic fixtures, including a small
EBM API fit; they do not download data or train on the real dataset.

Download the approximately 1.1 MB public hourly CSV only if it is absent:

```bash
python main.py download
```

Run both finalized stages, reusing the live BE1 models in BE2:

```bash
python main.py all --be1-run-label be1_replay --be2-run-label be2_replay
```

Alternatively, run stages separately. A new Python process has no live BE1
models, so BE2 reconstruction requires explicit authorization:

```bash
python main.py be1 --run-label be1_new
python main.py be2 --source-run-label be1_new --run-label be2_new --rebuild-models
```

To analyze the archived BE1 directly, set `--source-run-label` to
`be1_temperature_representation` and choose an unused BE2 output label. Saved
NumPy, SciPy, scikit-learn, and SHAP versions must match; dependency ranges alone
do not guarantee an identical archived environment. Reconstructed models must
reproduce saved TEST predictions before robustness analysis proceeds.

Optional global path arguments go before the command:

```bash
python main.py --data-dir /path/to/data --results-dir /path/to/results be1 --run-label be1_new
```

`--n-jobs` is available for `be1` and `all`, defaulting to one. CLI experiments
automatically run preflight tests before fitting or explaining. The combined
command checks both output directories before fitting BE1. Existing completed
or failed directories are never replaced. Data downloads and dependency
installation are separate from experiment execution.

### Google Colab Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WanruXX/bike_explanation_stability/blob/main/bike_experiment.ipynb)

[bike_experiment.ipynb](bike_experiment.ipynb) runs the CLI in Google Colab or
a local Jupyter environment. Setup clones the repository into
`bike_explanation_stability` when absent, reuses existing checkouts, and installs
`requirements-shap.txt`. A CPU runtime is sufficient.

Run setup and preflight, enable `DOWNLOAD_DATA` if the CSV is absent, then choose
`WORKFLOW="all"`, `"be1"`, or `"be2"` and explicitly enable `RUN_EXPERIMENT`.
All experiment commands call `main.py` using the kernel's Python executable.
`all` reuses live models within its subprocess; separate `be2` execution requires
`REBUILD_MODELS=True` and a completed `SOURCE_RUN_LABEL`. Use fresh output labels.
BE0 is documented as historical context, not offered as an executable CLI stage.

Enable `LOAD_BE1_REPORT` or `LOAD_BE2_REPORT` for read-only inspection without
training. Archived results are Git-ignored and may need uploading to a fresh
checkout. Default Colab packages may not match the archived BE1 environment.
The optional export cell archives one selected run and can download it in Colab;
export each required run before temporary runtime storage is discarded.

## Study Design

| Stage | Purpose and design | Main observations |
|---|---|---|
| BE0: pilot | Four model families, two representations, three seeds; 48 candidate fits and 24 selected models | Localized temperature-credit differences; forest/boosting qualified as validation-near-optimal. Detected a chronological-prefix SHAP sampling defect; froze BE0 and corrected sampling for BE1. |
| BE1: representation study | Forest/boosting, seed 42, full/temp/atemp representations; six fixed-candidate TRAIN-only fits | Predictive performance largely retained; individual allocation and local group explanations differ. |
| BE2: robustness | Fixed BE1 functions and cohorts; seven matched SHAP settings and paired one-day/seven-day block intervals | Comparative findings persist across tested controls. Absolute attribution remains background-dependent; supports a bounded exploratory case-study conclusion. |

TRAIN is 2011, validation is January-June 2012, and TEST is July-December 2012.
All representations share rows, targets, and other predictors. Candidate index
1 for both BE1 families was fixed from BE0 validation selections; BE1 performs
no new search. BE2 reconstructs or reuses those fits, rather than producing
independent training replications. Since TEST was inspected in BE0, later stages
are exploratory rather than confirmatory.

The CLI runs BE1/BE2; BE0 is retained as historical context.
The recorded runs use Python 3.13.15,
NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1,
interpret-core 0.7.8, and SHAP 0.49.1.

## Findings

**Grouping approximately preserves aggregate temperature magnitude, but it does
not make individual credit or local explanations invariant.**

- Across all seven matched BE2 settings, temp's share of absolute individual
  temperature SHAP magnitude is 38.9-43.9% for forest versus 58.1-62.3% for
  boosting when both features are present. The tested ranges do not overlap.
- Within each model family, replacing the two-feature representation with one
  temperature feature changes aggregate group SHAP magnitude by at most about
  4.6% for forest and 7.8% for boosting under matched settings.
- Local group disagreements remain: representation comparisons have SHAP MAE
  of about 2.90-7.54 rentals/hour across tested settings, versus 11.52-17.43
  for cross-family comparisons. Similar aggregate magnitude does not mean the
  same explanation for the same observation.
- All six within-family representation RMSE-difference intervals include zero
  under both block lengths. This supports small, uncertain risk differences,
  not formal statistical equivalence. Boosting is more accurate than forest
  on this TEST period, so cross-family comparisons are not identical-prediction
  or equal-risk experiments.

Temperature-group local SHAP sensitivity against each model's saved baseline:

| Changed control | Observed MAE range, rentals/hour |
|---|---:|
| SHAP RNG seed, same background | 1.00-2.15 |
| Cycles: ten to thirty | 0.79-1.72 |
| Background rows: 128 to 256, same seed | 1.06-1.63 |
| TRAIN background seed, size 128 | 5.08-14.89 |

Background changes are the largest observed sensitivity and also change the
empirical reference game. In full-feature models they flip the temperature-group
contribution's sign for 3-6 of the 64 explained rows. Matched comparative results
are more robust than reference-independent claims about absolute local credit.
The larger budget/background cases are comparators, not ground truth or proof
of convergence.

These findings are specific to this dataset and experimental design; they do
not establish causal faithfulness or universal stability. Broader claims require
predeclared repetitions that genuinely vary reference samples, fitted functions,
or development periods. BE0 boosting predictions were identical across seeds;
repeating a deterministic fit is not independent evidence.

Generated artifacts and detailed run analyses are stored under `results/`,
which is excluded from version control. The findings above summarize the
recorded runs.

## BE1 Protocol

BE1 examines representation-dependent attribution within the temperature group.

The finalized BE1 protocol uses forest and boosting, seed 42,
three representations (`full`, `temperature_only`, `apparent_temperature_only`).
The latter drops temp while retaining atemp and every other predictor. Candidate
index 1 is fixed for both families from BE0's validation selections, so BE1 runs
six fits without reselecting hyperparameters after each representation change.
The original split boundaries are retained; revisiting the same TEST makes BE1
exploratory, not confirmatory. The notebook and CLI use the same experiment
settings.

BE1 uses 1,024 permutation rows/20 repeats and 64 SHAP rows/128 TRAIN reference
rows/10 cycles. SHAP rows are now randomly selected from the common permutation
cohort before sorting, not its chronological prefix. Timestamps and sampling
policy are recorded. This corrects a BE0 sampling defect without rewriting BE0.
Larger budgets do not replace explicit background/budget convergence checks.

Temperature allocation shares, joint reliance, summed signed SHAP group values,
cancellation ratios, and paired local group disagreement are saved. Summed
raw-feature SHAP values are not recomputed group Shapley values. Missing
temperature features are reported as absent, not given invented importance.
Temporal-block uncertainty and explanation sensitivity sweeps are evaluated
separately in BE2.

## BE2 Robustness

Allocation differs across models and representations despite retained predictive
quality. Aggregate temperature magnitude is relatively stable within a family;
local and cross-family explanations are not invariant. BE2 adds conditional
uncertainty and sensitivity checks, not causal or confirmatory conclusions.

The notebook delegates robustness execution to `main.py`, which calls
`bike_robustness_pipeline.py`. Both `RUN_EXPERIMENT` and `REBUILD_MODELS` start
as false. Use `all` to reuse live BE1 models within one subprocess, or authorize
separate `be2` reconstruction with `REBUILD_MODELS=True`. Saved reports can be
inspected without training.

Reconstruction fits the saved settings on original TRAIN only, without candidate search
or validation refit. Data hash, splits, feature groups, saved cohorts, references,
and TEST predictions are checked. Recorded NumPy, SciPy, scikit-learn and SHAP
versions must match BE1 so a package change is not confused with a control change.
Prediction mismatch aborts the run.

BE2 holds explained row IDs fixed. Each setting changes exactly one control from
the saved baseline: SHAP RNG seed, TRAIN background seed, permutation-cycle
budget, or background size. Default settings are seven matched cases including
the reused baseline: 36 new SHAP computations for BE1's six models. This can be
expensive, especially for forest. Background
and size sensitivity concern the empirical reference game as well as numerical
variation, and are distinct from SHAP RNG sensitivity. The higher-budget case
is not ground truth or proof of convergence. Cross-model comparisons use matched
settings so within-model sensitivity can be inspected alongside disagreement.

Paired non-overlapping calendar-block percentile intervals use one-day and
seven-day bins anchored to the cohort's first observed day. The same sampled
occupied blocks/weights are used across compared arms. They summarize full-TEST
RMSE/MAE differences and prediction disagreement, fixed-cohort permutation loss
increases and differences, baseline mean absolute group SHAP, and paired local
SHAP differences. Per-row permutation losses are averaged over the original
paired orders before bootstrapping. Intervals are conditional on the fitted
models, fixed perturbations/reference and cohorts; they exclude fitting and
permutation-design uncertainty. The sparse SHAP cohort represents occupied
blocks only. Temporal shift, block-length sensitivity, and multiplicity limit
inferential claims. Intervals remain exploratory because TEST was inspected
during development.

Use a fresh `run_label` for each attempt. BE2 writes a guarded manifest,
`report.json`, `summary.txt`, raw SHAP per-case records, and per-row permutation
loss records. It never overwrites BE0/BE1. Synthetic preflight tests validate
invariants and orchestration; they are separate from the empirical study.

## BE0 Protocol

- Data: UCI Bike Sharing, hourly `hour.csv`, about 1.1 MB, Capital Bikeshare
  2011-2012. The source CSV is validated and hashed. No rows are silently dropped.
- TRAIN: timestamps before 2012-01-01. Validation: 2012-01-01 through
  2012-06-30. TEST: 2012-07-01 onward. All variants share identical row IDs.
- Predictors: calendar and observed weather. `casual` and `registered` are
  excluded because they sum to `cnt`; IDs and the target are also excluded.
- Full features versus `temperature_only`, which removes apparent temperature
  (`atemp`) but keeps every other predictor. It does not mean temperature is the
  only model input.
- Four models: spline ridge, random forest, histogram gradient boosting, and
  interaction-free Explainable Boosting Machine (EBM). Two predefined candidate
  configurations per model and three seeds per variant: 48 total fits by default.
- Preprocessing fits on TRAIN only. Candidate selection uses chronological
  validation RMSE. No refit on validation, random internal validation split, or
  test-based hyperparameter selection. EBM uses no internal validation holdout.
- Models within 5% of the best validation RMSE are flagged as near-optimal,
  separately within each feature variant. All selected models remain in reports;
  the flag does not imply equal predictions or statistically equivalent risk.
- Predictions are not clipped. Report RMSE, MAE, bias, R-squared, and negative
  prediction counts. Predictive disagreement is measured separately.
- Individual and jointly grouped marginal permutation importance use the same
  fixed random subset of TEST rows and paired permutation orders across models.
  Groups: calendar, daily schedule, temperature, and other weather.
- EBM additionally reports mean absolute joint centered-term contributions,
  following the 2025 GAM group-importance idea. This is not loss-based importance.
- Optional SHAP uses a small fixed TRAIN background and TEST cohort with
  model-agnostic permutation SHAP. The raw-feature background masker is marginal,
  not dependence-aware or causal. SHAP reconstruction error is recorded.
- Pairwise comparisons vary one factor at a time: seed, model family, or feature
  representation. Individual-feature rankings are not compared across schemas.

The held-out year category was unseen in TRAIN. One-hot models map it to the
unknown-category representation; EBM may handle it differently. This is an
explicit extrapolation property of the chronological protocol. Regional split discovery, new model
architectures, conditional SHAP, and temporal-shift analyses are not yet included.

## Interpretation

The study tests whether group-level importance is more stable than individual
importance when features share information. Grouping reduces explanatory
resolution; stability alone is not
faithfulness, truth, confidence in an individual prediction, or causality.

Marginal perturbation can create unrealistic feature combinations. Joint
permutation preserves dependence within a group, not across groups. The
permutation-repeat SD is descriptive Monte Carlo variability, not a sampling
confidence interval. Group and individual rankings have different dimensions,
so their agreement scores are not directly calibrated against one another.
Hourly observations are dependent; seeds are not independent datasets. One city
and two years do not establish general validity. Weather is observed, so this
is not an operational day-ahead forecasting benchmark.

Protocol changes made after TEST inspection do not constitute independent
holdout evaluation. Synthetic checks validate implementation behavior, not
real-data findings.

## Artifacts

Each run writes settings, package versions, dataset hash and provenance,
candidate histories, per-model metrics and explanation values, pairwise
comparisons, a text summary, and compressed aligned TEST predictions.
The `all` command reuses fitted models within one process; the notebook does
not retain those models in kernel memory. Separate BE2 execution reconstructs
models from the saved BE1 settings and verifies their predictions.
Data and generated results are excluded from version control.

## Relationship To Prior Work

The 2025 GAM group-importance paper motivates summing centered shape-function
contributions before taking absolute values. BE0's EBM panel implements this
core measure, but averages over a held-out subset rather than the paper's TRAIN
observations. EBM was outside BE0's near-optimal set and is absent from BE1/BE2.
Thus the finalized study uses related grouping ideas, not the paper's GAM
estimator, synthetic mechanisms, clinical datasets, or full experimental protocol.

BE1/BE2 use joint marginal permutation and summed raw-feature marginal SHAP.
These have different estimands from centered GAM-term importance. Summed SHAP
is not a recomputed grouped Shapley game. The findings align with the motivation
for examining joint signals while qualifying what grouping may conceal; they
are not a replication or refutation of the GAM paper.

Effector provided a public Bike-Sharing example and research anchor. Its regional
partitioning and PDP/ALE methods are outside this study's scope. The project is
an independent empirical sensitivity study informed by the cited literature.

## Sources

- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
  Hadi Fanaee-T (2013), DOI 10.24432/C5W894, CC BY 4.0.
- [Molnar, Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/).
- [Fisher, Rudin, Dominici, Model Class Reliance](https://arxiv.org/abs/1801.01489),
  JMLR 2019. A finite candidate collection here is not a certified Rashomon set.
- [Effector](https://arxiv.org/abs/2404.02629), 2024 preprint, revised 2025,
  [official code](https://github.com/givasile/effector). Its hourly bike example is
  a future regional-effect reproduction anchor, not yet reproduced here.
- [The Most Important Features in Generalized Additive Models Might Be Groups
  of Features](https://arxiv.org/abs/2506.19937), 2025 preprint;
  [InterpretML](https://github.com/interpretml/interpret).
- [SHAP](https://github.com/shap/shap). Conditional Shapley methods such as
  [shapr](https://github.com/NorskRegnesentral/shapr) are potential extensions,
  not implementations included in this first experiment.