"""Raw-feature perturbations on a common cohort and fixed TRAIN references."""
import time

import numpy as np
import pandas as pd

from bike_evaluation import regression_metrics


def explanation_indices(dataset, cfg):
    count = min(cfg.explanation_rows, len(dataset))
    return np.sort(np.random.default_rng(cfg.explanation_seed).choice(len(dataset), count, replace=False))


def shap_explanation_indices(indices, cfg):
    generator = np.random.default_rng(np.random.SeedSequence([cfg.explanation_seed, 1]))
    count = min(cfg.shap_rows, len(indices))
    return np.sort(generator.choice(indices, count, replace=False))


def group_shap_contributions(contributions, feature_names, groups):
    values = np.asarray(contributions, dtype=np.float64)
    return {name: values[:, [list(feature_names).index(feature) for feature in features]].sum(axis=1).tolist()
            for name, features in groups.items()}


def temperature_attribution_summary(explanations):
    panels = explanations["individual"]["panels"]
    individual = {name: panels[name]["mean_mse_increase"] if name in panels else None
                  for name in ("temp", "atemp")}
    scale = sum(abs(value) for value in individual.values() if value is not None)
    result = dict(individual_mse_increase=individual,
                  individual_absolute_share={name: abs(value) / scale if value is not None and scale else None
                                             for name, value in individual.items()},
                  joint_mse_increase=explanations["grouped"]["panels"]["temperature"]["mean_mse_increase"],
                  interpretation="Absolute shares describe allocation, not additive loss decomposition or causal credit")
    if "shap" in explanations:
        shap_result = explanations["shap"]
        magnitudes = dict(zip(shap_result["feature_names"], shap_result["mean_absolute_contribution"]))
        scale = sum(magnitudes.get(name, 0) for name in ("temp", "atemp"))
        joint = shap_result["mean_absolute_group_contribution"]["temperature"]
        result["shap"] = dict(individual_mean_absolute_contribution={name: magnitudes.get(name) for name in ("temp", "atemp")},
                              individual_absolute_share={name: magnitudes[name] / scale if name in magnitudes and scale else None
                                                         for name in ("temp", "atemp")},
                              mean_absolute_joint_contribution=joint,
                              joint_to_sum_absolute_ratio=joint / scale if scale else None,
                              interpretation="Sum raw-feature SHAP values before taking absolute values; not group Shapley recomputation")
    return result


def permutation_importance(model, dataset, indices, groups, cfg, *, retain_row_losses=False):
    inputs = dataset.inputs.iloc[indices].reset_index(drop=True)
    targets = dataset.targets[indices]
    baseline_predictions = np.asarray(model.predict(inputs), dtype=np.float64)
    baseline = regression_metrics(targets, baseline_predictions)["mse"]
    baseline_losses = (baseline_predictions - targets)**2
    panels = {}
    start = time.perf_counter()
    for name, features in groups.items():
        if not features or not set(features).issubset(inputs.columns):
            raise ValueError(f"Invalid permutation group: {name}")
        generator = np.random.default_rng(cfg.explanation_seed)
        increases = []
        row_losses = []
        for repeat in range(cfg.permutation_repeats):
            order = generator.permutation(len(inputs))
            perturbed = inputs.copy()
            perturbed.loc[:, list(features)] = inputs.iloc[order][list(features)].to_numpy()
            predictions = np.asarray(model.predict(perturbed), dtype=np.float64)
            score = regression_metrics(targets, predictions)["mse"]
            increases.append(score - baseline)
            if retain_row_losses:
                row_losses.append((predictions - targets)**2 - baseline_losses)
        panels[name] = dict(features=list(features), mean_mse_increase=float(np.mean(increases)),
                            std_mse_increase=float(np.std(increases)),
                            repeat_mse_increases=increases)
        if retain_row_losses:
            panels[name]["row_mean_mse_increase"] = np.mean(row_losses, axis=0).tolist()
    return dict(baseline_mse=baseline, rows=len(indices), panels=panels,
                seconds=time.perf_counter() - start,
                interpretation="MSE increase under marginal permutation; repeat SD is not a confidence interval")


def explain_shap_model(model, train, test, groups, cfg, indices, *, background_seed=None,
                       shap_seed=None, permutation_cycles=None, background_rows=None):
    import shap
    background_seed = cfg.explanation_seed if background_seed is None else background_seed
    shap_seed = cfg.explanation_seed if shap_seed is None else shap_seed
    permutation_cycles = cfg.shap_permutation_cycles if permutation_cycles is None else permutation_cycles
    background_rows = cfg.shap_background_rows if background_rows is None else background_rows
    for name, value in (("background_seed", background_seed), ("shap_seed", shap_seed),
                        ("permutation_cycles", permutation_cycles), ("background_rows", background_rows)):
        minimum = 0 if name.endswith("seed") else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum or (name.endswith("seed") and value >= 2**32):
            raise ValueError(f"Invalid SHAP control: {name}")
    indices = np.asarray(indices)
    if indices.dtype.kind not in "iu":
        raise ValueError("SHAP row indices must be integers")
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(test)):
        raise ValueError("SHAP requires valid unique explicit row indices")
    background_indices = np.sort(np.random.default_rng(background_seed).choice(
        len(train), min(background_rows, len(train)), replace=False))
    background = train.inputs.iloc[background_indices].to_numpy()
    explained = test.inputs.iloc[indices].to_numpy()

    def predict(values):
        return model.predict(pd.DataFrame(values, columns=list(test.feature_names)))

    start = time.perf_counter()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=shap_seed)
    values = explainer(explained, max_evals=(2 * len(test.feature_names) + 1) * permutation_cycles)
    contributions = np.asarray(values.values, dtype=np.float64)
    base_values = np.asarray(values.base_values, dtype=np.float64).reshape(-1)
    if contributions.shape != explained.shape or base_values.shape != (len(indices),):
        raise ValueError("SHAP result dimensions do not match the fixed cohort")
    if not np.isfinite(contributions).all() or not np.isfinite(base_values).all():
        raise ValueError("Nonfinite SHAP values")
    reconstruction = base_values + contributions.sum(axis=1)
    group_values = group_shap_contributions(contributions, test.feature_names, groups)
    return dict(feature_names=list(test.feature_names), row_ids=test.row_ids[indices].tolist(),
                timestamps=test.timestamps[indices].astype(str).tolist(),
                sampling="Explicit fixed cohort; background and SHAP RNG controls cannot change explained rows",
                background_seed=background_seed, shap_seed=shap_seed, permutation_cycles=permutation_cycles,
                background_row_ids=train.row_ids[background_indices].tolist(),
                values=contributions.tolist(), base_values=base_values.tolist(),
                mean_absolute_contribution=np.abs(contributions).mean(axis=0).tolist(),
                group_contributions=group_values,
                mean_absolute_group_contribution={name: float(np.abs(values).mean())
                                                  for name, values in group_values.items()},
                reconstruction_max_abs_error=float(np.abs(reconstruction - predict(explained)).max()),
                seconds=time.perf_counter() - start,
                interpretation="Marginal background masking, not conditional or causal SHAP")


def explain_bike_model(model, train, test, groups, cfg):
    indices = explanation_indices(test, cfg)
    individual = {feature: (feature,) for feature in test.feature_names}
    result = dict(row_ids=test.row_ids[indices].tolist(),
                  individual=permutation_importance(model, test, indices, individual, cfg),
                  grouped=permutation_importance(model, test, indices, groups, cfg))
    if cfg.compute_shap:
        result["shap"] = explain_shap_model(model, train, test, groups, cfg,
                                           shap_explanation_indices(indices, cfg))
    return result


def additive_group_importance(model, dataset, indices, groups):
    """Mean absolute joint contribution for an interaction-free EBM."""
    inputs = dataset.inputs.iloc[indices]
    predictions = model.predict(inputs)
    contributions = np.asarray(model.eval_terms(inputs), dtype=np.float64)
    if any(len(term) != 1 for term in model.term_features_):
        raise ValueError("This first experiment requires an interaction-free EBM")
    reconstruction = np.asarray(model.intercept_).reshape(-1)[0] + contributions.sum(axis=1)
    if not np.allclose(predictions, reconstruction, rtol=1e-6, atol=1e-6):
        raise ValueError("EBM terms do not reconstruct predictions")
    panels = {}
    for name, features in groups.items():
        positions = [term_index for term_index, term in enumerate(model.term_features_)
                     if dataset.feature_names[term[0]] in features]
        panels[name] = dict(features=list(features),
                            mean_absolute_joint_contribution=float(np.abs(contributions[:, positions].sum(axis=1)).mean()))
    return dict(panels=panels, rows=len(indices),
                interpretation="Absolute value after summing centered GAM terms; not MSE reliance or group Shapley values")