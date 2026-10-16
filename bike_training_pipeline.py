"""Notebook-facing experiment orchestration and small reproducible artifacts."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from bike_datasets import NUMERIC_FEATURES, prepare_bike_datasets
from bike_evaluation import importance_agreement, prediction_agreement, regression_metrics
from bike_explanations import additive_group_importance, explain_bike_model, explanation_indices, temperature_attribution_summary
from bike_training_process import train_bike_model


def _versions():
    result = dict(python=platform.python_version(), platform=platform.platform())
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "interpret-core", "shap", "joblib", "threadpoolctl"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _comparison_kind(first, second):
    same_variant = first["feature_variant"] == second["feature_variant"]
    same_model = first["model_name"] == second["model_name"]
    same_seed = first["seed"] == second["seed"]
    if same_variant and same_model and not same_seed:
        return "seed"
    if same_variant and not same_model and same_seed:
        return "model"
    if not same_variant and same_model and same_seed:
        return "feature_representation"
    return None


def _comparisons(records, predictions, targets):
    comparisons = []
    for first, second in combinations(records, 2):
        kind = _comparison_kind(first, second)
        if kind is None:
            continue
        result = dict(first=first["record_id"], second=second["record_id"], kind=kind,
                      both_near_optimal_on_validation=first["near_optimal_on_validation"] and second["near_optimal_on_validation"],
                      prediction_agreement=prediction_agreement(predictions[first["record_id"]],
                                                                predictions[second["record_id"]], targets))
        first_panels, second_panels = first["explanations"], second["explanations"]
        if first_panels["row_ids"] != second_panels["row_ids"]:
            raise ValueError("Explanation cohorts must be identical")
        result["grouped_importance_agreement"] = importance_agreement(first_panels["grouped"]["panels"], second_panels["grouped"]["panels"])
        if kind != "feature_representation":
            result["individual_importance_agreement"] = importance_agreement(first_panels["individual"]["panels"], second_panels["individual"]["panels"])
        if "shap" in first_panels and "shap" in second_panels:
            first_shap, second_shap = first_panels["shap"], second_panels["shap"]
            if first_shap["row_ids"] != second_shap["row_ids"]:
                raise ValueError("Local SHAP comparisons require identical cohorts")
            result["shap_group_contribution_agreement"] = {}
            for name, values in first_shap["group_contributions"].items():
                difference = np.asarray(second_shap["group_contributions"][name]) - np.asarray(values)
                result["shap_group_contribution_agreement"][name] = dict(
                    mean_absolute_difference=float(np.abs(difference).mean()),
                    rmse_difference=float(np.sqrt(np.mean(difference**2))),
                    mean_second_minus_first=float(difference.mean()))
        comparisons.append(result)
    return comparisons


def run_bike_experiment(cfg, data_cfg, *, show_progress=True):
    """Run all predeclared arms; never download data or select using TEST."""
    prepared = {variant: prepare_bike_datasets(data_cfg, variant) for variant in cfg.feature_variants}
    reference = prepared[cfg.feature_variants[0]]["datasets"]
    for source in prepared.values():
        for split, dataset in source["datasets"].items():
            if not np.array_equal(dataset.row_ids, reference[split].row_ids):
                raise ValueError("Feature variants must use identical chronological cohorts")
    output_dir = Path(cfg.diagnostics_dir) / cfg.run_label
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="started", started_utc=datetime.now(timezone.utc).isoformat(),
                    settings=asdict(cfg), data_settings=asdict(data_cfg), versions=_versions(),
                    data={variant: source["metadata"] for variant, source in prepared.items()},
                    selection=("Fixed candidates where configured; otherwise minimum chronological validation RMSE; no TRAIN+VAL refit"
                               if cfg.fixed_candidate_indices else
                               "Minimum chronological validation RMSE per model/seed/variant; no TRAIN+VAL refit"),
                    near_optimal_definition="Within configured relative RMSE of the best selected validation model, separately per variant",
                    limitations=["One city and two years; hourly rows are not independent observations",
                                 "Marginal permutation preserves dependence within groups, not between groups",
                                 "Agreement between rankings of different dimensions is descriptive, not directly calibrated",
                                 "Stable model explanations are not necessarily faithful or causal",
                                 "TEST is for one final report; repeated report-driven tuning consumes the holdout"])
    _write_json(output_dir / "manifest.json", manifest)
    records, models, predictions = [], {}, {}
    try:
        with threadpool_limits(limits=cfg.n_jobs if cfg.n_jobs > 0 else None):
            for variant, source in prepared.items():
                datasets = source["datasets"]
                train, validation, test = (datasets[name] for name in ("train", "validation", "test"))
                for model_name in cfg.model_names:
                    for seed in cfg.seeds:
                        record_id = f"{variant}_{model_name}_s{seed}"
                        if show_progress:
                            print(f"[{cfg.run_label}] {record_id}", flush=True)
                        fitted = train_bike_model(cfg, model_name, train, validation, seed)
                        model = fitted.pop("model")
                        record = dict(record_id=record_id, feature_variant=variant, **fitted)
                        predictions[record_id] = np.asarray(model.predict(test.inputs), dtype=np.float64)
                        record["results"] = regression_metrics(test.targets, predictions[record_id])
                        record["explanations"] = explain_bike_model(model, train, test, source["metadata"]["feature_groups"], cfg)
                        record["temperature_attribution"] = temperature_attribution_summary(record["explanations"])
                        if model_name == "ebm":
                            record["explanations"]["additive_group_importance"] = additive_group_importance(
                                model, test, explanation_indices(test, cfg), source["metadata"]["feature_groups"])
                        records.append(record)
                        models[record_id] = model
                        if cfg.save_models:
                            joblib.dump(model, output_dir / f"{record_id}.joblib", compress=3)
                        _write_json(output_dir / f"{record_id}.json", record)
        for variant in cfg.feature_variants:
            cohort = [record for record in records if record["feature_variant"] == variant]
            best_rmse = min(record["results_validation"]["rmse"] for record in cohort)
            cutoff = best_rmse * (1 + cfg.near_optimal_relative_rmse)
            for record in cohort:
                record["near_optimal_validation_rmse_cutoff"] = cutoff
                record["near_optimal_on_validation"] = record["results_validation"]["rmse"] <= cutoff
                _write_json(output_dir / f"{record['record_id']}.json", record)
        numeric = [name for name in NUMERIC_FEATURES if name in reference["train"].feature_names]
        correlations = reference["train"].inputs[numeric].corr().replace({np.nan: None}).to_dict()
        report = dict(**manifest, records=records,
                      comparisons=_comparisons(records, predictions, reference["test"].targets),
                      train_numeric_correlations=correlations)
        report.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output_dir / "report.json", report)
        np.savez_compressed(output_dir / "test_predictions.npz", row_ids=reference["test"].row_ids,
                            timestamps=reference["test"].timestamps.to_numpy(dtype="datetime64[ns]"),
                            targets=reference["test"].targets, **predictions)
        lines = [f"Bike explanation experiment: {cfg.run_label}", report["selection"],
                 "TEST: retrospective hourly demand, unmodified regression predictions",
                 "variant/model/seed | VAL RMSE | TEST RMSE | TEST MAE | near-optimal VAL"]
        for record in records:
            lines.append(f"{record['record_id']} | {record['results_validation']['rmse']:.3f} | "
                         f"{record['results']['rmse']:.3f} | {record['results']['mae']:.3f} | "
                         f"{record['near_optimal_on_validation']}")
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.update(status="completed", finished_utc=report["finished_utc"])
        _write_json(output_dir / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write_json(output_dir / "manifest.json", manifest)
        raise
    return dict(report=report, models=models, predictions=predictions,
                diagnostic_files=dict(report=str(output_dir / "report.json"),
                                      summary=str(output_dir / "summary.txt"),
                                      predictions=str(output_dir / "test_predictions.npz")))