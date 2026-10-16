"""Exploratory fixed-model sensitivity and paired temporal-block summaries."""
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from bike_config import BikeTrainingConfig
from bike_datasets import prepare_bike_datasets
from bike_evaluation import block_bootstrap_means, bootstrap_interval, paired_prediction_block_intervals
from bike_explanations import explain_shap_model, permutation_importance, temperature_attribution_summary
from bike_models import build_bike_model
from bike_training_pipeline import _comparison_kind, _versions, _write_json


def robustness_cases(cfg, source_cfg):
    baseline = dict(background_seed=source_cfg.explanation_seed, shap_seed=source_cfg.explanation_seed,
                    permutation_cycles=source_cfg.shap_permutation_cycles,
                    background_rows=source_cfg.shap_background_rows)
    cases = [dict(case_id="baseline", axis="saved_baseline", **baseline)]
    for axis, values in (("background_seed", cfg.background_seeds), ("shap_seed", cfg.shap_seeds),
                         ("permutation_cycles", cfg.permutation_cycles), ("background_rows", cfg.background_rows)):
        for value in values:
            if value != baseline[axis]:
                controls = dict(baseline)
                controls[axis] = value
                cases.append(dict(case_id=f"{axis}_{value}", axis=axis, **controls))
    return cases


def _cohort_indices(dataset, row_ids):
    ids = np.asarray(row_ids)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Saved cohort IDs must be nonempty and unique")
    lookup = {row_id: position for position, row_id in enumerate(dataset.row_ids.tolist())}
    try:
        indices = np.array([lookup[row_id] for row_id in ids.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Saved explanation row is absent from the current dataset") from error
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Saved cohort must be chronological")
    return indices


def _row_intervals(rows, timestamps, cfg, *, square_root=()):
    names = list(rows)
    values = np.column_stack([rows[name] for name in names])
    results = []
    for days in cfg.block_days:
        bootstrap = block_bootstrap_means(values, timestamps, block_days=days,
                                         repeats=cfg.bootstrap_repeats, seed=cfg.bootstrap_seed)
        metrics = {}
        for position, name in enumerate(names):
            point, draws = bootstrap["point"][position], bootstrap["draws"][:, position]
            if name in square_root:
                point, draws = np.sqrt(point), np.sqrt(draws)
            metrics[name] = bootstrap_interval(point, draws, cfg.confidence_level)
        results.append(dict(block_days=days, blocks=bootstrap["blocks"], rows=bootstrap["rows"],
                            repeats=cfg.bootstrap_repeats, method=bootstrap["method"], metrics=metrics))
    return results


def _local_rows(first, second):
    if first["row_ids"] != second["row_ids"] or first["timestamps"] != second["timestamps"]:
        raise ValueError("SHAP comparisons require identical ordered cohorts")
    if set(first["group_contributions"]) != set(second["group_contributions"]):
        raise ValueError("SHAP group schemas differ")
    rows, roots = {}, []
    for name, first_values in first["group_contributions"].items():
        first_values = np.asarray(first_values)
        second_values = np.asarray(second["group_contributions"][name])
        difference = second_values - first_values
        rows[f"{name}/mae_difference"] = np.abs(difference)
        rows[f"{name}/rmse_difference"] = difference**2
        roots.append(f"{name}/rmse_difference")
        rows[f"{name}/signed_second_minus_first"] = difference
        rows[f"{name}/absolute_magnitude_second_minus_first"] = np.abs(second_values) - np.abs(first_values)
    return rows, roots


def _local_agreement(first, second):
    rows, roots = _local_rows(first, second)
    return {name: float(np.sqrt(np.mean(values)) if name in roots else np.mean(values))
            for name, values in rows.items()}


def run_bike_robustness(cfg, data_cfg, *, models=None, rebuild_models=False, show_progress=True):
    """Reuse BE1 artifacts; reconstruction is explicit and never searches candidates."""
    cfg.__post_init__()
    source_dir = Path(cfg.diagnostics_dir) / cfg.source_run_label
    output_dir = Path(cfg.diagnostics_dir) / cfg.run_label
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh robustness run_label: {output_dir}")
    source = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    if source.get("status") != "completed" or source["settings"]["run_label"] != cfg.source_run_label:
        raise ValueError("A matching completed source report is required")
    source_cfg = BikeTrainingConfig(**source["settings"])
    current_versions = _versions()
    for package in ("numpy", "scipy", "scikit-learn", "shap"):
        saved_version = source["versions"].get(package)
        if saved_version is not None and saved_version != current_versions.get(package):
            raise ValueError(f"Restore BE1's {package} version {saved_version} before changing one explanation control")
    for boundary in ("train_end", "validation_end"):
        if getattr(data_cfg, boundary) != source["data_settings"][boundary]:
            raise ValueError("Source split boundaries must remain fixed")
    records = source["records"]
    if not records or len({record["record_id"] for record in records}) != len(records):
        raise ValueError("Source records must be nonempty and uniquely identified")
    if any("shap" not in record["explanations"] or record["candidate_policy"] != "fixed_before_run"
           or record["refit_on_validation"] for record in records):
        raise ValueError("This follow-up requires saved SHAP and fixed TRAIN-only source fits")
    for record in records:
        if (record["record_id"] != f"{record['feature_variant']}_{record['model_name']}_s{record['seed']}"
                or record["feature_variant"] not in source_cfg.feature_variants
                or record["model_name"] not in source_cfg.model_names or record["seed"] not in source_cfg.seeds
                or record["best_candidate_index"] != source_cfg.fixed_candidate_indices.get(record["model_name"])):
            raise ValueError("Source record identity or fixed candidate differs from source settings")
    prepared = {variant: prepare_bike_datasets(data_cfg, variant)
                for variant in source_cfg.feature_variants}
    reference = prepared[source_cfg.feature_variants[0]]["datasets"]
    for variant, data in prepared.items():
        for field in ("csv_sha256", "feature_names", "feature_groups"):
            current = data["metadata"][field]
            if field == "feature_groups":
                current = {name: list(features) for name, features in current.items()}
            if current != source["data"][variant][field]:
                raise ValueError(f"Source data or schema mismatch: {variant}/{field}")
        for split, dataset in data["datasets"].items():
            np.testing.assert_array_equal(dataset.row_ids, reference[split].row_ids)
            np.testing.assert_array_equal(dataset.timestamps, reference[split].timestamps)
            np.testing.assert_array_equal(dataset.targets, reference[split].targets)
    with np.load(source_dir / "test_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_ids"], reference["test"].row_ids)
        np.testing.assert_array_equal(saved["targets"], reference["test"].targets)
        np.testing.assert_array_equal(saved["timestamps"], reference["test"].timestamps.to_numpy(dtype="datetime64[ns]"))
        predictions = {record["record_id"]: saved[record["record_id"]].copy() for record in records}
    cases = robustness_cases(cfg, source_cfg)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="started", started_utc=datetime.now(timezone.utc).isoformat(),
                    settings=asdict(cfg), data_settings=asdict(data_cfg), source_report=str(source_dir / "report.json"),
                    source_settings=source["settings"], source_versions=source["versions"],
                    versions=current_versions, data={variant: data["metadata"] for variant, data in prepared.items()},
                    cases=cases, rebuild_models_authorized=rebuild_models,
                    limitations=["Exploratory reuse of an inspected TEST period; no confirmatory significance claims",
                                 "Background samples and size change the empirical reference game, not just approximation noise",
                                 "Intervals condition on fitted models and fixed perturbation/reference design; no training uncertainty",
                                 "Non-overlapping calendar blocks assume resampling is meaningful despite temporal shift",
                                 "Sparse fixed SHAP rows cover occupied blocks only, not all hourly outcomes",
                                 "Larger SHAP budgets are comparators, not ground truth; intervals have no multiplicity correction"])
    _write_json(output_dir / "manifest.json", manifest)
    fitted_models = {} if models is None else dict(models)
    shap_results, permutation_results, model_modes = {}, {}, {}
    summaries, within_sensitivity, baseline_intervals = [], [], []
    comparisons = []
    try:
        with threadpool_limits(limits=source_cfg.n_jobs if source_cfg.n_jobs > 0 else None):
            for record in records:
                record_id = record["record_id"]
                datasets = prepared[record["feature_variant"]]["datasets"]
                train, test = datasets["train"], datasets["test"]
                if record_id not in fitted_models:
                    if not rebuild_models:
                        raise ValueError(f"Missing live model {record_id}; explicitly set rebuild_models=True to reconstruct")
                    if show_progress:
                        print(f"[{cfg.run_label}] explicitly reconstructing {record_id} on original TRAIN", flush=True)
                    model = build_bike_model(record["model_name"], train.feature_names, record["settings"],
                                             source_cfg, record["seed"])
                    model.fit(train.inputs, train.targets)
                    fitted_models[record_id] = model
                    model_modes[record_id] = "explicit_saved_settings_reconstruction"
                else:
                    model_modes[record_id] = "live_model_reuse"
                model = fitted_models[record_id]
                actual = np.asarray(model.predict(test.inputs), dtype=np.float64)
                if not np.isfinite(actual).all() or not np.isfinite(predictions[record_id]).all():
                    raise ValueError("Saved and current predictions must be finite")
                np.testing.assert_allclose(actual, predictions[record_id], rtol=1e-9, atol=1e-8,
                                           err_msg=f"Model does not reproduce saved predictions: {record_id}")
                saved_explanations = record["explanations"]
                baseline = saved_explanations["shap"]
                if baseline["feature_names"] != list(test.feature_names):
                    raise ValueError("Saved SHAP feature schema differs")
                indices = _cohort_indices(test, baseline["row_ids"])
                if baseline["timestamps"] != test.timestamps[indices].astype(str).tolist():
                    raise ValueError("Saved SHAP timestamps differ")
                background_indices = np.sort(np.random.default_rng(source_cfg.explanation_seed).choice(
                    len(train), min(source_cfg.shap_background_rows, len(train)), replace=False))
                if baseline["background_row_ids"] != train.row_ids[background_indices].tolist():
                    raise ValueError("Saved SHAP reference cannot be reproduced from source controls")
                np.testing.assert_allclose(np.asarray(baseline["base_values"]) + np.asarray(baseline["values"]).sum(axis=1),
                                           actual[indices], rtol=1e-9, atol=1e-8)
                groups = prepared[record["feature_variant"]]["metadata"]["feature_groups"]
                for name, features in groups.items():
                    positions = [list(test.feature_names).index(feature) for feature in features]
                    np.testing.assert_allclose(np.asarray(baseline["values"])[:, positions].sum(axis=1),
                                               baseline["group_contributions"][name], rtol=1e-9, atol=1e-8)
                shap_results[record_id] = {}
                for case in cases:
                    if show_progress:
                        print(f"[{cfg.run_label}] {record_id}: {case['case_id']}", flush=True)
                    controls = {name: case[name] for name in ("background_seed", "shap_seed", "permutation_cycles", "background_rows")}
                    explanation = baseline if case["case_id"] == "baseline" else explain_shap_model(
                        model, train, test, groups, source_cfg, indices, **controls)
                    if explanation["reconstruction_max_abs_error"] > 1e-6:
                        raise ValueError("SHAP failed prediction reconstruction")
                    shap_results[record_id][case["case_id"]] = explanation
                    temperature = temperature_attribution_summary(dict(saved_explanations, shap=explanation))["shap"]
                    summaries.append(dict(record_id=record_id, **case, temperature=temperature,
                                          mean_absolute_group_contribution=explanation["mean_absolute_group_contribution"],
                                          background_rows_used=len(explanation["background_row_ids"]),
                                          reconstruction_max_abs_error=explanation["reconstruction_max_abs_error"]))
                    _write_json(output_dir / f"{record_id}__{case['case_id']}.json",
                                dict(record_id=record_id, **case, shap=explanation,
                                     saved_baseline_reused=case["case_id"] == "baseline"))
                    if case["case_id"] != "baseline":
                        within_sensitivity.append(dict(record_id=record_id, **case,
                                                       local_group_agreement=_local_agreement(baseline, explanation)))
                pfi_indices = _cohort_indices(test, saved_explanations["row_ids"])
                pfi_groups = dict(groups)
                pfi_groups.update({f"individual/{name}": (name,) for name in ("temp", "atemp") if name in test.feature_names})
                permutation = permutation_importance(model, test, pfi_indices, pfi_groups, source_cfg, retain_row_losses=True)
                for name, panel in permutation["panels"].items():
                    expected = (saved_explanations["individual"]["panels"][name.split("/")[1]]
                                if name.startswith("individual/") else saved_explanations["grouped"]["panels"][name])
                    np.testing.assert_allclose(panel["mean_mse_increase"], expected["mean_mse_increase"], rtol=1e-8, atol=1e-6)
                permutation_results[record_id] = permutation
                _write_json(output_dir / f"{record_id}__permutation_rows.json",
                            dict(record_id=record_id, row_ids=test.row_ids[pfi_indices].tolist(),
                                 timestamps=test.timestamps[pfi_indices].astype(str).tolist(), **permutation))
                baseline_intervals.append(dict(record_id=record_id,
                    permutation=_row_intervals({name: panel["row_mean_mse_increase"] for name, panel in permutation["panels"].items()},
                                               test.timestamps[pfi_indices], cfg),
                    shap_absolute_group_magnitude=_row_intervals({name: np.abs(values) for name, values in baseline["group_contributions"].items()},
                                                                baseline["timestamps"], cfg)))
        for first, second in combinations(records, 2):
            kind = _comparison_kind(first, second)
            if kind is None:
                continue
            first_id, second_id = first["record_id"], second["record_id"]
            result = dict(first=first_id, second=second_id, kind=kind)
            result["prediction_intervals"] = [paired_prediction_block_intervals(
                predictions[first_id], predictions[second_id], reference["test"].targets, reference["test"].timestamps,
                block_days=days, repeats=cfg.bootstrap_repeats, seed=cfg.bootstrap_seed,
                confidence_level=cfg.confidence_level) for days in cfg.block_days]
            first_panels, second_panels = permutation_results[first_id]["panels"], permutation_results[second_id]["panels"]
            if first["explanations"]["row_ids"] != second["explanations"]["row_ids"]:
                raise ValueError("Paired permutation intervals require identical cohorts")
            rows = {name: np.asarray(second_panels[name]["row_mean_mse_increase"]) - panel["row_mean_mse_increase"]
                    for name, panel in first_panels.items() if name in second_panels}
            indices = _cohort_indices(reference["test"], first["explanations"]["row_ids"])
            result["permutation_second_minus_first_intervals"] = _row_intervals(rows, reference["test"].timestamps[indices], cfg)
            result["matched_shap_cases"] = [dict(case_id=case["case_id"],
                local_group_agreement=_local_agreement(shap_results[first_id][case["case_id"]], shap_results[second_id][case["case_id"]]))
                for case in cases]
            rows, roots = _local_rows(shap_results[first_id]["baseline"], shap_results[second_id]["baseline"])
            result["baseline_shap_intervals"] = _row_intervals(rows, shap_results[first_id]["baseline"]["timestamps"], cfg, square_root=roots)
            comparisons.append(result)
        report = dict(manifest, status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      model_modes=model_modes, shap_summaries=summaries, within_model_sensitivity=within_sensitivity,
                      baseline_intervals=baseline_intervals, comparisons=comparisons)
        _write_json(output_dir / "report.json", report)
        lines = [f"Bike robustness: {cfg.run_label}", f"Source: {cfg.source_run_label}",
                 f"{len(records)} fixed models; {len(cases)} matched SHAP cases including saved baseline",
                 "Inspect within-model sensitivity beside matched cross-model/representation differences.",
                 "Intervals are exploratory and conditional; background changes alter the reference game."]
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.update(status="completed", finished_utc=report["finished_utc"], model_modes=model_modes)
        _write_json(output_dir / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error), model_modes=model_modes)
        _write_json(output_dir / "manifest.json", manifest)
        raise
    return dict(report=report, models=fitted_models,
                diagnostic_files=dict(report=str(output_dir / "report.json"), summary=str(output_dir / "summary.txt")))