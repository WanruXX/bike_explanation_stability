"""Small remote preflight checks; no downloads or experiment runs."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from bike_config import BikeDataConfig, BikeRobustnessConfig, BikeTrainingConfig
from bike_datasets import BikeDataset, FEATURE_NAMES, feature_groups, prepare_bike_datasets
from bike_evaluation import block_bootstrap_means, importance_agreement, paired_prediction_block_intervals, regression_metrics
from bike_explanations import additive_group_importance, explain_shap_model, group_shap_contributions, permutation_importance, shap_explanation_indices, temperature_attribution_summary
from bike_models import build_bike_model, model_candidates
from bike_robustness_pipeline import robustness_cases, run_bike_robustness
from bike_training_process import train_bike_model
from main import build_parser, main


def _frame():
    dates = ["2011-12-31", "2011-12-31", "2012-01-01", "2012-01-01", "2012-07-01", "2012-07-01"]
    count = len(dates)
    frame = pd.DataFrame({name: np.ones(count) for name in FEATURE_NAMES})
    frame["instant"] = np.arange(1, count + 1)
    frame["dteday"] = dates
    frame["hr"] = [0, 1, 0, 1, 0, 1]
    frame["temp"] = np.arange(count, dtype=float)
    frame["atemp"] = 2 * frame["temp"]
    frame["casual"], frame["registered"] = 1, 2
    frame["cnt"] = 3
    frame["timestamp"] = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
    return frame


class BikeCoreTests(unittest.TestCase):
    def test_cli_be1_uses_finalized_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            result = dict(models={}, diagnostic_files={})
            with patch("main._run_preflight", return_value=True), patch("main._print_artifacts"), \
                    patch("bike_training_pipeline.run_bike_experiment", return_value=result) as run:
                self.assertEqual(main(["--results-dir", folder, "be1", "--run-label", "cli_be1"]), 0)
            cfg = run.call_args.args[0]
            self.assertEqual(cfg.model_names, ("random_forest", "gradient_boosting"))
            self.assertEqual(cfg.feature_variants, ("full", "temperature_only", "apparent_temperature_only"))
            self.assertEqual(cfg.fixed_candidate_indices, {"random_forest": 1, "gradient_boosting": 1})
            self.assertEqual(cfg.seeds, (42,))
            self.assertEqual((cfg.explanation_rows, cfg.permutation_repeats, cfg.shap_rows,
                              cfg.shap_background_rows, cfg.shap_permutation_cycles), (1024, 20, 64, 128, 10))
            self.assertTrue(cfg.compute_shap)
            self.assertFalse(cfg.save_models)

    def test_cli_combined_reuses_models_and_checks_both_output_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            models = {"model": object()}
            with patch("main._run_preflight", return_value=True), patch("main._print_artifacts"), \
                    patch("bike_training_pipeline.run_bike_experiment", return_value=dict(models=models)) as train, \
                    patch("bike_robustness_pipeline.run_bike_robustness", return_value={}) as robustness:
                arguments = ["--results-dir", folder, "all", "--be1-run-label", "cli_be1", "--be2-run-label", "cli_be2"]
                self.assertEqual(main(arguments), 0)
                self.assertIs(robustness.call_args.kwargs["models"], models)
                self.assertFalse(robustness.call_args.kwargs["rebuild_models"])
                self.assertEqual(robustness.call_args.args[0].source_run_label, "cli_be1")
                train.reset_mock()
                Path(folder, "cli_be2").mkdir()
                with patch("sys.stderr"), self.assertRaises(SystemExit) as stopped:
                    main(arguments)
                self.assertEqual(stopped.exception.code, 1)
                train.assert_not_called()

    def test_cli_preflight_failure_stops_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("main._run_preflight", return_value=False), \
                    patch("bike_training_pipeline.run_bike_experiment") as train:
                self.assertEqual(main(["--results-dir", folder, "be1", "--run-label", "failed_preflight"]), 1)
                train.assert_not_called()
                self.assertFalse(Path(folder, "failed_preflight").exists())

    def test_cli_be2_requires_explicit_reconstruction(self):
        arguments = build_parser().parse_args(["be2"])
        self.assertFalse(arguments.rebuild_models)
        with patch("sys.stderr"), patch("main._run_preflight") as preflight, self.assertRaises(SystemExit) as stopped:
            main(["be2"])
        self.assertEqual(stopped.exception.code, 2)
        preflight.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch("main._run_preflight", return_value=True), patch("main._print_artifacts"), \
                    patch("bike_robustness_pipeline.run_bike_robustness", return_value={}) as robustness:
                self.assertEqual(main(["--results-dir", folder, "be2", "--run-label", "cli_be2", "--rebuild-models"]), 0)
                self.assertTrue(robustness.call_args.kwargs["rebuild_models"])

    def test_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            BikeTrainingConfig(run_label="../outside")
        with self.assertRaises(ValueError):
            BikeTrainingConfig(seeds=(42, 42))
        with self.assertRaises(ValueError):
            BikeDataConfig(train_end="2012-07-01", validation_end="2012-01-01")

    def test_chronological_boundaries_and_feature_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            _frame().drop(columns="timestamp").to_csv(f"{folder}/hour.csv", index=False)
            cfg = BikeDataConfig(data_dir=folder)
            full = prepare_bike_datasets(cfg)
            reduced = prepare_bike_datasets(cfg, "temperature_only")
            apparent = prepare_bike_datasets(cfg, "apparent_temperature_only")
            for split, expected in (("train", [1, 2]), ("validation", [3, 4]), ("test", [5, 6])):
                np.testing.assert_array_equal(full["datasets"][split].row_ids, expected)
                np.testing.assert_array_equal(full["datasets"][split].row_ids, reduced["datasets"][split].row_ids)
                np.testing.assert_array_equal(full["datasets"][split].row_ids, apparent["datasets"][split].row_ids)
            self.assertNotIn("atemp", reduced["datasets"]["train"].inputs)
            self.assertNotIn("temp", apparent["datasets"]["train"].inputs)
            self.assertIn("atemp", apparent["datasets"]["train"].inputs)
            self.assertEqual(set(full["datasets"]["train"].feature_names) - {"temp"},
                             set(apparent["datasets"]["train"].feature_names))
            self.assertFalse({"cnt", "casual", "registered", "instant"} & set(full["metadata"]["feature_names"]))
            groups = feature_groups(full["metadata"]["feature_names"])
            self.assertEqual(set(sum(groups.values(), ())), set(FEATURE_NAMES))

    def test_duplicate_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _frame().drop(columns="timestamp")
            frame.loc[1, "hr"] = 0
            frame.to_csv(f"{folder}/hour.csv", index=False)
            with self.assertRaises(ValueError):
                prepare_bike_datasets(BikeDataConfig(data_dir=folder))

    def test_shap_sampling_is_reproducible_not_a_chronological_prefix(self):
        indices = np.arange(1000)
        cfg = BikeTrainingConfig(shap_rows=32)
        selected = shap_explanation_indices(indices, cfg)
        np.testing.assert_array_equal(selected, shap_explanation_indices(indices, cfg))
        self.assertEqual(len(np.unique(selected)), 32)
        self.assertTrue(set(selected).issubset(indices))
        self.assertFalse(np.array_equal(selected, indices[:32]))
        self.assertGreater(selected.max(), 500)
        self.assertTrue(np.all(np.diff(selected) > 0))
        np.testing.assert_array_equal(shap_explanation_indices(indices[:3], cfg), indices[:3])

    def test_joint_permutation_preserves_internal_relationship(self):
        frame = _frame()
        frame["cnt"] = frame["temp"]
        dataset = BikeDataset(frame, FEATURE_NAMES, "test")

        class Predictor:
            def predict(self, inputs):
                np.testing.assert_allclose(inputs["atemp"], 2 * inputs["temp"])
                return inputs["temp"].to_numpy()

        cfg = BikeTrainingConfig(permutation_repeats=4)
        panel = permutation_importance(Predictor(), dataset, np.arange(len(dataset)),
                                       {"temperature": ("temp", "atemp")}, cfg, retain_row_losses=True)
        self.assertEqual(panel["baseline_mse"], 0)
        self.assertGreater(panel["panels"]["temperature"]["mean_mse_increase"], 0)
        self.assertAlmostEqual(np.mean(panel["panels"]["temperature"]["row_mean_mse_increase"]),
                               panel["panels"]["temperature"]["mean_mse_increase"])

    def test_shap_controls_do_not_move_explicit_cohort(self):
        dataset = BikeDataset(_frame(), FEATURE_NAMES, "test")
        cfg = BikeTrainingConfig(shap_background_rows=3)
        cohort = np.array([1, 4])
        captured = []

        class FakeExplainer:
            def __init__(self, predict, masker, algorithm, seed):
                self.seed = seed

            def __call__(self, inputs, max_evals):
                captured.append(dict(inputs=inputs.copy(), seed=self.seed, budget=max_evals))
                return SimpleNamespace(values=np.zeros_like(inputs), base_values=np.zeros(len(inputs)))

        module = SimpleNamespace(maskers=SimpleNamespace(Independent=Mock(return_value=object())),
                                 Explainer=FakeExplainer)
        model = Mock()
        model.predict.side_effect = lambda inputs: np.zeros(len(inputs))
        with patch.dict(sys.modules, {"shap": module}):
            first = explain_shap_model(model, dataset, dataset, feature_groups(FEATURE_NAMES), cfg,
                                       cohort, background_seed=1, shap_seed=2, permutation_cycles=10)
            second = explain_shap_model(model, dataset, dataset, feature_groups(FEATURE_NAMES), cfg,
                                        cohort, background_seed=3, shap_seed=4, permutation_cycles=30)
        self.assertEqual(first["row_ids"], second["row_ids"])
        self.assertEqual(first["row_ids"], [2, 5])
        np.testing.assert_array_equal(captured[0]["inputs"], captured[1]["inputs"])
        self.assertEqual([item["seed"] for item in captured], [2, 4])
        self.assertEqual(captured[1]["budget"], 3 * captured[0]["budget"])
        self.assertEqual((first["background_seed"], second["background_seed"]), (1, 3))
        model.fit.assert_not_called()
        with patch.dict(sys.modules, {"shap": module}):
            with self.assertRaises(ValueError):
                explain_shap_model(model, dataset, dataset, feature_groups(FEATURE_NAMES), cfg, [1.5, 4])
            with self.assertRaises(ValueError):
                explain_shap_model(model, dataset, dataset, feature_groups(FEATURE_NAMES), cfg, cohort, permutation_cycles=0)

    def test_calendar_block_pairing_and_reproducibility(self):
        times = pd.date_range("2012-07-01", periods=72, freq="h")
        values = np.repeat([1.0, 2.0, 5.0], 24)
        rows = np.column_stack((values, values))
        first = block_bootstrap_means(rows, times, block_days=1, repeats=50, seed=2027)
        second = block_bootstrap_means(rows, times, block_days=1, repeats=50, seed=2027)
        np.testing.assert_array_equal(first["draws"], second["draws"])
        np.testing.assert_array_equal(first["draws"][:, 0], first["draws"][:, 1])
        self.assertEqual(first["blocks"], 3)
        result = paired_prediction_block_intervals(values, values, np.zeros(72), times,
                                                  block_days=1, repeats=50, seed=2027, confidence_level=0.95)
        for metric in result["metrics"].values():
            self.assertEqual(metric["estimate"], 0)
            self.assertEqual(metric["lower"], 0)
            self.assertEqual(metric["upper"], 0)
        with self.assertRaisesRegex(ValueError, "two observed calendar blocks"):
            block_bootstrap_means(rows, times, block_days=7, repeats=50, seed=2027)
        with self.assertRaises(ValueError):
            BikeRobustnessConfig(run_label="be1_temperature_representation")
        with self.assertRaises(ValueError):
            BikeRobustnessConfig(shap_seeds=(1, 1))

    def test_robustness_reuses_saved_cohort_and_requires_explicit_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _frame()
            frame.loc[5, "dteday"] = "2012-07-09"
            frame.drop(columns="timestamp").to_csv(Path(folder) / "hour.csv", index=False)
            data_cfg = BikeDataConfig(data_dir=folder)
            prepared = prepare_bike_datasets(data_cfg)
            train, test = (prepared["datasets"][split] for split in ("train", "test"))
            source_cfg = BikeTrainingConfig(diagnostics_dir=folder, run_label="be1_temperature_representation",
                model_names=("random_forest",), feature_variants=("full",), seeds=(42,),
                fixed_candidate_indices={"random_forest": 1}, explanation_rows=2, shap_rows=2, shap_background_rows=2)
            cfg = BikeRobustnessConfig(diagnostics_dir=folder, background_seeds=(1729,), shap_seeds=(1729, 2718),
                                       permutation_cycles=(3,), background_rows=(2,), bootstrap_repeats=20)
            baseline = dict(row_ids=test.row_ids.tolist(), timestamps=test.timestamps.astype(str).tolist(),
                feature_names=list(test.feature_names), background_row_ids=train.row_ids.tolist(),
                values=np.zeros((len(test), len(test.feature_names))).tolist(), base_values=[0.0, 0.0],
                mean_absolute_contribution=np.zeros(len(test.feature_names)).tolist(),
                group_contributions={name: [0.0, 0.0] for name in prepared["metadata"]["feature_groups"]},
                mean_absolute_group_contribution={name: 0.0 for name in prepared["metadata"]["feature_groups"]},
                reconstruction_max_abs_error=0.0)
            record_id = "full_random_forest_s42"
            record = dict(record_id=record_id, feature_variant="full", model_name="random_forest", seed=42,
                candidate_policy="fixed_before_run", refit_on_validation=False, best_candidate_index=1,
                settings=model_candidates("random_forest")[1], explanations=dict(row_ids=test.row_ids.tolist(),
                    shap=baseline, individual=dict(panels={name: dict(mean_mse_increase=0.0) for name in ("temp", "atemp")}),
                    grouped=dict(panels={name: dict(mean_mse_increase=0.0) for name in baseline["group_contributions"]})))
            source = dict(status="completed", settings=asdict(source_cfg), data_settings=asdict(data_cfg),
                          data=dict(full=prepared["metadata"]), records=[record], versions={})
            source_dir = Path(folder) / cfg.source_run_label
            source_dir.mkdir()
            (source_dir / "report.json").write_text(json.dumps(source), encoding="utf-8")
            np.savez_compressed(source_dir / "test_predictions.npz", row_ids=test.row_ids,
                timestamps=test.timestamps.to_numpy(dtype="datetime64[ns]"), targets=test.targets,
                **{record_id: np.zeros(len(test))})
            model = Mock()
            model.predict.side_effect = lambda inputs: np.zeros(len(inputs))
            cases = robustness_cases(cfg, source_cfg)
            self.assertEqual([case["case_id"] for case in cases], ["baseline", "shap_seed_2718"])
            for case in cases[1:]:
                changed = [name for name in ("background_seed", "shap_seed", "permutation_cycles", "background_rows")
                           if case[name] != cases[0][name]]
                self.assertEqual(changed, [case["axis"]])
            with patch("bike_robustness_pipeline.explain_shap_model", return_value=baseline) as explain:
                result = run_bike_robustness(cfg, data_cfg, models={record_id: model}, show_progress=False)
                model.fit.assert_not_called()
                self.assertEqual(result["report"]["model_modes"][record_id], "live_model_reuse")
                self.assertEqual(len(result["report"]["shap_summaries"]), 2)
                np.testing.assert_array_equal(explain.call_args.args[5], [0, 1])
                self.assertEqual(explain.call_args.kwargs["shap_seed"], 2718)
                with self.assertRaises(FileExistsError):
                    run_bike_robustness(cfg, data_cfg, models={record_id: model}, show_progress=False)
                cfg.run_label = "missing_model"
                with self.assertRaisesRegex(ValueError, "Missing live model"):
                    run_bike_robustness(cfg, data_cfg, show_progress=False)
                failed = json.loads((Path(folder) / cfg.run_label / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(failed["status"], "failed")
                cfg.run_label = "reconstructed_model"
                with patch("bike_robustness_pipeline.build_bike_model", return_value=model) as build:
                    result = run_bike_robustness(cfg, data_cfg, rebuild_models=True, show_progress=False)
                    model.fit.assert_called_once()
                    build.assert_called_once()
                    self.assertEqual(build.call_args.args[2], record["settings"])
                    self.assertEqual(result["report"]["model_modes"][record_id], "explicit_saved_settings_reconstruction")

    def test_validation_selects_without_refit(self):
        frame = _frame()
        train = BikeDataset(frame.iloc[:2], FEATURE_NAMES, "train")
        validation = BikeDataset(frame.iloc[2:4], FEATURE_NAMES, "validation")
        fitted_ids = []

        class Predictor:
            def __init__(self, value):
                self.value = value

            def fit(self, inputs, targets):
                fitted_ids.append(inputs["temp"].tolist())
                return self

            def predict(self, inputs):
                return np.full(len(inputs), self.value)

        with patch("bike_training_process.model_candidates", return_value=[{}, {}]), \
             patch("bike_training_process.build_bike_model", side_effect=[Predictor(0), Predictor(3)]):
            result = train_bike_model(BikeTrainingConfig(), "spline_ridge", train, validation, 42)
        self.assertEqual(result["best_candidate_index"], 1)
        self.assertEqual(fitted_ids, [[0.0, 1.0], [0.0, 1.0]])
        self.assertFalse(result["refit_on_validation"])

    def test_fixed_candidate_does_not_search_or_refit(self):
        frame = _frame()
        train = BikeDataset(frame.iloc[:2], FEATURE_NAMES, "train")
        validation = BikeDataset(frame.iloc[2:4], FEATURE_NAMES, "validation")
        from unittest.mock import Mock
        model = Mock()
        model.predict.side_effect = lambda inputs: np.full(len(inputs), 3)
        cfg = BikeTrainingConfig(fixed_candidate_indices={"spline_ridge": 1})
        with patch("bike_training_process.model_candidates", return_value=[{}, {"alpha": 10}]), \
             patch("bike_training_process.build_bike_model", return_value=model) as factory:
            result = train_bike_model(cfg, "spline_ridge", train, validation, 42)
        factory.assert_called_once_with("spline_ridge", FEATURE_NAMES, {"alpha": 10}, cfg, 42)
        model.fit.assert_called_once()
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["best_candidate_index"], 1)
        self.assertEqual(result["candidate_policy"], "fixed_before_run")
        with self.assertRaises(ValueError):
            BikeTrainingConfig(fixed_candidate_indices={"unknown": 0})
        cfg.fixed_candidate_indices["spline_ridge"] = 100
        with self.assertRaisesRegex(ValueError, "out of range"):
            train_bike_model(cfg, "spline_ridge", train, validation, 42)

    def test_temperature_summaries_preserve_cancellation_and_missing_features(self):
        values = group_shap_contributions([[1, -1], [-2, 2]], ("temp", "atemp"),
                                         {"temperature": ("temp", "atemp")})
        self.assertEqual(values["temperature"], [0, 0])
        explanations = dict(
            individual=dict(panels={"temp": dict(mean_mse_increase=-2), "atemp": dict(mean_mse_increase=6)}),
            grouped=dict(panels={"temperature": dict(mean_mse_increase=10)}),
            shap=dict(feature_names=["temp", "atemp"], mean_absolute_contribution=[1.5, 1.5],
                      mean_absolute_group_contribution={"temperature": 0}))
        result = temperature_attribution_summary(explanations)
        self.assertEqual(result["individual_absolute_share"]["temp"], 0.25)
        self.assertEqual(result["shap"]["joint_to_sum_absolute_ratio"], 0)
        del explanations["individual"]["panels"]["temp"]
        del explanations["shap"]
        result = temperature_attribution_summary(explanations)
        self.assertIsNone(result["individual_absolute_share"]["temp"])
        self.assertEqual(result["individual_absolute_share"]["atemp"], 1)

    def test_additive_groups_use_term_api_and_sum_before_absolute_value(self):
        dataset = BikeDataset(_frame(), FEATURE_NAMES, "test")
        indices = np.array([1, 4])

        class Predictor:
            intercept_ = 10.0
            term_features_ = ((FEATURE_NAMES.index("temp"),), (FEATURE_NAMES.index("atemp"),))

            def eval_terms(self, inputs):
                values = inputs["temp"].to_numpy() - 2.5
                return np.column_stack((values, -values))

            def predict(self, inputs):
                return self.intercept_ + self.eval_terms(inputs).sum(axis=1)

        model = Predictor()
        groups = {"temperature": ("temp", "atemp"), "temp": ("temp",)}
        result = additive_group_importance(model, dataset, indices, groups)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["panels"]["temperature"]["mean_absolute_joint_contribution"], 0)
        self.assertEqual(result["panels"]["temp"]["mean_absolute_joint_contribution"], 1.5)
        with patch.object(model, "predict", return_value=np.full(2, 11.0)):
            with self.assertRaisesRegex(ValueError, "do not reconstruct"):
                additive_group_importance(model, dataset, indices, groups)
        model.term_features_ = ((0, 1),)
        with self.assertRaisesRegex(ValueError, "interaction-free"):
            additive_group_importance(model, dataset, indices, groups)

    def test_real_ebm_contribution_api(self):
        frame = _frame()
        frame["cnt"] = 3 + 10 * frame["temp"]
        dataset = BikeDataset(frame, FEATURE_NAMES, "train")
        model = build_bike_model("ebm", FEATURE_NAMES, model_candidates("ebm")[0],
                                 BikeTrainingConfig(), 42)
        self.assertEqual(model.outer_bags, 1)
        self.assertEqual(model.validation_size, 0)
        model.set_params(max_rounds=8, smoothing_rounds=0)
        model.fit(dataset.inputs, dataset.targets)
        terms = model.eval_terms(dataset.inputs)
        np.testing.assert_allclose(model.predict(dataset.inputs), model.intercept_ + terms.sum(axis=1))
        result = additive_group_importance(model, dataset, np.arange(len(dataset)), feature_groups(FEATURE_NAMES))
        self.assertEqual(result["rows"], len(dataset))
        self.assertTrue(all(np.isfinite(panel["mean_absolute_joint_contribution"])
                            for panel in result["panels"].values()))

    def test_constant_targets_and_tied_importance(self):
        self.assertIsNone(regression_metrics([3, 3], [3, 3])["r2"])
        panel = {"temp": {"mean_mse_increase": 0}, "atemp": {"mean_mse_increase": 0}}
        result = importance_agreement(panel, panel)
        self.assertIsNone(result["spearman"])
        self.assertIsNone(result["signed_normalized_l1"])
        self.assertEqual(result["top_tie_set_jaccard"], 1)