"""Command-line entry point for the finalized exploratory bike study."""
import argparse
import json
from pathlib import Path
import unittest

from bike_config import BikeDataConfig, BikeRobustnessConfig, BikeTrainingConfig, STUDY_ROOT


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=STUDY_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=STUDY_ROOT / "results")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Explicitly download the public hourly CSV; never overwrite")
    commands.add_parser("test", help="Run synthetic preflight tests without downloading data")
    be1 = commands.add_parser("be1", help="Run the finalized six fixed-candidate representation fits")
    be1.add_argument("--run-label", default="be1_temperature_representation")
    be1.add_argument("--n-jobs", type=int, default=1)
    be2 = commands.add_parser("be2", help="Run robustness from a completed BE1 report")
    be2.add_argument("--source-run-label", default="be1_temperature_representation")
    be2.add_argument("--run-label", default="be2_explanation_robustness")
    be2.add_argument("--rebuild-models", action="store_true",
                     help="Explicitly reconstruct saved BE1 fits on TRAIN and verify saved predictions")
    combined = commands.add_parser("all", help="Run BE1 then BE2 with live model reuse; no automatic download")
    combined.add_argument("--be1-run-label", default="be1_temperature_representation")
    combined.add_argument("--be2-run-label", default="be2_explanation_robustness")
    combined.add_argument("--n-jobs", type=int, default=1)
    return parser


def _run_preflight():
    suite = unittest.defaultTestLoader.discover(str(STUDY_ROOT / "tests"))
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def _be1_config(arguments):
    label = arguments.be1_run_label if arguments.command == "all" else arguments.run_label
    return BikeTrainingConfig(
        diagnostics_dir=str(arguments.results_dir), run_label=label,
        model_names=("random_forest", "gradient_boosting"),
        feature_variants=("full", "temperature_only", "apparent_temperature_only"),
        fixed_candidate_indices={"random_forest": 1, "gradient_boosting": 1}, seeds=(42,),
        n_jobs=arguments.n_jobs, permutation_repeats=20, explanation_rows=1024,
        compute_shap=True, shap_rows=64, shap_background_rows=128,
        shap_permutation_cycles=10, save_models=False,
    )


def _check_run_directories(*configs):
    for cfg in configs:
        folder = Path(cfg.diagnostics_dir) / cfg.run_label
        if folder.exists():
            raise FileExistsError(f"Output already exists; choose a fresh run label: {folder}")


def _print_artifacts(result):
    print(json.dumps(result["diagnostic_files"], indent=2))


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "be2" and not arguments.rebuild_models:
        parser.error("be2 requires --rebuild-models; use all or the notebook to reuse live models")
    if arguments.command == "test":
        return 0 if _run_preflight() else 1
    try:
        data_cfg = BikeDataConfig(data_dir=str(arguments.data_dir))
        if arguments.command == "download":
            from bike_datasets import download_bike_data
            print(json.dumps(download_bike_data(data_cfg), indent=2))
            return 0
        if arguments.command == "be2":
            cfg = BikeRobustnessConfig(diagnostics_dir=str(arguments.results_dir),
                source_run_label=arguments.source_run_label, run_label=arguments.run_label)
            _check_run_directories(cfg)
            if not _run_preflight():
                return 1
            from bike_robustness_pipeline import run_bike_robustness
            _print_artifacts(run_bike_robustness(cfg, data_cfg, rebuild_models=True))
            return 0
        training_cfg = _be1_config(arguments)
        if arguments.command == "all":
            robustness_cfg = BikeRobustnessConfig(diagnostics_dir=str(arguments.results_dir),
                source_run_label=training_cfg.run_label, run_label=arguments.be2_run_label)
            _check_run_directories(training_cfg, robustness_cfg)
        else:
            _check_run_directories(training_cfg)
        if not _run_preflight():
            return 1
        from bike_training_pipeline import run_bike_experiment
        trained = run_bike_experiment(training_cfg, data_cfg)
        _print_artifacts(trained)
        if arguments.command == "all":
            from bike_robustness_pipeline import run_bike_robustness
            _print_artifacts(run_bike_robustness(robustness_cfg, data_cfg,
                                              models=trained["models"], rebuild_models=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())