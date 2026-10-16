"""Standalone configuration for the first bike explanation experiment."""
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("spline_ridge", "random_forest", "gradient_boosting", "ebm")
FEATURE_VARIANTS = ("full", "temperature_only", "apparent_temperature_only")


@dataclass
class BikeDataConfig:
    data_dir: str = str(STUDY_ROOT / "data")
    download_url: str = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    train_end: str = "2012-01-01"
    validation_end: str = "2012-07-01"

    def __post_init__(self):
        from datetime import date
        if date.fromisoformat(self.train_end) >= date.fromisoformat(self.validation_end):
            raise ValueError("train_end must precede validation_end")
        if not self.download_url.startswith("https://"):
            raise ValueError("download_url must use HTTPS")


@dataclass
class BikeTrainingConfig:
    model_names: tuple = MODEL_NAMES
    feature_variants: tuple = ("full", "temperature_only")
    seeds: tuple = (42, 43, 44)
    fixed_candidate_indices: dict = field(default_factory=dict)
    n_jobs: int = 1
    near_optimal_relative_rmse: float = 0.05
    permutation_repeats: int = 10
    explanation_rows: int = 512
    explanation_seed: int = 1729
    compute_shap: bool = True
    shap_rows: int = 32
    shap_background_rows: int = 64
    shap_permutation_cycles: int = 3
    diagnostics_dir: str = str(STUDY_ROOT / "results")
    run_label: str = "be0_correlated_features"
    save_models: bool = False

    def __post_init__(self):
        for name, values, allowed in (("model_names", self.model_names, MODEL_NAMES),
                                      ("feature_variants", self.feature_variants, FEATURE_VARIANTS)):
            if not values or len(set(values)) != len(values) or any(value not in allowed for value in values):
                raise ValueError(f"{name} must contain unique supported values")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32
               for seed in self.seeds):
            raise ValueError("seeds must be unsigned 32-bit integers")
        if not isinstance(self.fixed_candidate_indices, dict):
            raise ValueError("fixed_candidate_indices must be a dictionary")
        for name, index in self.fixed_candidate_indices.items():
            if name not in self.model_names or not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("Fixed candidates must name configured models and nonnegative integer indices")
        for name in ("permutation_repeats", "explanation_rows", "shap_rows",
                     "shap_background_rows", "shap_permutation_cycles"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.n_jobs, int) or isinstance(self.n_jobs, bool) or self.n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        if not isinstance(self.explanation_seed, int) or isinstance(self.explanation_seed, bool) or not 0 <= self.explanation_seed < 2**32:
            raise ValueError("explanation_seed must be an unsigned 32-bit integer")
        if not isfinite(self.near_optimal_relative_rmse) or self.near_optimal_relative_rmse < 0:
            raise ValueError("near_optimal_relative_rmse must be finite and nonnegative")
        if not self.run_label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                     for character in self.run_label):
            raise ValueError("run_label must contain only letters, digits, underscores and hyphens")


@dataclass
class BikeRobustnessConfig:
    diagnostics_dir: str = str(STUDY_ROOT / "results")
    source_run_label: str = "be1_temperature_representation"
    run_label: str = "be2_explanation_robustness"
    background_seeds: tuple = (1729, 2718, 3141)
    shap_seeds: tuple = (1729, 2718, 3141)
    permutation_cycles: tuple = (10, 30)
    background_rows: tuple = (128, 256)
    block_days: tuple = (1, 7)
    bootstrap_repeats: int = 1000
    bootstrap_seed: int = 2027
    confidence_level: float = 0.95

    def __post_init__(self):
        for label in (self.source_run_label, self.run_label):
            BikeTrainingConfig(run_label=label)
        if self.source_run_label == self.run_label:
            raise ValueError("Robustness output must differ from its source run")
        for name in ("background_seeds", "shap_seeds", "permutation_cycles", "background_rows", "block_days"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            minimum = 0 if name.endswith("seeds") else 1
            if any(not isinstance(value, int) or isinstance(value, bool) or value < minimum
                   or (name.endswith("seeds") and value >= 2**32) for value in values):
                raise ValueError(f"Invalid {name}")
        if not isinstance(self.bootstrap_repeats, int) or isinstance(self.bootstrap_repeats, bool) or self.bootstrap_repeats < 2:
            raise ValueError("bootstrap_repeats must be an integer of at least two")
        if not isinstance(self.bootstrap_seed, int) or isinstance(self.bootstrap_seed, bool) or not 0 <= self.bootstrap_seed < 2**32:
            raise ValueError("bootstrap_seed must be an unsigned 32-bit integer")
        if not isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")