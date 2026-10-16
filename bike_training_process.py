"""Fit on TRAIN and select hyperparameters on chronological validation only."""
import time

from bike_evaluation import regression_metrics
from bike_models import build_bike_model, model_candidates


def train_bike_model(cfg, model_name, train, validation, seed):
    if train.feature_names != validation.feature_names:
        raise ValueError("TRAIN and validation feature schemas differ")
    if train.timestamps.max() >= validation.timestamps.min():
        raise ValueError("TRAIN must strictly precede validation")
    candidates = model_candidates(model_name)
    fixed_index = cfg.fixed_candidate_indices.get(model_name)
    if fixed_index is not None and not 0 <= fixed_index < len(candidates):
        raise ValueError(f"Fixed candidate index out of range for {model_name}")
    candidate_items = enumerate(candidates) if fixed_index is None else [(fixed_index, candidates[fixed_index])]
    best_model, best_record = None, None
    history = []
    for candidate_index, settings in candidate_items:
        model = build_bike_model(model_name, train.feature_names, settings, cfg, seed)
        start = time.perf_counter()
        model.fit(train.inputs, train.targets)
        fit_seconds = time.perf_counter() - start
        record = dict(candidate_index=candidate_index, settings=dict(settings),
                      fit_seconds=fit_seconds,
                      train=regression_metrics(train.targets, model.predict(train.inputs)),
                      validation=regression_metrics(validation.targets, model.predict(validation.inputs)))
        history.append(record)
        if best_record is None or record["validation"]["rmse"] < best_record["validation"]["rmse"]:
            best_model, best_record = model, record
    return dict(model=best_model, model_name=model_name, seed=seed, history=history,
                best_candidate_index=best_record["candidate_index"],
                settings=best_record["settings"], results_train=best_record["train"],
                results_validation=best_record["validation"],
                candidate_policy="validation_search" if fixed_index is None else "fixed_before_run",
                checkpoint_split="validation", refit_on_validation=False)