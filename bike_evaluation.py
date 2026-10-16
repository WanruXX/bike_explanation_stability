"""Demand scores and descriptive agreement metrics; no test-driven selection."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def regression_metrics(targets, predictions):
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape:
        raise ValueError("Targets and predictions must be aligned and nonempty")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite")
    error = predicted - actual
    mse = float(np.mean(error**2))
    variance = float(np.mean((actual - actual.mean())**2))
    return dict(rows=len(actual), mse=mse, rmse=float(np.sqrt(mse)),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                r2=1.0 - mse / variance if variance > 0 else None,
                target_mean=float(actual.mean()), prediction_mean=float(predicted.mean()),
                negative_prediction_count=int((predicted < 0).sum()))


def importance_agreement(first, second):
    if set(first) != set(second):
        raise ValueError("Importance panels must have the same names")
    names = list(first)
    first_values = np.array([first[name]["mean_mse_increase"] for name in names])
    second_values = np.array([second[name]["mean_mse_increase"] for name in names])
    first_ranks, second_ranks = rankdata(first_values), rankdata(second_values)
    correlation = None
    if np.std(first_ranks) > 0 and np.std(second_ranks) > 0:
        correlation = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    first_scale, second_scale = np.abs(first_values).sum(), np.abs(second_values).sum()
    normalized_distance = None
    if first_scale > 0 and second_scale > 0:
        normalized_distance = float(np.abs(first_values / first_scale - second_values / second_scale).sum())
    first_top = {name for name, value in zip(names, first_values) if value == first_values.max()}
    second_top = {name for name, value in zip(names, second_values) if value == second_values.max()}
    return dict(spearman=correlation, signed_normalized_l1=normalized_distance,
                top_tie_set_jaccard=len(first_top & second_top) / len(first_top | second_top))


def prediction_agreement(first, second, targets):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if first.shape != second.shape or first.shape != targets.shape:
        raise ValueError("Prediction comparison requires identical cohorts")
    difference = first - second
    target_std = float(np.std(targets))
    rmse = float(np.sqrt(np.mean(difference**2)))
    return dict(rmse=rmse, mae=float(np.mean(np.abs(difference))),
                rmse_over_target_std=rmse / target_std if target_std > 0 else None)


def block_bootstrap_means(row_values, timestamps, *, block_days, repeats, seed):
    values = np.asarray(row_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    times = pd.DatetimeIndex(timestamps)
    if values.ndim != 2 or not values.size or len(times) != len(values) or not np.isfinite(values).all():
        raise ValueError("Block bootstrap requires aligned finite row values")
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Block bootstrap timestamps must be unique and chronological")
    if not isinstance(block_days, int) or isinstance(block_days, bool) or block_days < 1:
        raise ValueError("block_days must be a positive integer")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 2:
        raise ValueError("At least two bootstrap repeats are required")
    labels = np.asarray((times.normalize() - times.min().normalize()).days) // block_days
    unique, inverse = np.unique(labels, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two observed calendar blocks are required")
    counts = np.bincount(inverse)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    sampled = np.random.default_rng(seed).integers(len(unique), size=(repeats, len(unique)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None]
    return dict(point=values.mean(axis=0), draws=means, blocks=len(unique), rows=len(values),
                block_days=block_days, repeats=repeats,
                method="Paired non-overlapping calendar-block percentile bootstrap; blocks sampled with replacement")


def bootstrap_interval(estimate, draws, confidence_level):
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(estimate):
        raise ValueError("Bootstrap intervals require finite scalar estimates and draws")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Invalid confidence level")
    tail = (1 - confidence_level) / 2
    bounds = np.quantile(values, [tail, 1 - tail])
    return dict(estimate=float(estimate), lower=float(bounds[0]), upper=float(bounds[1]),
                bootstrap_sd=float(np.std(values, ddof=1)), confidence_level=confidence_level)


def paired_prediction_block_intervals(first, second, targets, timestamps, *, block_days,
                                      repeats, seed, confidence_level):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.shape != targets.shape:
        raise ValueError("Paired predictions must be aligned vectors")
    errors_first, errors_second = first - targets, second - targets
    rows = np.column_stack((errors_first**2, errors_second**2, np.abs(errors_first),
                            np.abs(errors_second), (second - first)**2))
    bootstrap = block_bootstrap_means(rows, timestamps, block_days=block_days, repeats=repeats, seed=seed)

    def statistics(means):
        return dict(rmse_second_minus_first=np.sqrt(means[..., 1]) - np.sqrt(means[..., 0]),
                    mae_second_minus_first=means[..., 3] - means[..., 2],
                    prediction_disagreement_rmse=np.sqrt(means[..., 4]))

    point, draws = statistics(bootstrap["point"]), statistics(bootstrap["draws"])
    return dict(block_days=block_days, blocks=bootstrap["blocks"], rows=bootstrap["rows"],
                method=bootstrap["method"], repeats=repeats,
                metrics={name: bootstrap_interval(value, draws[name], confidence_level) for name, value in point.items()})