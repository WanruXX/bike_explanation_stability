"""Library-backed regressors with TRAIN-fitted preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from bike_datasets import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def model_candidates(model_name):
    candidates = dict(
        spline_ridge=[dict(alpha=1.0, n_knots=5), dict(alpha=10.0, n_knots=8)],
        random_forest=[dict(max_depth=12, min_samples_leaf=5),
                       dict(max_depth=None, min_samples_leaf=5)],
        gradient_boosting=[dict(max_leaf_nodes=15, l2_regularization=1.0),
                           dict(max_leaf_nodes=31, l2_regularization=1.0)],
        ebm=[dict(learning_rate=0.03, max_bins=128),
             dict(learning_rate=0.06, max_bins=256)],
    )
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")
    return candidates[model_name]


def build_bike_model(model_name, feature_names, settings, cfg, seed):
    if model_name == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as error:
            raise ImportError("The EBM arm requires interpret-core; install bike/requirements.txt") from error
        return ExplainableBoostingRegressor(
            feature_names=list(feature_names),
            feature_types=["nominal" if feature in CATEGORICAL_FEATURES else "continuous"
                           for feature in feature_names],
            interactions=0, outer_bags=1, max_rounds=3000, validation_size=0,
            n_jobs=cfg.n_jobs, random_state=seed, **settings,
        )
    categorical = [feature for feature in feature_names if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in feature_names if feature in NUMERIC_FEATURES]
    if model_name == "spline_ridge":
        numeric_transform = Pipeline([
            ("spline", SplineTransformer(n_knots=settings["n_knots"], degree=3,
                                          knots="quantile", include_bias=False)),
            ("scale", StandardScaler()),
        ])
        estimator = Ridge(alpha=settings["alpha"])
    elif model_name == "random_forest":
        numeric_transform = "passthrough"
        estimator = RandomForestRegressor(n_estimators=120, random_state=seed,
                                           n_jobs=cfg.n_jobs, **settings)
    elif model_name == "gradient_boosting":
        numeric_transform = "passthrough"
        estimator = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                                  early_stopping=False, random_state=seed,
                                                  **settings)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", numeric_transform, numeric),
    ], sparse_threshold=0)
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])