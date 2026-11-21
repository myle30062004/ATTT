"""LightGBM classifier with early stopping on the validation split.

``min_sum_hessian_in_leaf`` defaults to 1.0 (LightGBM default: 1e-3). With the
default, multiclass training on CICIDS2018 diverged: on a 150k-row training
sample the training multi-logloss fell to 0.185 at iteration 27 and then rose to
8.29 at iteration 300 (validation macro-F1 0.62). Nearly separable classes give
leaves with almost zero hessian and therefore huge leaf values; requiring a
minimum hessian of 1 per leaf removed the divergence (validation macro-F1 0.90).
"""
from __future__ import annotations

import lightgbm as lgb

SEARCH_SPACE = {
    "n_estimators": [300, 600, 1000],
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [-1, 12, 20],
    "learning_rate": [0.03, 0.05, 0.1],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.7, 0.85, 1.0],
    "lambda_l1": [0.0, 0.1, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0],
    "min_child_samples": [10, 20, 50],
    "class_weight": [None, "balanced"],
    "min_sum_hessian_in_leaf": [1.0, 10.0],
}


def build(params: dict | None = None, seed: int = 42, n_jobs: int = -1, n_classes: int = 2) -> lgb.LGBMClassifier:
    params = dict(params or {})
    params.setdefault("min_sum_hessian_in_leaf", 1.0)
    if params.get("bagging_fraction", 1.0) < 1.0:
        params.setdefault("bagging_freq", 1)
    return lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **params)


def fit(model, x_train, y_train, x_val=None, y_val=None, sample_weight=None):
    if x_val is None:
        return model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_val, y_val)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def best_iteration(model) -> int | None:
    return getattr(model, "best_iteration_", None)
