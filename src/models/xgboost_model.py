"""XGBoost classifier with early stopping on the validation split."""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

SEARCH_SPACE = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [6, 8, 10, 12],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.5],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [1.0, 2.0, 5.0],
}


def build(params: dict | None = None, seed: int = 42, n_jobs: int = -1, n_classes: int = 2) -> XGBClassifier:
    params = dict(params or {})
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return XGBClassifier(objective=objective, tree_method="hist", random_state=seed, n_jobs=n_jobs,
                         early_stopping_rounds=30, eval_metric="logloss" if n_classes == 2 else "mlogloss",
                         **params)


def fit(model, x_train, y_train, x_val=None, y_val=None, sample_weight=None):
    if x_val is None:
        model.set_params(early_stopping_rounds=None)
        return model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_val, y_val)], verbose=False)


def best_iteration(model) -> int | None:
    return getattr(model, "best_iteration", None)
