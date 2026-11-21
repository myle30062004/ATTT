"""CatBoost classifier (control model) with early stopping on validation."""
from __future__ import annotations

from catboost import CatBoostClassifier

SEARCH_SPACE = {
    "iterations": [500, 1000],
    "depth": [6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "l2_leaf_reg": [1, 3, 10],
    "auto_class_weights": [None, "Balanced"],
}


def build(params: dict | None = None, seed: int = 42, n_jobs: int = -1, n_classes: int = 2) -> CatBoostClassifier:
    params = dict(params or {})
    loss = "Logloss" if n_classes == 2 else "MultiClass"
    return CatBoostClassifier(loss_function=loss, random_seed=seed, thread_count=n_jobs, verbose=False,
                              early_stopping_rounds=30, allow_writing_files=False, **params)


def fit(model, x_train, y_train, x_val=None, y_val=None, sample_weight=None):
    if x_val is None:
        return model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=(x_val, y_val), use_best_model=True)


def best_iteration(model) -> int | None:
    return model.get_best_iteration()
