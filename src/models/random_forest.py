"""Random Forest classifier: builder and hyper-parameter search space."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def build(params: dict | None = None, seed: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    params = dict(params or {})
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)


def fit(model, x_train, y_train, x_val=None, y_val=None, sample_weight=None):
    return model.fit(x_train, y_train, sample_weight=sample_weight)
