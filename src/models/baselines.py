"""Simple baselines used as reference points."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def logistic_regression(seed: int = 42) -> LogisticRegression:
    # Expects scaled input (LogScaler).
    return LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)


def decision_tree(seed: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=seed)
