"""Classification and detection metrics.

Multi-class ROC-AUC / PR-AUC are computed one-vs-rest and reported both
macro-averaged (each class weighs the same, sensitive to rare attacks) and
micro-averaged (pooled over all class/sample pairs).

False-positive rate (FPR) for IDS purposes is the share of *benign* flows
flagged as any attack; false-negative rate (FNR) is the share of attack flows
predicted as benign. Per-class FPR is one-vs-rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
                           class_names: list[str], benign_index: int | None = 0) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "n_samples": int(len(y_true)),
    }
    if len(class_names) == 2:
        m["precision_attack"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["recall_attack"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        m["f1_attack"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    if benign_index is not None:
        benign = y_true == benign_index
        attack = ~benign
        m["fpr"] = float((y_pred[benign] != benign_index).mean()) if benign.any() else float("nan")
        m["fnr"] = float((y_pred[attack] == benign_index).mean()) if attack.any() else float("nan")
        m["attack_detection_rate"] = 1.0 - m["fnr"] if attack.any() else float("nan")
    if proba is not None:
        proba = np.asarray(proba)
        present = np.unique(y_true)
        try:
            if len(class_names) == 2:
                score = proba[:, 1] if proba.ndim == 2 else proba
                m["roc_auc"] = roc_auc_score(y_true, score)
                m["pr_auc"] = average_precision_score(y_true, score)
            else:
                y_bin = label_binarize(y_true, classes=labels)
                cols = [c for c in labels if c in present]
                m["roc_auc_ovr_macro"] = roc_auc_score(y_bin[:, cols], proba[:, cols], average="macro")
                m["roc_auc_ovr_micro"] = roc_auc_score(y_bin[:, cols], proba[:, cols], average="micro")
                m["pr_auc_ovr_macro"] = average_precision_score(y_bin[:, cols], proba[:, cols], average="macro")
                m["pr_auc_ovr_micro"] = average_precision_score(y_bin[:, cols], proba[:, cols], average="micro")
                # attack-vs-benign view of a multi-class model: P(attack) = 1 - P(benign)
                if benign_index is not None:
                    is_attack = (y_true != benign_index).astype(int)
                    if 0 < is_attack.sum() < len(is_attack):
                        p_attack = 1.0 - proba[:, benign_index]
                        m["binary_view_roc_auc"] = roc_auc_score(is_attack, p_attack)
                        m["binary_view_pr_auc"] = average_precision_score(is_attack, p_attack)
        except ValueError:
            pass
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def per_class_report(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    fpr = np.divide(fp, fp + tn, out=np.zeros(len(labels)), where=(fp + tn) > 0)
    return pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f,
                         "fpr_ovr": fpr, "fnr": 1 - r, "support": s})


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    """Binary anomaly/attack detection metrics (1 = attack/anomaly)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if score is not None and 0 < y_true.sum() < len(y_true):
        m["roc_auc"] = roc_auc_score(y_true, score)
        m["pr_auc"] = average_precision_score(y_true, score)
    return {k: (float(v) if isinstance(v, (np.floating, float, np.integer)) and k not in ("tp", "fp", "tn", "fn") else v)
            for k, v in m.items()}
