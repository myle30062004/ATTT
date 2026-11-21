"""Plotting helpers (matplotlib) with one consistent, colour-blind-validated style.

Palette: categorical hues are assigned in a fixed order (never cycled by rank);
magnitudes (confusion matrices) use a single-hue blue ramp; correlations use a
blue<->red diverging map with a neutral grey midpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#9a9890"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f5f9fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b42b2b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 9, "font.size": 10, "lines.linewidth": 2.0, "figure.dpi": 110,
})


def color_for(index: int) -> str:
    return CATEGORICAL[index] if index < len(CATEGORICAL) else OTHER


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def bar_counts(series: pd.Series, title: str, path: Path, log: bool = True, xlabel: str = "Rows") -> Path:
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(s) + 1)))
    ax.barh(s.index.astype(str), s.values, color=CATEGORICAL[0], height=0.6)
    if log:
        ax.set_xscale("log")
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel + (" (log scale)" if log else ""))
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def confusion_matrix_plot(cm: np.ndarray, class_names: list[str], title: str, path: Path,
                          normalize: bool = True) -> Path:
    cm = np.asarray(cm, dtype=np.float64)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 0.85 * n + 2))
    im = ax.imshow(shown, cmap=SEQUENTIAL, vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    ax.set_xticks(range(n), class_names, rotation=40, ha="right")
    ax.set_yticks(range(n), class_names)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            val = shown[i, j]
            txt = f"{val:.2f}\n({int(cm[i, j]):,})" if normalize else f"{int(cm[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="#ffffff" if (val > 0.55 if normalize else val > shown.max() * 0.55) else TEXT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised share" if normalize else "Count")
    return save(fig, path)


def curves_plot(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], kind: str, title: str, path: Path) -> Path:
    """``curves``: name -> (x, y, auc). kind = 'roc' or 'pr'."""
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, (name, (x, y, auc)) in enumerate(curves.items()):
        ax.plot(x, y, color=color_for(i), linewidth=2, label=f"{name} (AUC={auc:.4f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=OTHER, linewidth=1, linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right" if kind == "roc" else "lower left")
    return save(fig, path)


def importance_plot(importance: pd.Series, title: str, path: Path, top: int = 20) -> Path:
    s = importance.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=CATEGORICAL[0], height=0.6)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Importance")
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def training_curves(history, path: Path, title: str = "MLP training") -> Path:
    epochs = np.arange(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tr, va, name) in zip(axes, [(history.train_loss, history.val_loss, "Loss"),
                                         (history.train_acc, history.val_acc, "Accuracy")]):
        ax.plot(epochs, tr, color=CATEGORICAL[0], label="Train")
        ax.plot(epochs, va, color=CATEGORICAL[1], label="Validation")
        ax.annotate("Train", (epochs[-1], tr[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TEXT_SECONDARY, fontsize=8, va="center")
        ax.annotate("Validation", (epochs[-1], va[-1]), xytext=(4, 0), textcoords="offset points",
                    color=TEXT_SECONDARY, fontsize=8, va="center")
        if history.best_epoch > 0:
            ax.axvline(history.best_epoch, color=OTHER, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_title(f"{title}: {name}", loc="left")
        ax.legend()
    return save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, title: str, path: Path) -> Path:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.28 * n + 3, 0.28 * n + 2.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n), corr.index, fontsize=6)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    return save(fig, path)


def score_distribution(scores: dict[str, np.ndarray], threshold: float | None, title: str, path: Path,
                       xlabel: str = "Anomaly score (-log p(x))") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    all_vals = np.concatenate([v for v in scores.values() if len(v)])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    for i, (name, vals) in enumerate(scores.items()):
        ax.hist(np.clip(vals, lo, hi), bins=bins, density=True, histtype="step", linewidth=2,
                color=color_for(i), label=name)
    if threshold is not None:
        ax.axvline(threshold, color=TEXT, linestyle="--", linewidth=1)
        ax.annotate("threshold ε", (threshold, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left")
    ax.legend()
    return save(fig, path)
