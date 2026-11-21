"""Train and tune flow classifiers on CICIDS2018 (validation set only; test set untouched).

    python scripts/train.py --task multiclass --models rf xgb lgbm cat mlp lr dt
    python scripts/train.py --task binary --models lgbm
    python scripts/train.py --stage scalers        # scaler comparison for the MLP
    python scripts/train.py --stage bundle         # build deployment bundle (needs gaussian + host model)

Each model: random search over its SEARCH_SPACE on a stratified training
subsample, scored by validation macro-F1; the best configuration is refit on
the full training split (gradient boosting uses the validation split for early
stopping). Models, parameters, tuning tables, timings and validation metrics
are written under models/<model>/<task>/ and reports/experiments/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.preprocessing import LogScaler  # noqa: E402
from src.models import (baselines, catboost_model, lightgbm_model, random_forest,  # noqa: E402
                        xgboost_model)
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import classification_metrics  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
PROCESSED = project_path("processed")
MODELS = project_path("models")
EXP = project_path("experiments")
PREP_DIR = MODELS / "preprocessing"
LOG = get_logger("train", project_path("logs") / "train.log")

TREE_MODULES = {"rf": random_forest, "xgb": xgboost_model, "lgbm": lightgbm_model, "cat": catboost_model}
MODEL_DIRS = {"rf": "random_forest", "xgb": "xgboost", "lgbm": "lightgbm", "cat": "catboost", "mlp": "mlp",
              "lr": "baselines", "dt": "baselines"}
N_ITER = {"rf": 6, "xgb": 8, "lgbm": 10, "cat": 4}
TUNE_ROWS = 250_000


def load_data(task: str, prefix: str = "split", features: list[str] | None = None):
    selected = features or json.loads((PREP_DIR / "selected_features.json").read_text())
    families = json.loads((PREP_DIR / "family_classes.json").read_text())
    frames = {n: pd.read_parquet(PROCESSED / f"{prefix}_{n}_clean.parquet") for n in ("train", "val", "test")}
    if task == "binary":
        classes = ["BENIGN", "ATTACK"]
        ys = {n: f["IsAttack"].to_numpy(np.int64) for n, f in frames.items()}
    else:
        classes = families
        idx = {f: i for i, f in enumerate(families)}
        ys = {n: f["Family"].map(idx).to_numpy(np.int64) for n, f in frames.items()}
    xs = {n: f[selected].to_numpy(np.float32) for n, f in frames.items()}
    return xs, ys, classes, selected, frames


def stratified_indices(y: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    frac = n / len(y)
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        take = len(idx) if len(idx) * frac < 300 else int(round(len(idx) * frac))
        out.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def _fit_tree(key: str, params: dict, n_classes: int, x_tr, y_tr, x_va, y_va, seed: int = SEED):
    mod = TREE_MODULES[key]
    model = mod.build(params, seed=seed) if key == "rf" else mod.build(params, seed=seed, n_classes=n_classes)
    t0 = time.perf_counter()
    if key == "rf":
        mod.fit(model, x_tr, y_tr)
    else:
        mod.fit(model, x_tr, y_tr, x_va, y_va)
    return model, time.perf_counter() - t0


def tune_tree(key: str, xs, ys, classes, task: str) -> tuple[dict, pd.DataFrame]:
    sub = stratified_indices(ys["train"], TUNE_ROWS)
    rows = []
    sampler = ParameterSampler(TREE_MODULES[key].SEARCH_SPACE, n_iter=N_ITER[key], random_state=SEED)
    for i, params in enumerate(sampler):
        model, secs = _fit_tree(key, params, len(classes), xs["train"][sub], ys["train"][sub], xs["val"], ys["val"])
        pred = model.predict(xs["val"]).reshape(-1)
        met = classification_metrics(ys["val"], pred, None, classes)
        best_it = getattr(TREE_MODULES[key], "best_iteration", lambda m: None)(model)
        rows.append({"config": i, **{k: str(v) for k, v in params.items()}, "fit_seconds": round(secs, 1),
                     "best_iteration": best_it, "val_f1_macro": met["f1_macro"], "val_fpr": met["fpr"],
                     "val_recall_macro": met["recall_macro"]})
        LOG.info("[%s/%s] config %d macroF1=%.4f fpr=%.5f (%.0fs) %s", key, task, i, met["f1_macro"], met["fpr"],
                 secs, params)
    table = pd.DataFrame(rows).sort_values(["val_f1_macro", "val_fpr"], ascending=[False, True])
    best = list(ParameterSampler(TREE_MODULES[key].SEARCH_SPACE, n_iter=N_ITER[key],
                                 random_state=SEED))[int(table.iloc[0]["config"])]
    return best, table


def evaluate_split(model, x, y, classes, proba_fn=None) -> tuple[dict, np.ndarray, float]:
    t0 = time.perf_counter()
    proba = proba_fn(x) if proba_fn else model.predict_proba(x)
    secs = time.perf_counter() - t0
    pred = proba.argmax(1)
    return classification_metrics(y, pred, proba, classes), proba, secs


def model_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.parent.glob("*") if p.is_file()) if path.exists() else 0


def train_tree(key: str, task: str, prefix: str = "split") -> dict:
    xs, ys, classes, features, _ = load_data(task, prefix)
    best, table = tune_tree(key, xs, ys, classes, task)
    out = MODELS / MODEL_DIRS[key] / task
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tuning.csv", index=False)
    model, secs = _fit_tree(key, best, len(classes), xs["train"], ys["train"], xs["val"], ys["val"])
    joblib.dump(model, out / "model.joblib")
    best_it = getattr(TREE_MODULES[key], "best_iteration", lambda m: None)(model)
    train_m, _, _ = evaluate_split(model, xs["train"], ys["train"], classes)
    val_m, _, val_secs = evaluate_split(model, xs["val"], ys["val"], classes)
    record = {"model": key, "task": task, "params": {k: (v if not isinstance(v, np.generic) else v.item())
                                                    for k, v in best.items()},
              "best_iteration": best_it, "features": features, "classes": classes,
              "train_rows": int(len(ys["train"])), "train_seconds": round(secs, 2),
              "val_inference_us_per_flow": val_secs / len(ys["val"]) * 1e6,
              "model_size_mb": model_size_bytes(out / "model.joblib") / 2**20,
              "train_metrics": train_m, "val_metrics": val_m}
    (out / "params.json").write_text(json.dumps(record, indent=2, default=str))
    LOG.info("[%s/%s] FINAL train %.0fs val macroF1=%.4f fpr=%.5f", key, task, secs, val_m["f1_macro"], val_m["fpr"])
    return record


def train_mlp(task: str, prefix: str = "split") -> dict:
    from src.models.mlp import MLPClassifier, MLPConfig
    from src.utils import visualization as viz

    xs, ys, classes, features, frames = load_data(task, prefix)
    scaler = LogScaler("standard").fit(frames["train"][features])
    xt = {n: scaler.transform(frames[n][features]) for n in ("train", "val", "test")}
    out = MODELS / "mlp" / task
    out.mkdir(parents=True, exist_ok=True)
    # small search on a subsample, short schedule
    sub = stratified_indices(ys["train"], TUNE_ROWS)
    grid = [MLPConfig(dropout=0.2, lr=1e-3, batch_size=512, max_epochs=12, patience=3),
            MLPConfig(dropout=0.3, lr=1e-3, batch_size=512, max_epochs=12, patience=3),
            MLPConfig(dropout=0.2, lr=5e-4, batch_size=256, max_epochs=12, patience=3),
            MLPConfig(dropout=0.2, lr=1e-3, batch_size=512, max_epochs=12, patience=3, class_weighting=False)]
    rows = []
    for i, cfg in enumerate(grid):
        m = MLPClassifier(cfg).fit(xt["train"][sub], ys["train"][sub], xt["val"], ys["val"], log=LOG.debug)
        met = classification_metrics(ys["val"], m.predict(xt["val"]), None, classes)
        rows.append({"config": i, "dropout": cfg.dropout, "lr": cfg.lr, "batch_size": cfg.batch_size,
                     "class_weighting": cfg.class_weighting, "best_epoch": m.history.best_epoch,
                     "val_f1_macro": met["f1_macro"], "val_fpr": met["fpr"]})
        LOG.info("[mlp/%s] config %d macroF1=%.4f fpr=%.5f", task, i, met["f1_macro"], met["fpr"])
    table = pd.DataFrame(rows).sort_values(["val_f1_macro", "val_fpr"], ascending=[False, True])
    table.to_csv(out / "tuning.csv", index=False)
    best_cfg = grid[int(table.iloc[0]["config"])]
    best_cfg.max_epochs, best_cfg.patience = 40, 6
    t0 = time.perf_counter()
    model = MLPClassifier(best_cfg).fit(xt["train"], ys["train"], xt["val"], ys["val"], log=LOG.info)
    secs = time.perf_counter() - t0
    model.save(out)
    joblib.dump(scaler, out / "scaler.joblib")
    (out / "label_mapping.json").write_text(json.dumps(classes))
    viz.training_curves(model.history, project_path("figures") / f"mlp_{task}_training_curves.png",
                        f"MLP ({task})")
    train_m, _, _ = evaluate_split(model, xt["train"], ys["train"], classes, model.predict_proba)
    val_m, _, val_secs = evaluate_split(model, xt["val"], ys["val"], classes, model.predict_proba)
    record = {"model": "mlp", "task": task, "params": {k: v for k, v in best_cfg.__dict__.items()},
              "best_epoch": model.history.best_epoch, "epochs_run": len(model.history.train_loss),
              "features": features, "classes": classes, "train_rows": int(len(ys["train"])),
              "train_seconds": round(secs, 2), "val_inference_us_per_flow": val_secs / len(ys["val"]) * 1e6,
              "model_size_mb": model_size_bytes(out / "mlp_weights.pt") / 2**20,
              "train_metrics": train_m, "val_metrics": val_m}
    (out / "params.json").write_text(json.dumps(record, indent=2, default=str))
    LOG.info("[mlp/%s] FINAL %.0fs best_epoch=%d val macroF1=%.4f", task, secs, model.history.best_epoch,
             val_m["f1_macro"])
    return record


def train_baseline(key: str, task: str, prefix: str = "split") -> dict:
    xs, ys, classes, features, frames = load_data(task, prefix)
    out = MODELS / "baselines" / f"{key}_{task}"
    out.mkdir(parents=True, exist_ok=True)
    scaler = None
    if key == "lr":
        scaler = LogScaler("standard").fit(frames["train"][features])
        xs = {n: scaler.transform(frames[n][features]) for n in xs}
        model = baselines.logistic_regression(SEED)
    else:
        model = baselines.decision_tree(SEED)
    t0 = time.perf_counter()
    model.fit(xs["train"], ys["train"])
    secs = time.perf_counter() - t0
    joblib.dump({"model": model, "scaler": scaler}, out / "model.joblib")
    train_m, _, _ = evaluate_split(model, xs["train"], ys["train"], classes)
    val_m, _, val_secs = evaluate_split(model, xs["val"], ys["val"], classes)
    record = {"model": key, "task": task, "features": features, "classes": classes,
              "train_rows": int(len(ys["train"])), "train_seconds": round(secs, 2),
              "val_inference_us_per_flow": val_secs / len(ys["val"]) * 1e6,
              "model_size_mb": model_size_bytes(out / "model.joblib") / 2**20,
              "train_metrics": train_m, "val_metrics": val_m}
    (out / "params.json").write_text(json.dumps(record, indent=2, default=str))
    LOG.info("[%s/%s] val macroF1=%.4f", key, task, val_m["f1_macro"])
    return record


def compare_scalers() -> None:
    """Scaler choice for the MLP, decided on validation macro-F1 with a short schedule."""
    from src.models.mlp import MLPClassifier, MLPConfig

    xs, ys, classes, features, frames = load_data("multiclass")
    sub = stratified_indices(ys["train"], 200_000)
    rows = []
    for kind in ("standard", "robust", "minmax"):
        for log in (False, True):
            sc = LogScaler(kind, log=log).fit(frames["train"][features].iloc[sub])
            xt, xv = sc.transform(frames["train"][features].iloc[sub]), sc.transform(frames["val"][features])
            m = MLPClassifier(MLPConfig(max_epochs=6, patience=3)).fit(xt, ys["train"][sub], xv, ys["val"],
                                                                         log=LOG.debug)
            met = classification_metrics(ys["val"], m.predict(xv), None, classes)
            rows.append({"scaler": kind, "log1p": log, "val_f1_macro": met["f1_macro"], "val_fpr": met["fpr"],
                         "val_accuracy": met["accuracy"]})
            LOG.info("scaler %s log=%s macroF1=%.4f", kind, log, met["f1_macro"])
    pd.DataFrame(rows).to_csv(EXP / "scaler_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="models", choices=["models", "scalers"])
    parser.add_argument("--task", default="multiclass", choices=["multiclass", "binary"])
    parser.add_argument("--models", nargs="+", default=["lr", "dt", "rf", "xgb", "lgbm", "cat", "mlp"])
    args = parser.parse_args()
    if args.stage == "scalers":
        compare_scalers()
        return
    for key in args.models:
        LOG.info("===== %s / %s =====", key, args.task)
        if key in TREE_MODULES:
            record = train_tree(key, args.task)
        elif key == "mlp":
            record = train_mlp(args.task)
        else:
            record = train_baseline(key, args.task)
        update_summary(args.task, record)       # after every model, so an interrupted run keeps finished ones


def update_summary(task: str, record: dict) -> None:
    summary_path = EXP / f"training_summary_{task}.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    existing[record["model"]] = {k: record[k] for k in ("train_seconds", "val_inference_us_per_flow",
                                                        "model_size_mb", "val_metrics", "train_metrics")
                                 if k in record}
    summary_path.write_text(json.dumps(existing, indent=2, default=str))


if __name__ == "__main__":
    main()
