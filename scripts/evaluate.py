"""Independent test-set evaluation and experimental analysis.

    python scripts/evaluate.py --stage test        # all trained models, both tasks
    python scripts/evaluate.py --stage importance  # RF/XGB/LGBM importance + SHAP
    python scripts/evaluate.py --stage repeats     # seeds (model + data split) -> mean +- std
    python scripts/evaluate.py --stage temporal    # chronological split generalisation
    python scripts/evaluate.py --stage ablation    # Dst Port / TCP window shortcut features

The test split is only read here, after model selection on validation.
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
import psutil
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import MODEL_DIRS, TREE_MODULES, _fit_tree, load_data, stratified_indices  # noqa: E402
from src.data_processing.preprocessing import FlowCleaner, stratified_split  # noqa: E402
from src.models.mlp import MLPClassifier  # noqa: E402
from src.utils import visualization as viz  # noqa: E402
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import classification_metrics, per_class_report  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
MODELS = project_path("models")
FIG = project_path("figures")
METRICS = project_path("metrics")
EXP = project_path("experiments")
PROCESSED = project_path("processed")
LOG = get_logger("evaluate", project_path("logs") / "evaluate.log")
DISPLAY = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost",
           "lgbm": "LightGBM", "cat": "CatBoost", "mlp": "MLP"}


def load_model(key: str, task: str):
    """Return (predict_proba callable, model dir, train record) or None if not trained."""
    if key in ("lr", "dt"):
        d = MODELS / "baselines" / f"{key}_{task}"
        if not (d / "model.joblib").exists():
            return None
        obj = joblib.load(d / "model.joblib")
        model, scaler = obj["model"], obj["scaler"]
        fn = (lambda x, m=model, s=scaler: m.predict_proba(s.transform(x))) if scaler else model.predict_proba
        return fn, d, json.loads((d / "params.json").read_text())
    d = MODELS / MODEL_DIRS[key] / task
    if key == "mlp":
        if not (d / "mlp_weights.pt").exists():
            return None
        model = MLPClassifier.load(d)
        scaler = joblib.load(d / "scaler.joblib")
        return (lambda x: model.predict_proba(scaler.transform(x))), d, json.loads((d / "params.json").read_text())
    if not (d / "model.joblib").exists():
        return None
    model = joblib.load(d / "model.joblib")
    return model.predict_proba, d, json.loads((d / "params.json").read_text())


def dir_size_mb(d: Path) -> float:
    return sum(p.stat().st_size for p in d.glob("*") if p.is_file() and p.suffix in (".joblib", ".pt")) / 2**20


def stage_test() -> None:
    proc = psutil.Process()
    for task in ("multiclass", "binary"):
        xs, ys, classes, features, frames = load_data(task)
        x_test_df = frames["test"][features]
        rows, roc_curves, pr_curves = [], {}, {}
        for key in ["lr", "dt", "rf", "xgb", "lgbm", "cat", "mlp"]:
            loaded = load_model(key, task)
            if loaded is None:
                continue
            proba_fn, mdir, rec = loaded
            x_in = xs["test"] if key not in ("lr", "mlp") else x_test_df
            rss0 = proc.memory_info().rss
            t0 = time.perf_counter()
            proba = proba_fn(x_in)
            secs = time.perf_counter() - t0
            rss_delta = (proc.memory_info().rss - rss0) / 2**20
            single = x_in[:1] if not isinstance(x_in, pd.DataFrame) else x_in.iloc[:1]
            t1 = time.perf_counter()
            for _ in range(200):
                proba_fn(single)
            single_ms = (time.perf_counter() - t1) / 200 * 1000
            pred = proba.argmax(1)
            m = classification_metrics(ys["test"], pred, proba, classes)
            per_class = per_class_report(ys["test"], pred, classes)
            per_class.to_csv(METRICS / f"per_class_{task}_{key}.csv", index=False)
            cm = confusion_matrix(ys["test"], pred, labels=list(range(len(classes))))
            np.savetxt(METRICS / f"cm_{task}_{key}.csv", cm, fmt="%d", delimiter=",")
            viz.confusion_matrix_plot(cm, classes, f"{DISPLAY[key]}: {task} confusion matrix (test)",
                                      FIG / f"cm_{task}_{key}.png")
            score = proba[:, 1] if task == "binary" else 1 - proba[:, 0]
            is_attack = (ys["test"] != 0).astype(int)
            fpr, tpr, _ = roc_curve(is_attack, score)
            prec, recall, _ = precision_recall_curve(is_attack, score)
            auc_key = "roc_auc" if task == "binary" else "binary_view_roc_auc"
            pr_key = "pr_auc" if task == "binary" else "binary_view_pr_auc"
            roc_curves[DISPLAY[key]] = (fpr, tpr, m.get(auc_key, np.nan))
            pr_curves[DISPLAY[key]] = (recall, prec, m.get(pr_key, np.nan))
            row = {"model": DISPLAY[key], "key": key, **m,
                   "train_f1_macro": rec["train_metrics"]["f1_macro"], "val_f1_macro": rec["val_metrics"]["f1_macro"],
                   "train_seconds": rec["train_seconds"], "inference_us_per_flow": secs / len(pred) * 1e6,
                   "throughput_flows_per_s": len(pred) / secs, "single_flow_latency_ms": single_ms,
                   "model_size_mb": dir_size_mb(mdir), "predict_rss_delta_mb": rss_delta}
            rows.append(row)
            LOG.info("[%s/%s] test macroF1=%.4f weightedF1=%.4f FPR=%.5f FNR=%.5f", key, task, m["f1_macro"],
                     m["f1_weighted"], m["fpr"], m["fnr"])
        table = pd.DataFrame(rows)
        table.to_csv(METRICS / f"model_comparison_{task}.csv", index=False)
        viz.curves_plot(roc_curves, "roc", f"ROC, attack vs. benign ({task} models, test)", FIG / f"roc_{task}.png")
        viz.curves_plot(pr_curves, "pr", f"Precision-recall, attack vs. benign ({task} models, test)",
                        FIG / f"pr_{task}.png")


def stage_importance() -> None:
    xs, ys, classes, features, frames = load_data("multiclass")
    imps = {}
    for key, attr in (("rf", "feature_importances_"), ("xgb", "feature_importances_"), ("lgbm", None)):
        p = MODELS / MODEL_DIRS[key] / "multiclass" / "model.joblib"
        if not p.exists():
            continue
        model = joblib.load(p)
        if key == "lgbm":
            vals = model.booster_.feature_importance(importance_type="gain")
        else:
            vals = getattr(model, attr)
        s = pd.Series(vals, index=features, dtype=float)
        imps[DISPLAY[key]] = s / s.sum()
        viz.importance_plot(imps[DISPLAY[key]], f"{DISPLAY[key]} feature importance (multiclass)",
                            FIG / f"importance_{key}.png")
    table = pd.DataFrame(imps)
    table["mean"] = table.mean(axis=1)
    table.sort_values("mean", ascending=False).to_csv(METRICS / "feature_importance_models.csv")
    p = MODELS / "lightgbm" / "multiclass" / "model.joblib"
    if p.exists():
        # LightGBM's built-in TreeSHAP (pred_contrib) gives the same values as shap.TreeExplainer
        # but in seconds instead of ~1 h for this 300-tree x 255-leaf x 7-class model.
        model = joblib.load(p)
        idx = stratified_indices(ys["test"], 3000)
        nf, k = len(features), len(classes)
        contrib = model.booster_.predict(np.asarray(xs["test"][idx], dtype=np.float64), pred_contrib=True)
        sv = contrib.reshape(len(idx), k, nf + 1)[:, :, :nf]   # (n, classes, features); drop bias term
        mean_abs = np.abs(sv).mean(axis=(0, 1))
        per_class = {c: np.abs(sv[:, i, :]).mean(0) for i, c in enumerate(classes)}
        shap_table = pd.DataFrame(per_class, index=features)
        shap_table["mean_abs_shap"] = mean_abs
        shap_table.sort_values("mean_abs_shap", ascending=False).to_csv(METRICS / "shap_lightgbm_multiclass.csv")
        viz.importance_plot(pd.Series(mean_abs, index=features), "LightGBM mean |SHAP| (test sample, all classes)",
                            FIG / "shap_lightgbm.png")


def _best_params(key: str, task: str = "multiclass") -> dict:
    rec = json.loads((MODELS / MODEL_DIRS[key] / task / "params.json").read_text())
    return rec["params"]


def _eval(model, x, y, classes) -> dict:
    proba = model.predict_proba(x)
    return classification_metrics(y, proba.argmax(1), proba, classes)


def stage_repeats() -> None:
    xs, ys, classes, features, frames = load_data("multiclass")
    rows = []
    for key in ("lgbm", "xgb", "rf"):
        params = _best_params(key)
        for seed in CFG["project"]["seeds_for_repeats"]:
            model, secs = _fit_tree(key, params, len(classes), xs["train"], ys["train"], xs["val"], ys["val"], seed)
            m = _eval(model, xs["test"], ys["test"], classes)
            rows.append({"experiment": "model_seed", "model": DISPLAY[key], "seed": seed, **m})
            LOG.info("repeat %s seed %d macroF1=%.4f", key, seed, m["f1_macro"])
    from scripts.preprocess_dataset import drop_conflicting_vectors
    sample = drop_conflicting_vectors(pd.read_parquet(project_path("interim") / "cicids2018_sample.parquet"))
    fam_index = {f: i for i, f in enumerate(classes)}
    for seed in CFG["project"]["seeds_for_repeats"]:
        split = stratified_split(sample, "FineLabel", (0.7, 0.15, 0.15), seed)
        cleaner = FlowCleaner().fit(split.train)
        x = {n: cleaner.transform(getattr(split, n))[features].to_numpy(np.float32) for n in ("train", "val", "test")}
        y = {n: getattr(split, n)["Family"].map(fam_index).to_numpy() for n in ("train", "val", "test")}
        model, _ = _fit_tree("lgbm", _best_params("lgbm"), len(classes), x["train"], y["train"], x["val"], y["val"],
                             SEED)
        m = _eval(model, x["test"], y["test"], classes)
        rows.append({"experiment": "data_split_seed", "model": "LightGBM", "seed": seed, **m})
        LOG.info("resplit seed %d macroF1=%.4f", seed, m["f1_macro"])
    df = pd.DataFrame(rows)
    df.to_csv(EXP / "repeats_raw.csv", index=False)
    cols = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "precision_macro", "fpr", "fnr"]
    summary = df.groupby(["experiment", "model"])[cols].agg(["mean", "std"])
    summary.to_csv(EXP / "repeats_summary.csv")


def stage_temporal() -> None:
    rows = []
    for prefix in ("split", "temporal"):
        xs, ys, classes, features, _ = load_data("multiclass", prefix)
        for key in ("lgbm", "rf"):
            model, _ = _fit_tree(key, _best_params(key), len(classes), xs["train"], ys["train"], xs["val"], ys["val"])
            proba = model.predict_proba(xs["test"])
            pred = proba.argmax(1)
            m = classification_metrics(ys["test"], pred, proba, classes)
            pc = per_class_report(ys["test"], pred, classes).set_index("class")["recall"].add_prefix("recall_")
            rows.append({"split": prefix, "model": DISPLAY[key], **m, **pc.to_dict()})
            if prefix == "temporal":
                cm = confusion_matrix(ys["test"], pred, labels=list(range(len(classes))))
                viz.confusion_matrix_plot(cm, classes, f"{DISPLAY[key]}: temporal split confusion matrix (test)",
                                          FIG / f"cm_temporal_{key}.png")
            LOG.info("%s %s macroF1=%.4f", prefix, key, m["f1_macro"])
    pd.DataFrame(rows).to_csv(EXP / "temporal_vs_random_split.csv", index=False)


def stage_ablation() -> None:
    selected = json.loads((MODELS / "preprocessing" / "selected_features.json").read_text())
    variants = {
        "selected": selected,
        "without_dst_port": [f for f in selected if f != "Dst Port"],
        "without_tcp_init_window": [f for f in selected if f not in ("Init Fwd Win Byts", "Init Bwd Win Byts")],
        "without_both": [f for f in selected if f not in ("Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts")],
    }
    rows = []
    for name, feats in variants.items():
        xs, ys, classes, _, _ = load_data("multiclass", features=feats)
        model, _ = _fit_tree("lgbm", _best_params("lgbm"), len(classes), xs["train"], ys["train"], xs["val"], ys["val"])
        proba = model.predict_proba(xs["test"])
        pred = proba.argmax(1)
        m = classification_metrics(ys["test"], pred, proba, classes)
        pc = per_class_report(ys["test"], pred, classes).set_index("class")["f1"].add_prefix("f1_")
        rows.append({"variant": name, "n_features": len(feats), **m, **pc.to_dict()})
        LOG.info("ablation %s macroF1=%.4f", name, m["f1_macro"])
    pd.DataFrame(rows).to_csv(EXP / "feature_ablation.csv", index=False)


STAGES = {"test": stage_test, "importance": stage_importance, "repeats": stage_repeats,
          "temporal": stage_temporal, "ablation": stage_ablation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="test", choices=[*STAGES, "all"])
    args = parser.parse_args()
    for name in (STAGES if args.stage == "all" else [args.stage]):
        LOG.info("=== %s ===", name)
        STAGES[name]()


if __name__ == "__main__":
    main()
