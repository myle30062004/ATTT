"""Assemble the deployment bundle used by the detection pipeline and realtime monitor.

    python scripts/build_bundle.py [--model lgbm]

Classifier choice (if --model is not given) uses VALIDATION metrics only: the
multiclass tree model with the best validation macro-F1; among models within
0.005 of the best, the one with the lowest inference time is preferred
(realtime constraint). The test set plays no part in this choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import MODEL_DIRS  # noqa: E402
from src.detection.bundle import save_bundle  # noqa: E402
from src.utils.config import project_path  # noqa: E402

MODELS = project_path("models")


def choose_model() -> tuple[str, dict]:
    rows = []
    for key in ("rf", "xgb", "lgbm", "cat"):
        p = MODELS / MODEL_DIRS[key] / "multiclass" / "params.json"
        if p.exists():
            rec = json.loads(p.read_text())
            rows.append({"key": key, "val_f1_macro": rec["val_metrics"]["f1_macro"],
                         "us_per_flow": rec["val_inference_us_per_flow"]})
    if not rows:
        raise SystemExit("no trained multiclass tree model found; run scripts/train.py first")
    df = pd.DataFrame(rows)
    best = df["val_f1_macro"].max()
    near = df[df["val_f1_macro"] >= best - 0.005].sort_values("us_per_flow")
    return near.iloc[0]["key"], {"candidates": rows, "rule": "best val macro-F1, fastest within 0.005"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb", "lgbm", "cat"])
    args = parser.parse_args()
    key, selection = (args.model, {"rule": "user choice"}) if args.model else choose_model()
    mdir = MODELS / MODEL_DIRS[key] / "multiclass"
    rec = json.loads((mdir / "params.json").read_text())
    clf = joblib.load(mdir / "model.joblib")
    cleaner = joblib.load(MODELS / "preprocessing" / "cleaner_split.joblib")
    gauss = joblib.load(MODELS / "gaussian" / "gaussian.joblib")
    features = rec["features"]
    imp_path = project_path("metrics") / "feature_importance_models.csv"
    importances = {}
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)["mean"].sort_values(ascending=False)
        importances = imp.to_dict()
    host = MODELS / "botnet" / "host_model.joblib"
    host_obj = joblib.load(host) if host.exists() else None
    out = save_bundle(
        MODELS / "deployment", cleaner=cleaner, classifier=clf, classifier_features=features,
        class_names=rec["classes"], gaussian=gauss["detector"], gaussian_scaler=gauss["scaler"],
        gaussian_features=gauss["features"], importances=importances,
        host_model=host_obj["model"] if host_obj else None,
        host_model_features=host_obj["features"] if host_obj else None,
        meta={"classifier_key": key, "selection": selection, "val_metrics": rec["val_metrics"],
              "classifier_params": rec["params"]})
    print(f"bundle written to {out} (classifier={key})")


if __name__ == "__main__":
    main()
