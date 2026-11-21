"""Train the DEPLOYMENT models on parity-verified features and write the deployment bundle.

    python scripts/parity_check.py ...        # produces models/preprocessing/parity_safe_features.json
    python scripts/train_deployment.py [--model lgbm]

The research models (scripts/train.py) use the 30 selected CICFlowMeter
features. Some of them cannot be reproduced from packets the way CICFlowMeter
computed them (flag counts, Init Bwd Win Byts, IAT minima, Pkt Len Mean; see
reports/metrics/feature_parity_check.json). Models that score PCAP/live traffic
must only use features whose values agree with CICFlowMeter, so the
deployment classifier and Gaussian mixture are retrained on
``selected ∩ parity_safe`` with the hyper-parameters tuned on validation
(classifier) and the validation-selected number of mixture components.
Thresholds: Gaussian epsilon at 1% benign FPR on validation. Test metrics are
reported for transparency (they do not drive any choice).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import MODEL_DIRS, TREE_MODULES, _fit_tree, load_data  # noqa: E402
from src.data_processing.preprocessing import LogScaler  # noqa: E402
from src.detection.bundle import save_bundle  # noqa: E402
from src.models.gaussian_detector import GaussianMixtureDetector, threshold_at_fpr  # noqa: E402
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import classification_metrics, detection_metrics, per_class_report  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
MODELS = project_path("models")
METRICS = project_path("metrics")
PROCESSED = project_path("processed")
LOG = get_logger("deploy", project_path("logs") / "deployment.log")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lgbm", choices=list(TREE_MODULES))
    a = p.parse_args()
    selected = json.loads((MODELS / "preprocessing" / "selected_features.json").read_text())
    safe = set(json.loads((MODELS / "preprocessing" / "parity_safe_features.json").read_text()))
    feats = [f for f in selected if f in safe]
    dropped = [f for f in selected if f not in safe]
    LOG.info("deployment features: %d (dropped %s)", len(feats), dropped)

    xs, ys, classes, _, frames = load_data("multiclass", features=feats)
    rec = json.loads((MODELS / MODEL_DIRS[a.model] / "multiclass" / "params.json").read_text())
    t0 = time.perf_counter()
    clf, _ = _fit_tree(a.model, rec["params"], len(classes), xs["train"], ys["train"], xs["val"], ys["val"], SEED)
    fit_s = time.perf_counter() - t0
    out = {"classifier": a.model, "features": feats, "dropped_non_parity_features": dropped,
           "classifier_fit_seconds": fit_s}
    for split in ("val", "test"):
        proba = clf.predict_proba(xs[split])
        out[f"classifier_{split}"] = classification_metrics(ys[split], proba.argmax(1), proba, classes)
    out["classifier_test_per_class"] = per_class_report(ys["test"], clf.predict_proba(xs["test"]).argmax(1),
                                                         classes).to_dict(orient="records")
    out["research_model_test_reference"] = "reports/metrics/model_comparison_multiclass.csv"

    tr, va, te = frames["train"], frames["val"], frames["test"]
    benign = tr[tr["IsAttack"] == 0]
    benign = benign.sample(min(len(benign), 300_000), random_state=SEED)
    scaler = LogScaler("standard").fit(benign[feats])
    anomaly = json.loads((METRICS / "anomaly_results.json").read_text())
    k = anomaly["_gmm_component_selection"]["chosen_k"]
    gmm = GaussianMixtureDetector(k, CFG["anomaly"]["covariance_regularization"], SEED).fit(scaler.transform(benign[feats]))
    sv = gmm.log_density(scaler.transform(va[feats]))
    gmm.log_epsilon = threshold_at_fpr(sv[va["IsAttack"].to_numpy() == 0], 0.01)
    st = gmm.log_density(scaler.transform(te[feats]))
    out["gaussian_mixture_k"] = k
    out["gaussian_test_at_1pct_val_fpr"] = detection_metrics(te["IsAttack"].to_numpy(), st < gmm.log_epsilon, -st)

    cleaner = joblib.load(MODELS / "preprocessing" / "cleaner_split.joblib")
    gain = getattr(clf, "feature_importances_", np.ones(len(feats)))
    importances = pd.Series(gain, index=feats, dtype=float).sort_values(ascending=False)
    host = MODELS / "botnet" / "host_model.joblib"
    host_obj = joblib.load(host) if host.exists() else None
    save_bundle(MODELS / "deployment", cleaner=cleaner, classifier=clf, classifier_features=feats,
                class_names=classes, gaussian=gmm, gaussian_scaler=scaler, gaussian_features=feats,
                importances=(importances / importances.sum()).to_dict(),
                host_model=host_obj["model"] if host_obj else None,
                host_model_features=host_obj["features"] if host_obj else None,
                meta={"classifier_key": a.model, "classifier_params": rec["params"],
                      "feature_policy": "selected features that passed the CICFlowMeter parity check",
                      "dropped_non_parity_features": dropped, "gaussian": f"GaussianMixture(K={k})",
                      "gaussian_operating_point": "1% FPR on benign validation flows",
                      "val_metrics": out["classifier_val"]})
    (METRICS / "deployment_model_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    LOG.info("deployment classifier test macroF1=%.4f FPR=%.4f | GMM test recall=%.4f FPR=%.4f",
             out["classifier_test"]["f1_macro"], out["classifier_test"]["fpr"],
             out["gaussian_test_at_1pct_val_fpr"]["recall"], out["gaussian_test_at_1pct_val_fpr"]["fpr"])


if __name__ == "__main__":
    main()
