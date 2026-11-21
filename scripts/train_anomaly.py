"""Gaussian anomaly detection on CICIDS2018 with Isolation Forest / One-Class SVM baselines.

    python scripts/train_anomaly.py

* Detectors are fitted on BENIGN training flows only (no attack labels used):
  univariate Gaussian, multivariate Gaussian, Gaussian mixture (K chosen by
  validation PR-AUC), Isolation Forest, One-Class SVM.
* Thresholds are chosen on the validation split, at three operating points:
  - ``f1_optimal``: maximise F1 (attack = positive). With ~50% attacks in the
    evaluation splits this degenerates towards "flag everything", so it is
    reported for completeness only;
  - ``fpr_1pct`` / ``fpr_5pct``: the score quantile giving 1% / 5% false
    positives on BENIGN validation flows. These are the IDS-relevant points.
* Test metrics are computed once with those thresholds.
* Unknown-attack experiment (leave-one-family-out, LOFO): for each attack
  family F, thresholds are set at the same benign FPR on validation data that
  does not contain F, for (a) the deployed Gaussian model and (b) a supervised
  LightGBM trained on all training data except F. Recall on F in test measures
  detection of an attack never seen in training, at a matched false-positive rate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.preprocessing import LogScaler  # noqa: E402
from src.models.gaussian_detector import (GaussianMixtureDetector, MultivariateGaussianDetector,  # noqa: E402
                                          UnivariateGaussianDetector, select_epsilon, threshold_at_fpr)
from src.utils import visualization as viz  # noqa: E402
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import detection_metrics  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
PROCESSED = project_path("processed")
MODELS = project_path("models")
FIG = project_path("figures")
METRICS = project_path("metrics")
OUT = MODELS / "gaussian"
LOG = get_logger("anomaly", project_path("logs") / "anomaly.log")
BENIGN_FIT_ROWS = 300_000
GMM_COMPONENTS = [2, 4, 8, 16]
SLUGS = {"Gaussian (univariate)": "gaussian_uv", "Gaussian (multivariate)": "gaussian_mv",
         "Gaussian mixture": "gaussian_gmm", "Isolation Forest": "isolation_forest", "One-Class SVM": "ocsvm"}


def evaluate_threshold(st, yt, te, thr) -> dict:
    pred = (st < thr).astype(int)
    m = detection_metrics(yt, pred, -st)
    m["threshold"] = float(thr)
    m["confusion_matrix"] = confusion_matrix(yt, pred, labels=[0, 1]).tolist()
    m["recall_by_family"] = {fam: float(pred[(te["Family"] == fam).to_numpy()].mean())
                             for fam in sorted(te["Family"].unique()) if fam != "BENIGN"}
    return m


def main() -> None:
    features = json.loads((MODELS / "preprocessing" / "selected_features.json").read_text())
    tr = pd.read_parquet(PROCESSED / "split_train_clean.parquet")
    va = pd.read_parquet(PROCESSED / "split_val_clean.parquet")
    te = pd.read_parquet(PROCESSED / "split_test_clean.parquet")
    benign = tr[tr["IsAttack"] == 0]
    if len(benign) > BENIGN_FIT_ROWS:
        benign = benign.sample(BENIGN_FIT_ROWS, random_state=SEED)
    scaler = LogScaler("standard", log=CFG["anomaly"]["log_transform"]).fit(benign[features])
    xb = scaler.transform(benign[features])
    xv, yv = scaler.transform(va[features]), va["IsAttack"].to_numpy()
    xt, yt = scaler.transform(te[features]), te["IsAttack"].to_numpy()

    detectors, fit_times = {}, {}

    def timed_fit(name, make):
        t0 = time.perf_counter()
        detectors[name] = make()
        fit_times[name] = time.perf_counter() - t0

    timed_fit("Gaussian (univariate)", lambda: UnivariateGaussianDetector().fit(xb))
    timed_fit("Gaussian (multivariate)",
              lambda: MultivariateGaussianDetector(CFG["anomaly"]["covariance_regularization"]).fit(xb))
    # K for the mixture: validation PR-AUC (attack = positive), fitted on a 100k benign subsample
    sub = xb[np.random.default_rng(SEED).choice(len(xb), min(100_000, len(xb)), replace=False)]
    k_scores = {}
    for k in GMM_COMPONENTS:
        gm = GaussianMixtureDetector(k, CFG["anomaly"]["covariance_regularization"], SEED).fit(sub)
        k_scores[k] = float(average_precision_score(yv, -gm.log_density(xv)))
        LOG.info("GMM K=%d val PR-AUC=%.4f", k, k_scores[k])
    best_k = max(k_scores, key=k_scores.get)
    timed_fit("Gaussian mixture",
              lambda: GaussianMixtureDetector(best_k, CFG["anomaly"]["covariance_regularization"], SEED).fit(xb))
    timed_fit("Isolation Forest",
              lambda: IsolationForest(n_estimators=200, max_samples=256, random_state=SEED, n_jobs=-1).fit(xb))
    ocsvm_rows = 15_000
    timed_fit("One-Class SVM", lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(
        xb[np.random.default_rng(SEED).choice(len(xb), ocsvm_rows, replace=False)]))
    mvg = detectors["Gaussian (multivariate)"]
    LOG.info("MVG condition number %.3g, ridge %.1e; GMM K=%d", mvg.condition_number, mvg.added_jitter, best_k)

    def normality(name, x):  # higher = more normal
        d = detectors[name]
        if hasattr(d, "log_density"):
            return d.log_density(x)
        if name == "Isolation Forest":
            return d.score_samples(x)
        return d.decision_function(x)

    names = list(detectors)
    results, curves_roc, curves_pr, val_scores, test_scores = {}, {}, {}, {}, {}
    for name in names:
        sv = normality(name, xv)
        t0 = time.perf_counter()
        st = normality(name, xt)
        infer = time.perf_counter() - t0
        val_scores[name], test_scores[name] = sv, st
        res = {"fit_seconds": fit_times[name], "inference_us_per_flow": infer / len(xt) * 1e6,
               "fit_rows": int(ocsvm_rows if name == "One-Class SVM" else len(xb)),
               "val_pr_auc": float(average_precision_score(yv, -sv))}
        f1sel = select_epsilon(sv, yv, "f1")
        res["f1_optimal"] = evaluate_threshold(st, yt, te, f1sel["log_epsilon"])
        for tag, target in (("fpr_1pct", 0.01), ("fpr_5pct", 0.05)):
            res[tag] = evaluate_threshold(st, yt, te, threshold_at_fpr(sv[yv == 0], target))
        results[name] = res
        fpr, tpr, _ = roc_curve(yt, -st)
        prec, rec, _ = precision_recall_curve(yt, -st)
        curves_roc[name] = (fpr, tpr, res["fpr_1pct"]["roc_auc"])
        curves_pr[name] = (rec, prec, res["fpr_1pct"]["pr_auc"])
        LOG.info("%s: ROC-AUC=%.4f PR-AUC=%.4f | @1%%FPR recall=%.4f | @5%%FPR recall=%.4f | F1-opt recall=%.3f fpr=%.3f",
                 name, res["fpr_1pct"]["roc_auc"], res["fpr_1pct"]["pr_auc"], res["fpr_1pct"]["recall"],
                 res["fpr_5pct"]["recall"], res["f1_optimal"]["recall"], res["f1_optimal"]["fpr"])
    results["_gmm_component_selection"] = {"val_pr_auc_by_k": k_scores, "chosen_k": best_k}
    viz.curves_plot(curves_roc, "roc", "Anomaly detectors, ROC (test, benign vs. attack)", FIG / "anomaly_roc.png")
    viz.curves_plot(curves_pr, "pr", "Anomaly detectors, precision-recall (test)", FIG / "anomaly_pr.png")

    # deployed density model: Gaussian (multivariate) vs mixture, by validation PR-AUC
    deployed = max(["Gaussian (multivariate)", "Gaussian mixture"], key=lambda n: results[n]["val_pr_auc"])
    fams = ["BENIGN"] + sorted(f for f in te["Family"].unique() if f != "BENIGN")
    dist = {fam: -test_scores[deployed][(te["Family"] == fam).to_numpy()] for fam in fams}
    viz.score_distribution(dist, -results[deployed]["fpr_1pct"]["threshold"],
                           f"{deployed}: anomaly score by traffic family (test); threshold = 1% val FPR",
                           FIG / "anomaly_score_distribution.png")
    for name in names:
        cm = np.array(results[name]["fpr_1pct"]["confusion_matrix"])
        viz.confusion_matrix_plot(cm, ["BENIGN", "ATTACK"], f"{name} @ 1% validation FPR (test)",
                                  FIG / f"anomaly_cm_{SLUGS[name]}.png")

    # ---------------- unknown-attack (leave-one-family-out) at matched FPR
    import lightgbm as lgb
    families = sorted(f for f in tr["Family"].unique() if f != "BENIGN")
    ben_te = (te["Family"] == "BENIGN").to_numpy()
    rows = []
    for fam in families:
        keep_va = (va["Family"] != fam).to_numpy()
        fam_te = (te["Family"] == fam).to_numpy()
        parts = [tr[tr["Family"] == "BENIGN"].sample(200_000, random_state=SEED)]
        for other in families:
            if other != fam:
                g = tr[tr["Family"] == other]
                parts.append(g.sample(min(len(g), 60_000), random_state=SEED))
        train_sup = pd.concat(parts)
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, random_state=SEED, verbose=-1,
                                 min_sum_hessian_in_leaf=1.0).fit(train_sup[features], train_sup["IsAttack"])
        p_val = clf.predict_proba(va.loc[keep_va, features])[:, 1]
        p_te = clf.predict_proba(te[features])[:, 1]
        yv_keep = yv[keep_va]
        for target in (0.01, 0.05):
            g_thr = threshold_at_fpr(val_scores[deployed][keep_va][yv_keep == 0], target)
            g_pred = test_scores[deployed] < g_thr
            s_thr = float(np.quantile(p_val[yv_keep == 0], 1 - target))
            s_pred = p_te > s_thr
            rows.append({"held_out_family": fam, "target_val_fpr": target, "test_rows": int(fam_te.sum()),
                         "gaussian_model": deployed,
                         "gaussian_recall_unknown": float(g_pred[fam_te].mean()),
                         "gaussian_test_fpr": float(g_pred[ben_te].mean()),
                         "supervised_recall_unknown": float(s_pred[fam_te].mean()),
                         "supervised_test_fpr": float(s_pred[ben_te].mean()),
                         "either_recall_unknown": float((g_pred | s_pred)[fam_te].mean()),
                         "either_test_fpr": float((g_pred | s_pred)[ben_te].mean())})
            LOG.info("LOFO %s @%.0f%% FPR: gaussian %.3f | supervised %.3f | either %.3f", fam, target * 100,
                     rows[-1]["gaussian_recall_unknown"], rows[-1]["supervised_recall_unknown"],
                     rows[-1]["either_recall_unknown"])
    pd.DataFrame(rows).to_csv(METRICS / "unknown_attack_lofo.csv", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("Gaussian (univariate)", "Gaussian (multivariate)", "Gaussian mixture"):
        detectors[name].log_epsilon = results[name]["fpr_1pct"]["threshold"]
    joblib.dump({"detector": detectors[deployed], "deployed_name": deployed,
                 "multivariate": detectors["Gaussian (multivariate)"], "univariate": detectors["Gaussian (univariate)"],
                 "mixture": detectors["Gaussian mixture"], "scaler": scaler, "features": features,
                 "operating_point": "1% false-positive rate on benign validation flows",
                 "isolation_forest": detectors["Isolation Forest"],
                 "iso_threshold": results["Isolation Forest"]["fpr_1pct"]["threshold"]}, OUT / "gaussian.joblib")
    results["_deployed"] = {"model": deployed, "operating_point": "fpr_1pct"}
    (METRICS / "anomaly_results.json").write_text(json.dumps(results, indent=2, default=float))
    LOG.info("saved %s (epsilon at 1%% validation FPR) to %s", deployed, OUT)


if __name__ == "__main__":
    main()
