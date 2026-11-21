"""Render every experiment result file into markdown tables (reports/results_tables.md).

    python scripts/make_results_tables.py

The documentation quotes these tables; nothing in them is typed by hand.
Missing result files are skipped with a note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
E = ROOT / "reports" / "experiments"
OUT = ROOT / "reports" / "results_tables.md"


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def table(df: pd.DataFrame, digits=4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def model_tables(parts: list[str]) -> None:
    for task in ("multiclass", "binary"):
        p = M / f"model_comparison_{task}.csv"
        if not p.exists():
            parts.append(section(f"Model comparison ({task})", "_not available_"))
            continue
        d = pd.read_csv(p)
        auc = ("roc_auc", "pr_auc") if task == "binary" else ("binary_view_roc_auc", "binary_view_pr_auc")
        t = pd.DataFrame({
            "Model": d["model"], "Accuracy": d["accuracy"], "Precision (macro)": d["precision_macro"],
            "Recall (macro)": d["recall_macro"], "Macro F1": d["f1_macro"], "Weighted F1": d["f1_weighted"],
            "FPR": d["fpr"], "FNR": d["fnr"], "ROC-AUC (attack vs benign)": d[auc[0]],
            "PR-AUC (attack vs benign)": d[auc[1]], "Train time (s)": d["train_seconds"].round(1),
            "Inference (µs/flow, batch)": d["inference_us_per_flow"].round(2),
            "Single-flow latency (ms)": d["single_flow_latency_ms"].round(3),
            "Model size (MB)": d["model_size_mb"].round(2)})
        if task == "multiclass":
            t["ROC-AUC OvR macro"] = d["roc_auc_ovr_macro"]
            t["PR-AUC OvR macro"] = d["pr_auc_ovr_macro"]
        parts.append(section(f"Model comparison – {task} (test set)", table(t)))
        g = pd.DataFrame({"Model": d["model"], "Train macro F1": d["train_f1_macro"],
                          "Validation macro F1": d["val_f1_macro"], "Test macro F1": d["f1_macro"],
                          "Train − Test": d["train_f1_macro"] - d["f1_macro"]})
        parts.append(section(f"Generalisation – {task}", table(g)))
        if task == "multiclass":
            rows = []
            for key, name in zip(d["key"], d["model"]):
                pc = M / f"per_class_multiclass_{key}.csv"
                if pc.exists():
                    x = pd.read_csv(pc).set_index("class")
                    rows.append(pd.Series({f"{c} F1": x.loc[c, "f1"] for c in x.index} |
                                          {f"{c} recall": x.loc[c, "recall"] for c in x.index}, name=name))
            if rows:
                pcdf = pd.DataFrame(rows)
                f1c = [c for c in pcdf.columns if c.endswith(" F1")]
                rc = [c for c in pcdf.columns if c.endswith(" recall")]
                parts.append(section("Per-class F1 (multiclass, test)", table(pcdf[f1c].reset_index(names="Model"))))
                parts.append(section("Per-class recall (multiclass, test)", table(pcdf[rc].reset_index(names="Model"))))


def anomaly_tables(parts: list[str]) -> None:
    p = M / "anomaly_results.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = []
    for name, r in d.items():
        if name.startswith("_"):
            continue
        for op in ("fpr_1pct", "fpr_5pct", "f1_optimal"):
            m = r[op]
            rows.append({"Detector": name, "Operating point": op, "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"], "FPR": m["fpr"], "ROC-AUC": m.get("roc_auc"),
                         "PR-AUC": m.get("pr_auc"), "Fit (s)": r["fit_seconds"],
                         "Inference (µs/flow)": r["inference_us_per_flow"]})
    parts.append(section("Anomaly detection (test; thresholds from validation)", table(pd.DataFrame(rows))))
    fam = []
    for name, r in d.items():
        if name.startswith("_"):
            continue
        fam.append(pd.Series(r["fpr_5pct"]["recall_by_family"], name=f"{name} @5% FPR"))
        fam.append(pd.Series(r["fpr_1pct"]["recall_by_family"], name=f"{name} @1% FPR"))
    parts.append(section("Anomaly detection – recall by attack family (test)",
                         table(pd.DataFrame(fam).reset_index(names="Detector / operating point"))))
    sel = d.get("_gmm_component_selection")
    if sel:
        parts.append(section("Gaussian mixture – number of components (validation PR-AUC)",
                             table(pd.DataFrame({"K": list(sel["val_pr_auc_by_k"]),
                                                 "Val PR-AUC": list(sel["val_pr_auc_by_k"].values())}))
                             + f"\n\nChosen K = {sel['chosen_k']}; deployed model: {d.get('_deployed', {}).get('model')}"))
    lofo = M / "unknown_attack_lofo.csv"
    if lofo.exists():
        parts.append(section("Unknown-attack experiment (leave-one-family-out, matched validation FPR)",
                             table(pd.read_csv(lofo))))


def misc_tables(parts: list[str]) -> None:
    for title, path, idx in [
        ("Imbalance strategies (validation, LightGBM)", M / "imbalance_comparison.csv", None),
        ("Feature-count sweep (validation, LightGBM)", M / "feature_k_sweep.csv", None),
        ("Scaler comparison for the MLP (validation)", E / "scaler_comparison.csv", None),
        ("Random vs temporal split (test)", E / "temporal_vs_random_split.csv", None),
        ("Feature ablation (LightGBM, test)", E / "feature_ablation.csv", None),
        ("Botnet detectors on CTU-13 host windows (test windows)", M / "botnet_detector_comparison.csv", 0),
        ("Topology simulation confusion (rows = truth)", M / "topology_simulation_confusion.csv", 0),
    ]:
        if path.exists():
            df = pd.read_csv(path, index_col=idx)
            if idx is not None:
                df = df.reset_index(names=df.index.name or "")
            keep = [c for c in df.columns if not str(c).startswith("recall_") or "imbalance" in str(path)]
            parts.append(section(title, table(df[keep])))
    rep = E / "repeats_summary.csv"
    if rep.exists():
        df = pd.read_csv(rep, header=[0, 1], index_col=[0, 1])
        out = pd.DataFrame(index=df.index)
        for metric in sorted({m for m, _ in df.columns}):
            out[metric] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(df[(metric, "mean")], df[(metric, "std")])]
        parts.append(section("Repeated runs (mean ± std over 3 seeds, test)", table(out.reset_index())))
    imp = M / "feature_importance_models.csv"
    if imp.exists():
        df = pd.read_csv(imp, index_col=0).head(15)
        parts.append(section("Feature importance (normalised, top 15 by mean)", table(df.reset_index(names="Feature"))))
    shap = M / "shap_lightgbm_multiclass.csv"
    if shap.exists():
        df = pd.read_csv(shap, index_col=0).head(15)
        parts.append(section("LightGBM mean |SHAP| (multiclass, top 15)", table(df.reset_index(names="Feature"))))
    lab = E / "lab_results.json"
    if lab.exists():
        d = json.loads(lab.read_text())
        parts.append(section("Lab scenarios", table(pd.DataFrame(d["scenarios"]).drop(columns=["modules"], errors="ignore"))
                             + f"\n\nBackground alerts: {d['background_alerts']} {d['background_alert_types']}"))


def main() -> None:
    parts = ["# Result tables (generated by scripts/make_results_tables.py)\n"]
    model_tables(parts)
    anomaly_tables(parts)
    misc_tables(parts)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
