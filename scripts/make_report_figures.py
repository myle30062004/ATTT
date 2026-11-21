"""Figures and tables for the written report (BAO_CAO). Validation split only for supervised models.

    python scripts/make_report_figures.py [--skip-predictions]

Outputs: reports/figures/report/*.png and reports/metrics/report_val_per_class.csv.
Every number comes from saved models, saved metrics or the validation split;
the test split is not read here.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import visualization as viz  # noqa: E402
from src.utils.visualization import CATEGORICAL, OTHER, TEXT, TEXT_SECONDARY, plt  # noqa: E402

OUT = ROOT / "reports" / "figures" / "report"
METRICS = ROOT / "reports" / "metrics"
MODELS = ROOT / "models"
EXP = ROOT / "reports" / "experiments"
NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost",
         "lgbm": "LightGBM", "cat": "CatBoost", "mlp": "MLP"}
PATHS = {"lr": "baselines/lr_multiclass", "dt": "baselines/dt_multiclass", "rf": "random_forest/multiclass",
         "xgb": "xgboost/multiclass", "lgbm": "lightgbm/multiclass", "cat": "catboost/multiclass",
         "mlp": "mlp/multiclass"}


def predictions() -> None:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    from src.models.mlp import MLPClassifier

    features = json.loads((MODELS / "preprocessing" / "selected_features.json").read_text())
    classes = json.loads((MODELS / "preprocessing" / "family_classes.json").read_text())
    val = pd.read_parquet(ROOT / "data" / "processed" / "split_val_clean.parquet", columns=features + ["Family"])
    y = val["Family"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    x = val[features].to_numpy(np.float32)
    rows = []
    for key, rel in PATHS.items():
        d = MODELS / rel
        if key == "mlp":
            model = MLPClassifier.load(d)
            scaler = joblib.load(d / "scaler.joblib")
            pred = model.predict(scaler.transform(val[features]))
        else:
            model = joblib.load(d / "model.joblib")
            if isinstance(model, dict):                       # baselines store {"model", "scaler"}
                xin = model["scaler"].transform(val[features]) if model.get("scaler") is not None else x
                pred = model["model"].predict(xin)
            else:
                pred = model.predict(x)
        pred = np.asarray(pred).reshape(-1).astype(int)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
        for i, c in enumerate(classes):
            rows.append({"model": NAMES[key], "class": c, "precision": p[i], "recall": r[i], "f1": f[i],
                         "support": int(s[i])})
        if key in ("lgbm", "rf"):
            cm = confusion_matrix(y, pred, labels=range(len(classes)))
            np.save(OUT / f"cm_val_{key}.npy", cm)
            viz.confusion_matrix_plot(cm, classes, f"{NAMES[key]} – ma trận nhầm lẫn (validation, chuẩn hoá theo hàng)",
                                      OUT / f"cm_val_{key}.png")
        print(key, "macroF1", round(float(np.mean(f)), 4), flush=True)
        del model
        gc.collect()
    pd.DataFrame(rows).to_csv(METRICS / "report_val_per_class.csv", index=False)


def per_class_heatmap() -> None:
    df = pd.read_csv(METRICS / "report_val_per_class.csv")
    order = [NAMES[k] for k in ("rf", "lgbm", "xgb", "cat", "dt", "mlp", "lr")]
    piv = df.pivot(index="model", columns="class", values="f1").loc[order]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    im = ax.imshow(piv.values, cmap=viz.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.55 else TEXT)
    ax.set_title("F1 theo từng lớp trên tập validation (bài toán 7 lớp)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="F1")
    viz.save(fig, OUT / "val_per_class_f1.png")


def model_comparison() -> None:
    rows = []
    for task in ("multiclass", "binary"):
        s = json.loads((EXP / f"training_summary_{task}.json").read_text())
        for k, r in s.items():
            v = r["val_metrics"]
            rows.append({"task": task, "model": NAMES[k], "f1_macro": v["f1_macro"], "fpr": v["fpr"]})
    df = pd.DataFrame(rows)
    order = [NAMES[k] for k in ("lr", "dt", "mlp", "cat", "xgb", "lgbm", "rf")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, metric, label in ((axes[0], "f1_macro", "Macro-F1 (cao hơn tốt hơn)"),
                              (axes[1], "fpr", "FPR – tỷ lệ BENIGN bị báo động (thấp hơn tốt hơn)")):
        for t_i, task in enumerate(("multiclass", "binary")):
            d = df[df.task == task].set_index("model").reindex(order)
            ypos = np.arange(len(order)) + (-0.2 if t_i == 0 else 0.2)
            ax.barh(ypos, d[metric], height=0.36, color=CATEGORICAL[t_i],
                    label="7 lớp" if task == "multiclass" else "Nhị phân", edgecolor="#fcfcfb", linewidth=1)
            for yy, v in zip(ypos, d[metric]):
                if pd.notna(v):
                    ax.text(v, yy, f" {v:.3f}", va="center", fontsize=7.5, color=TEXT_SECONDARY)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(label, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0.6, 1.0)
    axes[1].set_xlim(0, 0.45)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    fig.suptitle("So sánh mô hình có giám sát trên tập validation (266.305 flow)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    viz.save(fig, OUT / "val_model_comparison.png")


def imbalance_and_k() -> None:
    imb = pd.read_csv(METRICS / "imbalance_comparison.csv")
    names = {"none": "Không xử lý", "class_weight_balanced": "Class weight", "random_undersample": "Undersampling",
             "smote": "SMOTE", "smote_plus_undersample": "SMOTE + undersampling"}
    imb["name"] = imb["strategy"].map(names)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, col, title in ((axes[0], "f1_macro", "Macro-F1"), (axes[1], "fpr", "FPR (BENIGN)"),
                           (axes[2], "recall_Infiltration", "Recall Infiltration")):
        ax.barh(imb["name"], imb[col], color=CATEGORICAL[0], height=0.55)
        for yy, v in enumerate(imb[col]):
            ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(1.0, imb[col].max() * 1.25) if col != "fpr" else imb[col].max() * 1.35)
    axes[0].invert_yaxis()
    fig.suptitle("So sánh xử lý mất cân bằng (LightGBM, validation; chỉ resample tập train)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    viz.save(fig, OUT / "imbalance_comparison.png")

    k = pd.read_csv(METRICS / "feature_k_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(k["k"], k["val_f1_macro"], marker="o", color=CATEGORICAL[0], markersize=6)
    best = k["val_f1_macro"].max()
    ax.axhline(best - 0.002, color=OTHER, linestyle="--", linewidth=1)
    ax.text(5, best - 0.002, " ngưỡng: tốt nhất − 0,002", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.scatter([30], k.loc[k.k == 30, "val_f1_macro"], s=110, facecolor="none", edgecolor=CATEGORICAL[1],
               linewidth=2, zorder=3)
    ax.annotate("k = 30 được chọn", (30, float(k.loc[k.k == 30, "val_f1_macro"].iloc[0])), xytext=(34, 0.878),
                fontsize=9, color=TEXT, arrowprops={"arrowstyle": "-", "color": OTHER})
    ax.set_xlabel("Số đặc trưng k (theo thứ hạng tổng hợp)")
    ax.set_ylabel("Macro-F1 validation")
    ax.set_title("Chọn số đặc trưng k trên tập validation", loc="left")
    viz.save(fig, OUT / "feature_k_sweep.png")


def lofo() -> None:
    df = pd.read_csv(METRICS / "unknown_attack_lofo.csv")
    d = df[df.target_val_fpr == 0.05].set_index("held_out_family")
    fams = ["DoS", "Botnet", "DDoS", "WebAttack", "BruteForce", "Infiltration"]
    d = d.loc[fams]
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.barh(y - 0.2, d["gaussian_recall_unknown"], height=0.38, color=CATEGORICAL[0],
            label="Gaussian mixture (chỉ học BENIGN)", edgecolor="#fcfcfb")
    ax.barh(y + 0.2, d["supervised_recall_unknown"], height=0.38, color=CATEGORICAL[1],
            label="LightGBM không thấy họ này khi train", edgecolor="#fcfcfb")
    for yy, a, b in zip(y, d["gaussian_recall_unknown"], d["supervised_recall_unknown"]):
        ax.text(a, yy - 0.2, f" {a:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.text(b, yy + 0.2, f" {b:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(y, fams)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Recall trên họ tấn công bị giấu (cùng FPR ≈ 5% trên BENIGN)")
    ax.set_title("Tấn công chưa từng thấy (leave-one-family-out)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    viz.save(fig, OUT / "lofo_unknown_attack.png")


def architecture() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fill = {"in": "#eef4fc", "proc": "#f3f2ef", "model": "#fdf0e9", "out": "#ebf7f1"}
    edge = {"in": CATEGORICAL[0], "proc": OTHER, "model": CATEGORICAL[1], "out": CATEGORICAL[2]}

    def box(x, y, w, h, title, sub, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fill[kind], edgecolor=edge[kind], linewidth=1.4))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=TEXT)
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=7.8, color=TEXT_SECONDARY,
                linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=TEXT_SECONDARY, linewidth=1.2))

    box(1, 44, 22, 16, "Nguồn dữ liệu", "CSE-CIC-IDS2018 (CSV)\nCTU-13 (PCAP)\nCard mạng (Scapy)", "in")
    box(29, 44, 24, 16, "Giải mã gói & dựng flow", "decode_frame()\nFlowBuilder 2 chiều\n120 s timeout, FIN/RST", "proc")
    box(59, 44, 24, 16, "Làm sạch & đặc trưng", "FlowCleaner (fit trên train)\n30 đặc trưng chọn\n24 đặc trưng parity-safe", "proc")
    box(88, 44, 21, 16, "Đặc trưng host", "cửa sổ 300 s\n18 đặc trưng/host\nđồ thị liên lạc", "proc")
    box(1, 20, 25, 17, "Tầng 1: Gaussian mixture", "log p(x) < ε → bất thường\nε chọn trên validation\n(FPR 1% BENIGN)", "model")
    box(31, 20, 25, 17, "Tầng 2: LightGBM", "7 họ tấn công\nxác suất < 0,6 → Uncertain\nBENIGN + bất thường → Unknown", "model")
    box(61, 20, 22, 17, "Tầng 3: mức host", "quét cổng / quét host\nbeaconing (CV + MAD)\nRF host + đồ thị", "model")
    box(88, 20, 21, 17, "Topology", "centralized / P2P /\nkhông đủ bằng chứng\n(bỏ phiếu nhiều chỉ số)", "model")
    box(20, 1, 70, 12, "Quyết định & cảnh báo", "mức INFO → CRITICAL · gộp cảnh báo lặp 30 s · console / JSONL / CSV / SQLite", "out")
    arrow(23.6, 52, 28.4, 52); arrow(53.6, 52, 58.4, 52); arrow(83.6, 52, 87.4, 52)
    arrow(66, 43.5, 14, 37.6); arrow(71, 43.5, 43, 37.6); arrow(98, 43.5, 74, 37.6); arrow(98.5, 43.5, 98.5, 37.6)
    arrow(26.6, 28.5, 30.4, 28.5)
    arrow(43, 19.5, 45, 13.6); arrow(72, 19.5, 62, 13.6); arrow(98.5, 19.5, 80, 13.6)
    ax.text(28.5, 30.5, "", fontsize=7)
    viz.save(fig, OUT / "architecture.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-predictions", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    model_comparison()
    imbalance_and_k()
    lofo()
    if not a.skip_predictions:
        predictions()
    per_class_heatmap()


if __name__ == "__main__":
    main()
