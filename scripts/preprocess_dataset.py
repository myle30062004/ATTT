"""CICIDS2018 preprocessing pipeline (run stages in order or all at once).

    python scripts/preprocess_dataset.py --stage all
    python scripts/preprocess_dataset.py --stage inspect|sample|split|clean|select|imbalance

Outputs: reports/metrics/*.json|csv, data/interim/cicids2018_sample.parquet,
data/processed/*.parquet, models/preprocessing/*.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.dataset_loader import build_cicids_sample, inspect_cicids  # noqa: E402
from src.data_processing.preprocessing import (FlowCleaner, aggregate_rankings, correlation_filter,  # noqa: E402
                                               mutual_information, resample, stratified_split, temporal_split)
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import classification_metrics  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
METRICS = project_path("metrics")
FIG = project_path("figures")
INTERIM = project_path("interim")
PROCESSED = project_path("processed")
PREP_DIR = project_path("models") / "preprocessing"
LOG = get_logger("preprocess", project_path("logs") / "preprocess.log")
SAMPLE_PATH = INTERIM / "cicids2018_sample.parquet"
LABEL_COLS = ["FineLabel", "Family", "IsAttack", "Day", "Timestamp", "row_hash"]


def stage_inspect() -> dict:
    t0 = time.time()
    summary = inspect_cicids(project_path("raw_cicids"), METRICS, CFG["cicids"]["chunksize"], log=LOG.info)
    LOG.info("inspection done in %.0fs: %s rows, %s duplicates", time.time() - t0, f"{summary['total_rows']:,}",
             f"{summary['duplicate_rows_features_and_label']:,}")
    return summary


def stage_sample() -> pd.DataFrame:
    inspection = json.loads((METRICS / "cicids_inspection.json").read_text())
    c = CFG["cicids"]
    sample = build_cicids_sample(project_path("raw_cicids"), inspection, SAMPLE_PATH, c["max_rows_per_attack_label"],
                                 c["max_benign_rows"], seed=SEED, chunksize=c["chunksize"], log=LOG.info)
    LOG.info("working sample: %s rows\n%s", f"{len(sample):,}", sample["FineLabel"].value_counts().to_string())
    return sample


def drop_conflicting_vectors(sample: pd.DataFrame) -> pd.DataFrame:
    """Remove feature vectors that carry more than one label.

    Such rows are unlearnable (identical input, different target) and would put
    the same feature vector into several splits. Removed rows are reported.
    """
    n_labels = sample.groupby("row_hash")["FineLabel"].transform("nunique")
    conflict = n_labels > 1
    combos = (sample.loc[conflict].groupby("row_hash")["FineLabel"]
              .agg(lambda s: " | ".join(sorted(s))).value_counts())
    report = {"conflicting_vectors": int(sample.loc[conflict, "row_hash"].nunique()),
              "rows_removed": int(conflict.sum()), "rows_kept": int((~conflict).sum()),
              "removed_by_label": sample.loc[conflict, "FineLabel"].value_counts().to_dict(),
              "label_combinations": combos.to_dict()}
    (METRICS / "conflicting_labels_removed.json").write_text(json.dumps(report, indent=2))
    LOG.info("removed %d rows of %d label-conflicting feature vectors", report["rows_removed"],
             report["conflicting_vectors"])
    return sample.loc[~conflict].reset_index(drop=True)


def stage_split() -> None:
    sample = drop_conflicting_vectors(pd.read_parquet(SAMPLE_PATH))
    ratios = tuple(CFG["cicids"]["split"][k] for k in ("train", "val", "test"))
    split = stratified_split(sample, "FineLabel", ratios, SEED)
    for name in ("train", "val", "test"):
        getattr(split, name).to_parquet(PROCESSED / f"split_{name}.parquet", index=False)
    overlap = split.overlap_check("row_hash")
    split.distribution("FineLabel").to_csv(METRICS / "split_distribution_fine.csv")
    split.distribution("Family").to_csv(METRICS / "split_distribution_family.csv")
    tsplit = temporal_split(sample, "FineLabel", "Timestamp", ratios)
    for name in ("train", "val", "test"):
        getattr(tsplit, name).to_parquet(PROCESSED / f"temporal_{name}.parquet", index=False)
    tsplit.distribution("Family").to_csv(METRICS / "temporal_split_distribution_family.csv")
    config = {"method": split.method, "ratios": ratios, "seed": SEED, "sizes": {n: len(getattr(split, n)) for n in
              ("train", "val", "test")}, "row_hash_overlap": overlap,
              "temporal_method": tsplit.method, "temporal_overlap": tsplit.overlap_check("row_hash"),
              "temporal_sizes": {n: len(getattr(tsplit, n)) for n in ("train", "val", "test")}}
    (METRICS / "split_config.json").write_text(json.dumps(config, indent=2))
    LOG.info("split config: %s", config)


def stage_temporal() -> None:
    """Regenerate only the chronological split (and its cleaned files)."""
    import joblib
    sample = drop_conflicting_vectors(pd.read_parquet(SAMPLE_PATH))
    ratios = tuple(CFG["cicids"]["split"][k] for k in ("train", "val", "test"))
    tsplit = temporal_split(sample, "FineLabel", "Timestamp", ratios)
    for name in ("train", "val", "test"):
        getattr(tsplit, name).to_parquet(PROCESSED / f"temporal_{name}.parquet", index=False)
    tsplit.distribution("Family").to_csv(METRICS / "temporal_split_distribution_family.csv")
    cleaner = FlowCleaner().fit(tsplit.train)
    for name in ("train", "val", "test"):
        df = getattr(tsplit, name)
        x = cleaner.transform(df).astype(np.float32).reset_index(drop=True)
        pd.concat([x, df[LABEL_COLS].reset_index(drop=True)], axis=1).to_parquet(
            PROCESSED / f"temporal_{name}_clean.parquet", index=False)
    joblib.dump(cleaner, PREP_DIR / "cleaner_temporal.joblib")
    cfg_path = METRICS / "split_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temporal_overlap"] = tsplit.overlap_check("row_hash")
    cfg["temporal_time_ranges_utc"] = {n: [str(pd.to_datetime(getattr(tsplit, n)["Timestamp"].min(), unit="s")),
                                           str(pd.to_datetime(getattr(tsplit, n)["Timestamp"].max(), unit="s"))]
                                       for n in ("train", "val", "test")}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    LOG.info("temporal split regenerated: %s", cfg["temporal_overlap"])


def _load_split(prefix: str = "split"):
    return [pd.read_parquet(PROCESSED / f"{prefix}_{n}.parquet") for n in ("train", "val", "test")]


def stage_clean() -> None:
    for prefix in ("split", "temporal"):
        train, val, test = _load_split(prefix)
        cleaner = FlowCleaner().fit(train)
        for name, df in zip(("train", "val", "test"), (train, val, test)):
            x = cleaner.transform(df).astype(np.float32).reset_index(drop=True)
            x = pd.concat([x, df[LABEL_COLS].reset_index(drop=True)], axis=1)
            x.to_parquet(PROCESSED / f"{prefix}_{name}_clean.parquet", index=False)
        PREP_DIR.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(cleaner, PREP_DIR / f"cleaner_{prefix}.joblib")
        if prefix == "split":
            raw = train
            report = {
                "dropped_constant_columns": cleaner.dropped_constant,
                "kept_columns": cleaner.kept_columns,
                "n_kept": len(cleaner.kept_columns),
                "train_cells_pos_inf": int(np.isposinf(raw[cleaner.feature_columns].to_numpy()).sum()),
                "train_cells_nan": int(raw[cleaner.feature_columns].isna().to_numpy().sum()),
                "train_negative_time_cells": int((raw[[c for c in cleaner.feature_columns if "IAT" in c or c == "Flow Duration"]] < 0).to_numpy().sum()),
            }
            (METRICS / "cleaning_report.json").write_text(json.dumps(report, indent=2))
            LOG.info("cleaning: dropped constant %s; kept %d", cleaner.dropped_constant, len(cleaner.kept_columns))


def _subsample(df: pd.DataFrame, n: int, by: str = "FineLabel") -> pd.DataFrame:
    """Stratified subsample keeping every row of small classes."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g if len(g) * frac < 200 else g.sample(frac=frac, random_state=SEED) for _, g in df.groupby(by)]
    return pd.concat(parts).reset_index(drop=True)


def stage_select() -> None:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from src.utils import visualization as viz

    train = pd.read_parquet(PROCESSED / "split_train_clean.parquet")
    val = pd.read_parquet(PROCESSED / "split_val_clean.parquet")
    features = [c for c in train.columns if c not in LABEL_COLS]
    families = sorted(train["Family"].unique(), key=lambda f: (f != "BENIGN", f))
    fam_index = {f: i for i, f in enumerate(families)}
    sub = _subsample(train, 300_000)
    y_sub = sub["Family"].map(fam_index).to_numpy()
    x_sub = sub[features]

    vt = VarianceThreshold(CFG["features"]["variance_threshold"]).fit(x_sub)
    low_var = [f for f, keep in zip(features, vt.get_support()) if not keep]
    candidates = [f for f in features if f not in low_var]
    LOG.info("variance filter removed %s", low_var)

    LOG.info("mutual information ...")
    mi = mutual_information(x_sub[candidates], y_sub, SEED)
    LOG.info("random forest importance ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=SEED,
                                class_weight="balanced_subsample").fit(x_sub[candidates], y_sub)
    rf_imp = pd.Series(rf.feature_importances_, index=candidates)
    LOG.info("lightgbm / xgboost importance ...")
    lg = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, random_state=SEED, verbose=-1, min_sum_hessian_in_leaf=1.0,
                            importance_type="gain").fit(x_sub[candidates], y_sub)
    lgb_imp = pd.Series(lg.feature_importances_, index=candidates)
    xg = XGBClassifier(n_estimators=200, max_depth=8, tree_method="hist", random_state=SEED,
                       importance_type="gain").fit(x_sub[candidates], y_sub)
    xgb_imp = pd.Series(xg.feature_importances_, index=candidates)
    ranking = aggregate_rankings({"mutual_info": mi, "rf_importance": rf_imp,
                                  "lgbm_gain": lgb_imp / lgb_imp.sum(), "xgb_gain": xgb_imp})
    ranking.to_csv(METRICS / "feature_ranking.csv")

    priority = -ranking["mean_rank"]
    kept_after_corr, removed = correlation_filter(x_sub[candidates], CFG["features"]["correlation_threshold"], priority)
    LOG.info("correlation filter removed %d features", len(removed))
    corr_top = x_sub[ranking.index[:35].tolist()].corr()
    viz.correlation_heatmap(corr_top, "Pearson correlation, top-35 ranked features (train)",
                            FIG / "fs_correlation_heatmap.png")
    viz.importance_plot(ranking.set_index(ranking.index)["rf_importance"], "Random Forest importance (train subsample)",
                        FIG / "fs_rf_importance.png")
    viz.importance_plot(mi, "Mutual information with family label (train subsample)", FIG / "fs_mutual_information.png")

    # Leakage / shortcut audit: how well does each feature alone separate the families?
    single = {}
    y_val = val["Family"].map(fam_index).to_numpy()
    for f in candidates:
        tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(x_sub[[f]], y_sub)
        single[f] = classification_metrics(y_val, tree.predict(val[[f]]), None, families)["f1_macro"]
    single = pd.Series(single).sort_values(ascending=False)
    single.to_csv(METRICS / "single_feature_macro_f1.csv", header=["val_macro_f1"])

    # k-sweep: choose the smallest feature set whose validation macro-F1 is within 0.002 of the best.
    ordered = [f for f in ranking.index if f in kept_after_corr]
    sweep = []
    y_train_sub = y_sub
    for k in [5, 10, 15, 20, 25, 30, 40, len(ordered)]:
        k = min(k, len(ordered))
        cols = ordered[:k]
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, random_state=SEED, verbose=-1, min_sum_hessian_in_leaf=1.0)
        m.fit(x_sub[cols], y_train_sub)
        met = classification_metrics(y_val, m.predict(val[cols]), m.predict_proba(val[cols]), families)
        sweep.append({"k": k, "val_f1_macro": met["f1_macro"], "val_f1_weighted": met["f1_weighted"],
                      "val_fpr": met["fpr"]})
        LOG.info("k=%d macroF1=%.4f fpr=%.5f", k, met["f1_macro"], met["fpr"])
        if k == len(ordered):
            break
    sweep_df = pd.DataFrame(sweep).drop_duplicates("k")
    sweep_df.to_csv(METRICS / "feature_k_sweep.csv", index=False)
    best = sweep_df["val_f1_macro"].max()
    chosen_k = int(sweep_df.loc[sweep_df["val_f1_macro"] >= best - 0.002, "k"].min())
    selected = ordered[:chosen_k]
    result = {"families": families, "all_candidates": candidates, "low_variance_removed": low_var,
              "correlation_removed": removed, "kept_after_correlation": kept_after_corr,
              "ranked_after_correlation": ordered, "k_sweep": sweep, "chosen_k": chosen_k,
              "selected_features": selected,
              "single_feature_top10": single.head(10).to_dict()}
    (METRICS / "feature_selection.json").write_text(json.dumps(result, indent=2))
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    (PREP_DIR / "selected_features.json").write_text(json.dumps(selected, indent=2))
    (PREP_DIR / "family_classes.json").write_text(json.dumps(families, indent=2))
    LOG.info("selected %d features: %s", chosen_k, selected)


def stage_imbalance() -> None:
    import lightgbm as lgb

    train = pd.read_parquet(PROCESSED / "split_train_clean.parquet")
    val = pd.read_parquet(PROCESSED / "split_val_clean.parquet")
    selected = json.loads((PREP_DIR / "selected_features.json").read_text())
    families = json.loads((PREP_DIR / "family_classes.json").read_text())
    fam_index = {f: i for i, f in enumerate(families)}
    sub = _subsample(train, 300_000)
    x, y = sub[selected], sub["Family"].map(fam_index).to_numpy()
    xv, yv = val[selected], val["Family"].map(fam_index).to_numpy()
    counts = pd.Series(y).value_counts().sort_index()
    rows = []
    strategies = {
        "none": dict(strategy="none", class_weight=None),
        "class_weight_balanced": dict(strategy="none", class_weight="balanced"),
        "random_undersample": dict(strategy="undersample", class_weight=None),
        "smote": dict(strategy="smote", class_weight=None),
        "smote_plus_undersample": dict(strategy="smote_under", class_weight=None),
    }
    for name, s in strategies.items():
        t0 = time.time()
        majority_cap = int(counts.drop(fam_index["BENIGN"]).max())
        xr, yr = resample(x, y, s["strategy"], SEED, majority_cap=majority_cap, smote_target=20_000)
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, random_state=SEED, verbose=-1, min_sum_hessian_in_leaf=1.0,
                               class_weight=s["class_weight"])
        m.fit(xr, yr)
        met = classification_metrics(yv, m.predict(xv), m.predict_proba(xv), families)
        per_class_recall = {}
        pred = m.predict(xv)
        for f, i in fam_index.items():
            mask = yv == i
            per_class_recall[f] = float((pred[mask] == i).mean()) if mask.any() else None
        rows.append({"strategy": name, "train_rows": int(len(yr)), "fit_seconds": round(time.time() - t0, 1),
                     **{k: met[k] for k in ("precision_macro", "recall_macro", "f1_macro", "f1_weighted", "fpr")},
                     **{f"recall_{k}": v for k, v in per_class_recall.items()}})
        LOG.info("imbalance %s: macroF1=%.4f fpr=%.5f", name, met["f1_macro"], met["fpr"])
    pd.DataFrame(rows).to_csv(METRICS / "imbalance_comparison.csv", index=False)


STAGES = {"inspect": stage_inspect, "sample": stage_sample, "split": stage_split, "clean": stage_clean,
          "select": stage_select, "imbalance": stage_imbalance, "temporal": stage_temporal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = parser.parse_args()
    stages = [s for s in STAGES if s != "temporal"] if args.stage == "all" else [args.stage]
    for name in stages:
        LOG.info("=== stage %s ===", name)
        t0 = time.time()
        STAGES[name]()
        LOG.info("stage %s finished in %.0fs", name, time.time() - t0)


if __name__ == "__main__":
    main()
