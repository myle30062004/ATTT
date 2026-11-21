"""Dataset loader (CICIDS CSV quirks) and preprocessing pipeline."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data_processing.dataset_loader import (FAMILY_OF, build_cicids_sample, inspect_cicids,
                                                iter_cicids_chunks, normalise_label)
from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS
from src.data_processing.preprocessing import (FlowCleaner, LogScaler, correlation_filter, resample,
                                               stratified_split, temporal_split)
from tests.helpers import synthetic_cic_frame


def _write_cicids_csv(path, n=60, with_header_rows=True):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 100, size=(n, len(CIC_FEATURE_COLUMNS))), columns=CIC_FEATURE_COLUMNS).astype(object)
    df.insert(2, "Timestamp", "14/02/2018 08:31:01")
    df["Label"] = ["Benign"] * (n - 20) + ["FTP-BruteForce"] * 20
    df.loc[0, "Flow Byts/s"] = "Infinity"
    df.loc[1, "Flow Pkts/s"] = "NaN"
    df.iloc[5] = df.iloc[4]                                   # exact duplicate row
    text = df.to_csv(index=False)
    if with_header_rows:
        header = text.splitlines()[0]
        lines = text.splitlines()
        lines.insert(10, header)                              # repeated header inside the file
        text = "\n".join(lines) + "\n"
    path.write_text(text)
    return df


def test_label_normalisation_and_family():
    assert normalise_label("  Infilteration ") == "Infiltration"
    assert normalise_label("DoS attacks-Hulk") == "DoS-Hulk"
    assert FAMILY_OF[normalise_label("Brute Force -XSS")] == "WebAttack"
    with pytest.raises(KeyError):
        normalise_label("PortScan")                           # not present in CICIDS2018


def test_chunks_drop_repeated_header_and_parse_values(tmp_path):
    _write_cicids_csv(tmp_path / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
    chunks = list(iter_cicids_chunks(tmp_path / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", 1000))
    df = chunks[0]
    assert len(df) == 60 and df.attrs["header_rows"] == 1
    assert np.isposinf(df["Flow Byts/s"]).sum() == 1 and df["Flow Pkts/s"].isna().sum() == 1
    assert df["Day"].iat[0] == "2018-02-14"
    assert df["Timestamp"].iat[0] == 1518597061          # 14/02/2018 08:31:01 as epoch seconds (morning)


def test_inspection_and_dedup_sampling(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_cicids_csv(raw / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
    summary = inspect_cicids(raw, tmp_path / "out", log=lambda *a: None)
    assert summary["total_rows"] == 60
    assert summary["duplicate_rows_features_and_label"] == 1
    assert summary["total_pos_inf_cells"] == 1
    sample = build_cicids_sample(raw, summary, tmp_path / "s.parquet", max_per_attack=1000, max_benign=1000,
                                 log=lambda *a: None)
    assert len(sample) == 59                                  # duplicate collapsed
    assert set(sample["Family"]) == {"BENIGN", "BruteForce"}
    meta = json.loads((tmp_path / "s.json").read_text())
    assert meta["rows_after_dedup"] == 59


def test_cleaner_inf_nan_negative_constant_and_missing_columns():
    df = synthetic_cic_frame(100, 50)
    df.loc[0, "Flow Byts/s"] = np.inf
    df.loc[1, "Flow Byts/s"] = np.nan
    df.loc[2, "Flow IAT Min"] = -5
    df.loc[3, "Bwd IAT Mean"] = np.nan
    cleaner = FlowCleaner().fit(df)
    out = cleaner.transform(df)
    assert "Fwd Byts/b Avg" in cleaner.dropped_constant and "Fwd Byts/b Avg" not in out
    assert np.isfinite(out.to_numpy()).all()
    assert out.loc[0, "Flow Byts/s"] == pytest.approx(df["Flow Byts/s"].replace(np.inf, np.nan).max())
    assert out.loc[1, "Flow Byts/s"] == 0.0
    assert out.loc[2, "Flow IAT Min"] == 0.0
    partial = df.drop(columns=["Flow Duration"])
    assert cleaner.missing_columns(partial) == ["Flow Duration"]
    assert np.isfinite(cleaner.transform(partial).to_numpy()).all()  # imputed with train median


def test_split_ratios_no_overlap_and_temporal():
    df = synthetic_cic_frame(700, 300)
    split = stratified_split(df, "FineLabel", (0.7, 0.15, 0.15), seed=1)
    n = len(df)
    assert abs(len(split.train) - 0.7 * n) <= 2 and abs(len(split.val) - 0.15 * n) <= 2
    assert split.overlap_check() == {"train_val": 0, "train_test": 0, "val_test": 0}
    dist = split.distribution("FineLabel")
    assert (dist.loc["DoS-Hulk", ["train", "val", "test"]] > 0).all()
    t = temporal_split(df, "FineLabel")
    for label, g in df.groupby("FineLabel"):
        assert t.train.loc[t.train.FineLabel == label, "Timestamp"].max() < t.test.loc[t.test.FineLabel == label, "Timestamp"].min()


def test_resampling_strategies_only_change_given_set():
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(1000, 4)), columns=list("abcd"))
    y = np.array([0] * 900 + [1] * 80 + [2] * 20)
    xu, yu = resample(x, y, "undersample", majority_cap=80)
    assert np.bincount(yu).tolist() == [80, 80, 20]
    xs, ys = resample(x, y, "smote", smote_target=200)
    assert np.bincount(ys).tolist() == [900, 200, 200]
    xn, yn = resample(x, y, "none")
    assert len(yn) == 1000


def test_correlation_filter_and_log_scaler():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    x = pd.DataFrame({"a": a, "b": a * 2 + 1e-6 * rng.normal(size=500), "c": rng.normal(size=500)})
    kept, removed = correlation_filter(x, 0.98, pd.Series({"a": 2, "b": 1, "c": 0}))
    assert kept == ["a", "c"] and removed[0]["feature"] == "b"
    sc = LogScaler("standard").fit(pd.DataFrame({"v": [0, 10, 1e9, -5]}))
    out = sc.transform(pd.DataFrame({"v": [0, 10, 1e9, -5]}))
    assert out.dtype == np.float32 and abs(out.mean()) < 1e-5
