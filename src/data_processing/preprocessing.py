"""Reusable preprocessing: cleaning, splitting, feature selection, scaling, imbalance.

Every statistic used to transform data (medians, clipping bounds, selected
features, scaler parameters) is learned from the *training* split only and
persisted, so the same transformation is applied to validation, test, PCAP and
live flows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .feature_extractor import CIC_FEATURE_COLUMNS

RATE_COLUMNS = ["Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s"]
TIME_COLUMNS = [c for c in CIC_FEATURE_COLUMNS if "IAT" in c or c in ("Flow Duration",)
                or c.startswith("Active") or c.startswith("Idle")]
# -1 is a legitimate sentinel ("no TCP window observed") in these columns.
SENTINEL_COLUMNS = ["Init Fwd Win Byts", "Init Bwd Win Byts"]


# ------------------------------------------------------------------ cleaning
@dataclass
class FlowCleaner:
    """Train-fitted cleaning of CICFlowMeter features.

    * ``+inf`` in rate columns (duration = 0 with traffic) -> train finite max.
    * ``NaN`` in rate columns (0 bytes / 0 s) -> 0.
    * negative durations/IATs (clock artefacts) -> 0; other negatives kept.
    * any other NaN -> train median.
    * columns constant on the training split are dropped.
    """

    feature_columns: list[str] = field(default_factory=lambda: list(CIC_FEATURE_COLUMNS))
    medians: dict = field(default_factory=dict)
    finite_max: dict = field(default_factory=dict)
    finite_min: dict = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)
    kept_columns: list[str] = field(default_factory=list)

    def _base_clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        x = frame.reindex(columns=self.feature_columns).astype(np.float64)
        if missing:
            x[missing] = np.nan  # imputed below; reported by callers via ``missing_columns``
        for col in TIME_COLUMNS:
            if col in x:
                x.loc[x[col] < 0, col] = 0.0
        return x

    def fit(self, frame: pd.DataFrame) -> "FlowCleaner":
        x = self._base_clean(frame)
        values = x.replace([np.inf, -np.inf], np.nan)
        self.medians = values.median().fillna(0.0).to_dict()
        self.finite_max = values.max().fillna(0.0).to_dict()
        self.finite_min = values.min().fillna(0.0).to_dict()
        cleaned = self._fill(x)
        nunique = cleaned.nunique()
        self.dropped_constant = nunique[nunique <= 1].index.tolist()
        self.kept_columns = [c for c in self.feature_columns if c not in self.dropped_constant]
        return self

    def _fill(self, x: pd.DataFrame) -> pd.DataFrame:
        for col in x.columns:
            series = x[col]
            if col in RATE_COLUMNS:
                series = series.replace(np.inf, self.finite_max[col]).replace(-np.inf, 0.0).fillna(0.0)
            else:
                series = series.replace(np.inf, self.finite_max[col]).replace(-np.inf, self.finite_min[col])
                series = series.fillna(self.medians[col])
            x[col] = series
        return x

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.kept_columns:
            raise RuntimeError("FlowCleaner must be fitted before transform")
        return self._fill(self._base_clean(frame))[self.kept_columns]

    def missing_columns(self, frame: pd.DataFrame) -> list[str]:
        return [c for c in self.kept_columns if c not in frame.columns]


# ------------------------------------------------------------------ splitting
@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: str

    def distribution(self, column: str) -> pd.DataFrame:
        table = pd.DataFrame({name: getattr(self, name)[column].value_counts()
                              for name in ("train", "val", "test")}).fillna(0).astype(int)
        table["total"] = table.sum(axis=1)
        return table.sort_values("total", ascending=False)

    def overlap_check(self, key: str = "row_hash") -> dict:
        tr, va, te = (set(getattr(self, n)[key]) for n in ("train", "val", "test"))
        return {"train_val": len(tr & va), "train_test": len(tr & te), "val_test": len(va & te)}


def stratified_split(frame: pd.DataFrame, stratify_col: str, ratios=(0.70, 0.15, 0.15),
                     seed: int = 42) -> SplitResult:
    """Random stratified 70/15/15 split. Labels with < 3 rows are put in train."""
    assert abs(sum(ratios) - 1.0) < 1e-9
    counts = frame[stratify_col].value_counts()
    tiny = frame[stratify_col].isin(counts[counts < 3].index)
    main, rest = frame.loc[~tiny], frame.loc[tiny]
    train, temp = train_test_split(main, train_size=ratios[0], stratify=main[stratify_col], random_state=seed)
    val_share = ratios[1] / (ratios[1] + ratios[2])
    val, test = train_test_split(temp, train_size=val_share, stratify=temp[stratify_col], random_state=seed)
    train = pd.concat([train, rest])
    return SplitResult(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True),
                       f"stratified_random(seed={seed}, by={stratify_col})")


def temporal_split(frame: pd.DataFrame, label_col: str, time_col: str = "Timestamp",
                   ratios=(0.70, 0.15, 0.15)) -> SplitResult:
    """Chronological split *within each label*: earliest 70% -> train, next 15% -> val, last 15% -> test.

    Tests whether a model trained on the early phase of an attack campaign (and
    earlier benign traffic) recognises its later phase, which a random split
    cannot show because neighbouring flows of the same session land in all sets.
    """
    parts = {"train": [], "val": [], "test": []}
    for _, group in frame.groupby(label_col, sort=False):
        group = group.sort_values(time_col, kind="stable")
        n = len(group)
        a = int(round(n * ratios[0]))
        b = int(round(n * (ratios[0] + ratios[1])))
        parts["train"].append(group.iloc[:a])
        parts["val"].append(group.iloc[a:b])
        parts["test"].append(group.iloc[b:])
    return SplitResult(*(pd.concat(parts[k]).reset_index(drop=True) for k in ("train", "val", "test")),
                       method=f"temporal_within_label(by={label_col}, time={time_col})")


# ------------------------------------------------------------ feature selection
def correlation_filter(x: pd.DataFrame, threshold: float, priority: pd.Series) -> tuple[list[str], list[dict]]:
    """Greedy removal of features with |Pearson r| > threshold; keeps the higher-priority feature."""
    corr = x.corr().abs().fillna(0.0)
    order = priority.reindex(x.columns).fillna(0).sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    removed: list[dict] = []
    for col in order:
        partner = next((k for k in kept if corr.at[col, k] > threshold), None)
        if partner is None:
            kept.append(col)
        else:
            removed.append({"feature": col, "redundant_with": partner, "abs_corr": float(corr.at[col, partner])})
    return [c for c in x.columns if c in kept], removed


def mutual_information(x: pd.DataFrame, y: np.ndarray, seed: int = 42, max_rows: int = 200_000) -> pd.Series:
    if len(x) > max_rows:
        idx = np.random.default_rng(seed).choice(len(x), max_rows, replace=False)
        x, y = x.iloc[idx], y[idx]
    mi = mutual_info_classif(x.to_numpy(np.float32), y, random_state=seed)
    return pd.Series(mi, index=x.columns).sort_values(ascending=False)


def aggregate_rankings(rankings: dict[str, pd.Series]) -> pd.DataFrame:
    """Average rank across importance methods (1 = most important)."""
    table = pd.DataFrame({name: s for name, s in rankings.items()})
    ranks = table.rank(ascending=False, method="average")
    table["mean_rank"] = ranks.mean(axis=1)
    return table.sort_values("mean_rank")


# ------------------------------------------------------------------- scaling
SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


class LogScaler:
    """sign(x)*log1p(|x|) followed by a fitted scaler.

    Flow features are heavy-tailed (bytes/s spans 0..1e9); the log compresses
    the range so MLP training and Gaussian density estimates are not dominated
    by a few extreme flows, without discarding those flows as outliers.
    """

    def __init__(self, kind: str = "standard", log: bool = True) -> None:
        self.kind = kind
        self.log = log
        self.scaler = SCALERS[kind]()
        self.columns: list[str] = []

    def _prep(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        return np.sign(arr) * np.log1p(np.abs(arr)) if self.log else arr

    def fit(self, x: pd.DataFrame) -> "LogScaler":
        self.columns = list(getattr(x, "columns", []))
        self.scaler.fit(self._prep(x))
        return self

    def transform(self, x) -> np.ndarray:
        if self.columns and isinstance(x, pd.DataFrame):
            x = x[self.columns]
        out = self.scaler.transform(self._prep(x))
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, x) -> np.ndarray:
        return self.fit(x).transform(x)


# ----------------------------------------------------------------- imbalance
def resample(x: pd.DataFrame, y: np.ndarray, strategy: str, seed: int = 42,
             majority_cap: int | None = None, smote_target: int | None = None):
    """Resample the *training* split only. Strategies: none, undersample, smote, smote_under."""
    if strategy == "none":
        return x, y
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    counts = pd.Series(y).value_counts()
    if strategy in ("undersample", "smote_under"):
        cap = majority_cap or int(counts.iloc[1]) if len(counts) > 1 else int(counts.iloc[0])
        target = {cls: min(int(n), cap) for cls, n in counts.items()}
        x, y = RandomUnderSampler(sampling_strategy=target, random_state=seed).fit_resample(x, y)
        counts = pd.Series(y).value_counts()
    if strategy in ("smote", "smote_under"):
        goal = smote_target or int(counts.median())
        target = {cls: max(int(n), goal) for cls, n in counts.items() if n < goal}
        k = max(1, min(5, int(counts.min()) - 1))
        x, y = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=seed).fit_resample(x, y)
    return x, y


# --------------------------------------------------------------- persistence
@dataclass
class PreprocessingArtifacts:
    cleaner: FlowCleaner
    selected_features: list[str]
    scaler: LogScaler | None
    label_mapping: dict
    meta: dict

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "preprocessing.joblib")
        (directory / "feature_list.json").write_text(json.dumps(self.selected_features, indent=2))
        (directory / "label_mapping.json").write_text(json.dumps(self.label_mapping, indent=2))

    @staticmethod
    def load(directory: Path) -> "PreprocessingArtifacts":
        return joblib.load(directory / "preprocessing.joblib")

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.cleaner.transform(frame)[self.selected_features]
