"""CSE-CIC-IDS2018 and CTU-13 loaders.

CICIDS2018 is ~16M rows over 10 daily CSVs, too large for 16 GB RAM, so it is
handled in two streaming passes:

1. :func:`inspect_cicids` reads every row once and records schema, label,
   missing/infinite value, statistical and duplicate information.
2. :func:`build_cicids_sample` draws a working set *over unique rows*: the
   keep/drop decision is a deterministic function of the row hash, so every
   copy of a duplicated flow gets the same decision and duplicates are not
   over-represented; copies are then collapsed to one.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .feature_extractor import CIC_FEATURE_COLUMNS

LABEL_COLUMN = "Label"
IDENTIFIER_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

# Raw CICIDS2018 label -> normalised fine label
FINE_LABEL_NORMALISATION = {
    "benign": "Benign",
    "bot": "Bot",
    "ddos attack-hoic": "DDoS-HOIC",
    "ddos attack-loic-udp": "DDoS-LOIC-UDP",
    "ddos attacks-loic-http": "DDoS-LOIC-HTTP",
    "dos attacks-goldeneye": "DoS-GoldenEye",
    "dos attacks-hulk": "DoS-Hulk",
    "dos attacks-slowhttptest": "DoS-SlowHTTPTest",
    "dos attacks-slowloris": "DoS-Slowloris",
    "ftp-bruteforce": "BruteForce-FTP",
    "ssh-bruteforce": "BruteForce-SSH",
    "brute force -web": "Web-BruteForce",
    "brute force -xss": "Web-XSS",
    "sql injection": "Web-SQLInjection",
    "infilteration": "Infiltration",
    "infiltration": "Infiltration",
}

# Fine label -> attack family used for multi-class classification
FAMILY_OF = {
    "Benign": "BENIGN",
    "Bot": "Botnet",
    "DDoS-HOIC": "DDoS", "DDoS-LOIC-UDP": "DDoS", "DDoS-LOIC-HTTP": "DDoS",
    "DoS-GoldenEye": "DoS", "DoS-Hulk": "DoS", "DoS-SlowHTTPTest": "DoS", "DoS-Slowloris": "DoS",
    "BruteForce-FTP": "BruteForce", "BruteForce-SSH": "BruteForce",
    "Web-BruteForce": "WebAttack", "Web-XSS": "WebAttack", "Web-SQLInjection": "WebAttack",
    "Infiltration": "Infiltration",
}


def normalise_label(raw: str) -> str:
    key = re.sub(r"\s+", " ", str(raw).strip().lower())
    if key not in FINE_LABEL_NORMALISATION:
        raise KeyError(f"Unknown CICIDS2018 label: {raw!r}")
    return FINE_LABEL_NORMALISATION[key]


def day_from_filename(path: Path) -> str:
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", path.name)
    if not match:
        return path.stem
    d, m, y = match.groups()
    return f"{y}-{m}-{d}"


def cicids_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).glob("*.csv"))


_NUMERIC_TOKEN = r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$|^\s*[-+]?(Infinity|inf|NaN|nan)\s*$"


def _to_float(column) -> np.ndarray:
    """String column -> float64; malformed tokens (e.g. text, empty) become NaN."""
    import pyarrow as pa
    import pyarrow.compute as pc

    valid = pc.match_substring_regex(column, _NUMERIC_TOKEN)
    cleaned = pc.utf8_trim_whitespace(pc.if_else(valid, column, pa.scalar(None, pa.string())))
    return pc.cast(cleaned, pa.float64()).to_numpy(zero_copy_only=False)


def iter_cicids_chunks(path: Path, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    """Yield typed chunks: numeric CIC features (float64), ``Timestamp`` (epoch s),
    ``FineLabel``, ``Day``, ``RawLabel``.

    Uses pyarrow's streaming CSV reader with every column read as text (the
    files contain repeated header rows, so per-column type inference is unsafe).
    Repeated header rows are removed and counted in ``chunk.attrs['header_rows']``;
    malformed numeric tokens become NaN. ``chunksize`` sets the approximate number
    of rows per chunk (via the reader block size, ~ 600 bytes per row).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    day = day_from_filename(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_names = fh.readline().rstrip("\r\n").split(",")
    names = [c.strip() for c in raw_names]
    missing = [c for c in CIC_FEATURE_COLUMNS + [LABEL_COLUMN, "Timestamp"] if c not in names]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=max(1 << 20, chunksize * 600)),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in raw_names}),
    )
    extra = [c for c in names if c in IDENTIFIER_COLUMNS]
    for batch in reader:
        table = pa.Table.from_batches([batch]).rename_columns(names)
        label = pc.utf8_trim_whitespace(table.column(LABEL_COLUMN))
        header_mask = pc.equal(label, LABEL_COLUMN)
        n_header = int(pc.sum(header_mask).as_py() or 0)
        if n_header:
            keep = pc.invert(header_mask)
            table, label = table.filter(keep), label.filter(keep)
        if table.num_rows == 0:
            continue
        out = pd.DataFrame({col: _to_float(table.column(col)) for col in CIC_FEATURE_COLUMNS})
        ts = pd.to_datetime(pd.Series(table.column("Timestamp").to_pylist()), format=TIMESTAMP_FORMAT, errors="coerce")
        # explicit unit: pandas >= 3 parses to microsecond resolution, so integer
        # nanosecond arithmetic would be off by a factor of 1000
        seconds = ts.to_numpy(dtype="datetime64[s]").astype(np.int64)
        # The CSVs use a 12-hour clock without AM/PM (only hours 01-12 occur).
        # Captures ran during the working day, so hours 01-07 are afternoon/evening.
        seconds = np.where(ts.dt.hour.to_numpy() < 8, seconds + 12 * 3600, seconds)
        out["Timestamp"] = np.where(ts.notna().to_numpy(), seconds, -1).astype(np.int64)
        out["RawLabel"] = label.to_pylist()
        out["FineLabel"] = out["RawLabel"].map(normalise_label)
        out["Day"] = day
        out.attrs["header_rows"] = n_header
        out.attrs["extra_columns"] = extra
        yield out


def row_hash(frame: pd.DataFrame, columns: list[str] = CIC_FEATURE_COLUMNS) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(np.uint64)


def _hash_to_unit(hashes: np.ndarray, seed: int) -> np.ndarray:
    """Map row hashes to deterministic pseudo-uniform numbers in [0, 1)."""
    x = hashes ^ np.uint64((seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class _ColumnStats:
    def __init__(self, columns: list[str]) -> None:
        k = len(columns)
        self.columns = columns
        self.count = np.zeros(k, np.int64)
        self.nan = np.zeros(k, np.int64)
        self.pos_inf = np.zeros(k, np.int64)
        self.neg_inf = np.zeros(k, np.int64)
        self.negative = np.zeros(k, np.int64)
        self.zero = np.zeros(k, np.int64)
        self.total = np.zeros(k, np.float64)
        self.total_sq = np.zeros(k, np.float64)
        self.min = np.full(k, np.inf)
        self.max = np.full(k, -np.inf)

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.count += values.shape[0]
        self.nan += np.isnan(values).sum(0)
        self.pos_inf += np.isposinf(values).sum(0)
        self.neg_inf += np.isneginf(values).sum(0)
        clean = np.where(finite, values, 0.0)
        self.negative += (finite & (values < 0)).sum(0)
        self.zero += (finite & (values == 0)).sum(0)
        self.total += clean.sum(0)
        self.total_sq += (clean ** 2).sum(0)
        self.min = np.minimum(self.min, np.where(finite, values, np.inf).min(0))
        self.max = np.maximum(self.max, np.where(finite, values, -np.inf).max(0))

    def to_frame(self) -> pd.DataFrame:
        n_finite = self.count - self.nan - self.pos_inf - self.neg_inf
        mean = np.divide(self.total, n_finite, out=np.zeros_like(self.total), where=n_finite > 0)
        var = np.divide(self.total_sq, n_finite, out=np.zeros_like(self.total), where=n_finite > 0) - mean ** 2
        return pd.DataFrame({
            "feature": self.columns, "count": self.count, "nan": self.nan, "pos_inf": self.pos_inf,
            "neg_inf": self.neg_inf, "negative": self.negative, "zero": self.zero,
            "min": self.min, "max": self.max, "mean": mean, "std": np.sqrt(np.clip(var, 0, None)),
        })


def inspect_cicids(raw_dir: Path, out_dir: Path, chunksize: int = 500_000, log=print) -> dict:
    """Full-dataset streaming inspection. Writes CSV/JSON summaries into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _ColumnStats(CIC_FEATURE_COLUMNS)
    per_file: dict[str, dict] = {}
    label_counts: Counter = Counter()
    day_label = defaultdict(Counter)
    hashes_feat, hashes_full, label_codes = [], [], []
    label_index: dict[str, int] = {}
    medians_sample = []
    rng = np.random.default_rng(0)
    for path in cicids_files(raw_dir):
        info = {"rows": 0, "header_rows": 0, "bad_timestamp": 0, "extra_columns": [], "size_bytes": path.stat().st_size}
        for chunk in iter_cicids_chunks(path, chunksize):
            info["rows"] += len(chunk)
            info["header_rows"] += chunk.attrs["header_rows"]
            info["extra_columns"] = chunk.attrs["extra_columns"]
            info["bad_timestamp"] += int((chunk["Timestamp"] < 0).sum())
            values = chunk[CIC_FEATURE_COLUMNS].to_numpy()
            stats.update(values)
            label_counts.update(chunk["FineLabel"].tolist())
            day_label[chunk["Day"].iat[0]].update(chunk["FineLabel"].tolist())
            codes = chunk["FineLabel"].map(lambda l: label_index.setdefault(l, len(label_index))).to_numpy(np.int8)
            hf = row_hash(chunk)
            hashes_feat.append(hf)
            label_codes.append(codes)
            hashes_full.append(hf ^ (codes.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)))
            take = rng.random(len(chunk)) < 0.02
            medians_sample.append(chunk.loc[take, CIC_FEATURE_COLUMNS])
            log(f"  {path.name}: {info['rows']:,} rows")
        per_file[path.name] = info
    hf = np.concatenate(hashes_feat)
    hfull = np.concatenate(hashes_full)
    codes = np.concatenate(label_codes)
    n_total = len(hf)
    n_unique_full = len(np.unique(hfull))
    # conflicting labels: identical feature vectors carrying different labels
    order = np.argsort(hf, kind="stable")
    hs, cs = hf[order], codes[order]
    boundaries = np.flatnonzero(np.diff(hs)) + 1
    groups_start = np.concatenate([[0], boundaries])
    cmin = np.minimum.reduceat(cs, groups_start)
    cmax = np.maximum.reduceat(cs, groups_start)
    sizes = np.diff(np.concatenate([groups_start, [len(hs)]]))
    conflicting_vectors = int((cmin != cmax).sum())
    conflicting_rows = int(sizes[cmin != cmax].sum())
    inv_label = {v: k for k, v in label_index.items()}
    unique_per_label = {}
    for code, name in inv_label.items():
        unique_per_label[name] = int(len(np.unique(hfull[codes == code])))
    column_stats = stats.to_frame()
    median_frame = pd.concat(medians_sample)
    finite = median_frame.replace([np.inf, -np.inf], np.nan)
    column_stats["median_2pct_sample"] = finite.median().values
    column_stats["n_unique_2pct_sample"] = finite.nunique().values
    column_stats.to_csv(out_dir / "cicids_column_stats_raw.csv", index=False)
    summary = {
        "files": per_file,
        "total_rows": int(n_total),
        "n_feature_columns": len(CIC_FEATURE_COLUMNS),
        "label_counts": dict(label_counts.most_common()),
        "label_counts_by_day": {d: dict(c) for d, c in sorted(day_label.items())},
        "unique_rows_per_label": unique_per_label,
        "duplicate_rows_features_and_label": int(n_total - n_unique_full),
        "unique_rows_features_and_label": int(n_unique_full),
        "conflicting_label_feature_vectors": conflicting_vectors,
        "rows_in_conflicting_vectors": conflicting_rows,
        "total_nan_cells": int(stats.nan.sum()),
        "total_pos_inf_cells": int(stats.pos_inf.sum()),
        "total_neg_inf_cells": int(stats.neg_inf.sum()),
        "constant_columns": column_stats.loc[column_stats["min"] == column_stats["max"], "feature"].tolist(),
    }
    (out_dir / "cicids_inspection.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_cicids_sample(raw_dir: Path, inspection: dict, out_path: Path, max_per_attack: int,
                        max_benign: int, seed: int = 42, chunksize: int = 500_000, log=print) -> pd.DataFrame:
    """Draw the working dataset uniformly over *unique* rows of each fine label."""
    unique_counts = inspection["unique_rows_per_label"]
    rates = {}
    for label, n_unique in unique_counts.items():
        cap = max_benign if label == "Benign" else max_per_attack
        rates[label] = min(1.0, cap / max(1, n_unique))
    parts = []
    for path in cicids_files(raw_dir):
        for chunk in iter_cicids_chunks(path, chunksize):
            h_feat = row_hash(chunk)
            u = _hash_to_unit(h_feat, seed)
            keep = u < chunk["FineLabel"].map(rates).to_numpy()
            kept = chunk.loc[keep].copy()
            kept["row_hash"] = h_feat[keep]
            parts.append(kept)
        log(f"  sampled {path.name}")
    sample = pd.concat(parts, ignore_index=True)
    before = len(sample)
    sample = sample.drop_duplicates(subset=["row_hash", "FineLabel"]).reset_index(drop=True)
    sample["Family"] = sample["FineLabel"].map(FAMILY_OF)
    sample["IsAttack"] = (sample["Family"] != "BENIGN").astype(np.int8)
    sample.attrs["sampling_rates"] = rates
    sample.attrs["rows_before_dedup"] = before
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)
    (out_path.with_suffix(".json")).write_text(json.dumps({
        "sampling_rates": rates, "rows_before_dedup": before, "rows_after_dedup": len(sample),
        "seed": seed, "max_per_attack": max_per_attack, "max_benign": max_benign,
        "label_counts": sample["FineLabel"].value_counts().to_dict(),
    }, indent=2))
    return sample


# --------------------------------------------------------------------- CTU-13
CTU_BINETFLOW_DTYPES = {
    "Dur": "float64", "Proto": "string", "SrcAddr": "string", "Sport": "string", "Dir": "string",
    "DstAddr": "string", "Dport": "string", "State": "string", "sTos": "float64", "dTos": "float64",
    "TotPkts": "int64", "TotBytes": "int64", "SrcBytes": "int64", "Label": "string",
}


def load_ctu13_binetflow(path: Path) -> pd.DataFrame:
    """Load a CTU-13 ``.binetflow`` (Argus bidirectional flows with labels)."""
    df = pd.read_csv(path, dtype=CTU_BINETFLOW_DTYPES)
    df["StartTime"] = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df["Label"] = df["Label"].str.strip()
    df["LabelClass"] = np.select(
        [df["Label"].str.contains("Botnet", case=False), df["Label"].str.contains("Normal", case=False)],
        ["Botnet", "Normal"], default="Background")
    for col in ("Sport", "Dport"):
        df[col + "Num"] = df[col].map(_parse_port).astype("float64")
    return df


def _parse_port(value) -> float:
    """Argus writes ports as decimal, hex (``0x0303`` for ICMP type/code) or empty."""
    if value is None or value is pd.NA:
        return np.nan
    text = str(value).strip()
    try:
        return float(int(text, 16)) if text.lower().startswith("0x") else float(int(text))
    except ValueError:
        return np.nan
