"""Beaconing (periodic call-home) detection.

For every (source IP, destination IP, destination port) pair with enough
connections:

1. sort connection start times;
2. compute inter-connection intervals dt_i = t_{i+1} - t_i;
3. compute mean mu, standard deviation sigma and CV = sigma / mu.

A low CV indicates regular timing. Legitimate software also beacons (NTP,
update checks, keep-alives), so CV is never used alone:

* starts closer than ``timestamp_tolerance_s`` are merged (bursts,
  retransmissions and timestamp jitter are not new beacons);
* a minimum number of connections and a minimum observation span are required;
* the robust CV (1.4826 * MAD / median) is computed too and the smaller of the
  two must pass, so one missed beacon does not hide an otherwise regular
  series;
* the median period must exceed ``min_period_s`` (sub-second regularity is
  streaming or scanning rather than beaconing);
* well-known periodic service ports (DNS, NTP) are whitelisted.

``beacon_score`` in [0, 1] combines regularity, number of connections and
payload-size consistency. It is a ranking heuristic, not a probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["src_ip", "dst_ip", "dst_port", "n_connections", "span_s", "interval_mean", "interval_median",
           "interval_std", "cv", "robust_cv", "bytes_cv", "beacon_score", "is_beacon"]


def merge_bursts(times: np.ndarray, tolerance: float) -> np.ndarray:
    if len(times) == 0:
        return times
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] > tolerance:
            keep.append(t)
    return np.asarray(keep)


def interval_stats(times: np.ndarray) -> dict:
    iv = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    if len(iv) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "cv": np.inf, "robust_cv": np.inf}
    mean, std, median = float(iv.mean()), float(iv.std()), float(np.median(iv))
    mad = float(np.median(np.abs(iv - median)))
    return {"mean": mean, "median": median, "std": std,
            "cv": std / mean if mean > 0 else np.inf,
            "robust_cv": 1.4826 * mad / median if median > 0 else np.inf}


def detect_beaconing(flows: pd.DataFrame, min_connections: int = 8, cv_threshold: float = 0.25,
                     min_period_s: float = 2.0, min_span_s: float = 60.0, timestamp_tolerance_s: float = 1.0,
                     whitelist_ports=(53, 123)) -> pd.DataFrame:
    """One row per candidate (src, dst, port) with interval statistics and ``is_beacon``."""
    if flows.empty:
        return pd.DataFrame(columns=COLUMNS)
    keys = ["src_ip", "dst_ip", "dst_port"]
    counts = flows.groupby(keys, sort=False)["start_time"].transform("size")
    cand = flows.loc[counts >= min_connections]
    whitelist = set(whitelist_ports)
    rows = []
    for (src, dst, port), g in cand.groupby(keys, sort=False):
        times = merge_bursts(np.sort(g["start_time"].to_numpy(np.float64)), timestamp_tolerance_s)
        if len(times) < min_connections:
            continue
        st = interval_stats(times)
        b = g["bytes"].to_numpy(np.float64)
        bytes_cv = float(b.std() / b.mean()) if b.mean() > 0 else 0.0
        span = float(times[-1] - times[0])
        regular = min(st["cv"], st["robust_cv"])
        score = (max(0.0, 1 - regular / (2 * cv_threshold)) * 0.6
                 + min(1.0, len(times) / (4 * min_connections)) * 0.25
                 + max(0.0, 1 - min(bytes_cv, 1.0)) * 0.15)
        is_beacon = bool(regular <= cv_threshold and st["median"] >= min_period_s and span >= min_span_s
                         and int(port) not in whitelist)
        rows.append([src, dst, int(port), len(times), span, st["mean"], st["median"], st["std"],
                     st["cv"], st["robust_cv"], bytes_cv, score, is_beacon])
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("beacon_score", ascending=False).reset_index(drop=True)
