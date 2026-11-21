"""Host-level behavioural features computed from flow records.

Input is a flow table with (at least) the columns ``src_ip, dst_ip, dst_port,
protocol, start_time, duration, packets, bytes`` and optionally
``syn_without_ack, failed``. The same function is used for CTU-13 binetflow
records, PCAP-reconstructed flows and the realtime sliding window, so host
features are identical in all three settings.

Feature definitions (per source host h within the window):

* ``unique_dst_ips`` / ``unique_dst_ports``: distinct peers / services contacted.
* ``total_connections``: number of flows initiated by h.
* ``connections_per_sec``: total_connections / observed span of h's activity.
* ``failed_connections``: flows that never got a reply (no backward packets,
  or Argus states such as ``S_``/``REQ``/``RST``) – typical of scanning.
* ``syn_without_ack``: TCP SYN packets sent without ACK (connection attempts).
* ``avg_duration``: mean flow duration (s).
* ``dst_port_entropy`` / ``dst_ip_entropy``: Shannon entropy (bits) of the
  destination distribution; high values = scanning/spreading, low = focused.
* ``conn_interval_mean`` / ``conn_interval_std``: mean/std of the time between
  consecutive flow starts of h (s).
* ``periodicity_score``: 1 - min(1, CV) of those intervals, CV = std/mean; close
  to 1 means very regular (machine-like) connection timing.
* ``repeated_connections``: flows to an (ip, port) h already contacted.
* ``distinct_protocols``: number of different L4 protocols used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOST_FEATURES = [
    "unique_dst_ips", "unique_dst_ports", "total_connections", "connections_per_sec",
    "failed_connections", "failed_ratio", "syn_without_ack", "avg_duration", "dst_port_entropy",
    "dst_ip_entropy", "conn_interval_mean", "conn_interval_std", "periodicity_score",
    "repeated_connections", "repeated_ratio", "distinct_protocols", "bytes_out_mean", "packets_out_mean",
]

REQUIRED = ["src_ip", "dst_ip", "dst_port", "protocol", "start_time", "duration", "packets", "bytes"]


def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True).to_numpy()
    if len(counts) <= 1:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def _intervals(times: np.ndarray) -> np.ndarray:
    times = np.sort(times)
    return np.diff(times) if len(times) > 1 else np.empty(0)


def host_features(flows: pd.DataFrame) -> pd.DataFrame:
    """One row per source IP with :data:`HOST_FEATURES` (empty frame if no flows)."""
    missing = [c for c in REQUIRED if c not in flows.columns]
    if missing:
        raise ValueError(f"flow table missing columns: {missing}")
    if flows.empty:
        return pd.DataFrame(columns=["src_ip"] + HOST_FEATURES)
    df = flows.copy()
    if "failed" not in df:
        df["failed"] = False
    if "syn_without_ack" not in df:
        df["syn_without_ack"] = 0
    df["pair"] = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str)
    rows = []
    for src, g in df.groupby("src_ip", sort=False):
        n = len(g)
        start = g["start_time"].to_numpy(np.float64)
        span = float(start.max() - start.min() + g["duration"].iloc[np.argmax(start)]) if n > 1 else float(g["duration"].iloc[0])
        iv = _intervals(start)
        iv_mean = float(iv.mean()) if len(iv) else 0.0
        iv_std = float(iv.std()) if len(iv) else 0.0
        cv = iv_std / iv_mean if iv_mean > 0 else np.inf
        repeated = int(n - g["pair"].nunique())
        failed = int(g["failed"].sum())
        rows.append({
            "src_ip": src,
            "unique_dst_ips": g["dst_ip"].nunique(),
            "unique_dst_ports": g["dst_port"].nunique(),
            "total_connections": n,
            "connections_per_sec": n / span if span > 0 else float(n),
            "failed_connections": failed,
            "failed_ratio": failed / n,
            "syn_without_ack": int(g["syn_without_ack"].sum()),
            "avg_duration": float(g["duration"].mean()),
            "dst_port_entropy": shannon_entropy(g["dst_port"]),
            "dst_ip_entropy": shannon_entropy(g["dst_ip"]),
            "conn_interval_mean": iv_mean,
            "conn_interval_std": iv_std,
            "periodicity_score": float(1.0 - min(1.0, cv)) if len(iv) >= 2 else 0.0,
            "repeated_connections": repeated,
            "repeated_ratio": repeated / n,
            "distinct_protocols": g["protocol"].nunique(),
            "bytes_out_mean": float(g["bytes"].mean()),
            "packets_out_mean": float(g["packets"].mean()),
        })
    return pd.DataFrame(rows, columns=["src_ip"] + HOST_FEATURES)


def windowed_host_features(flows: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Host features per fixed time window (column ``window`` = window start)."""
    if flows.empty:
        return pd.DataFrame(columns=["window", "src_ip"] + HOST_FEATURES)
    t0 = flows["start_time"].min()
    win = ((flows["start_time"] - t0) // window_s).astype(int)
    parts = []
    for w, g in flows.groupby(win, sort=True):
        hf = host_features(g)
        hf.insert(0, "window", t0 + w * window_s)
        parts.append(hf)
    return pd.concat(parts, ignore_index=True)


def flows_from_cic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt a PCAP-reconstructed CIC flow frame (with metadata columns) to the host-feature schema."""
    return pd.DataFrame({
        "src_ip": frame["Src IP"], "dst_ip": frame["Dst IP"], "dst_port": frame["Dst Port"].astype(int),
        "protocol": frame["Protocol"].astype(int), "start_time": frame["Start Time"].astype(float),
        "duration": frame["Flow Duration"].astype(float) / 1e6,
        "packets": frame["Tot Fwd Pkts"] + frame["Tot Bwd Pkts"],
        "bytes": frame["TotLen Fwd Pkts"] + frame["TotLen Bwd Pkts"],
        "failed": (frame["Tot Bwd Pkts"] == 0).to_numpy(),
        "syn_without_ack": frame.get("SYN Without ACK", 0),
    })


ARGUS_FAILED_STATES = ("S_", "REQ", "INT", "RST", "_RA", "S_RA", "URP", "URH")


def flows_from_binetflow(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a CTU-13 binetflow table to the host-feature schema."""
    state = df["State"].fillna("").astype(str)
    failed = state.str.contains("|".join(["S_$", "^REQ", "^INT", "RST", "URP", "URH", "^S_RA"]), regex=True)
    proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
    return pd.DataFrame({
        "src_ip": df["SrcAddr"].astype(str), "dst_ip": df["DstAddr"].astype(str),
        "dst_port": df["DportNum"].fillna(-1).astype(int),
        "protocol": df["Proto"].str.lower().map(proto_map).fillna(0).astype(int),
        "start_time": df["StartTime"].astype("int64") / 1e9,
        "duration": df["Dur"].astype(float),
        "packets": df["TotPkts"], "bytes": df["TotBytes"],
        "failed": failed.to_numpy(),
        "syn_without_ack": (state.str.startswith("S_") & ~state.str.contains("A")).astype(int).to_numpy(),
        "label": df["LabelClass"].to_numpy(),
    })
