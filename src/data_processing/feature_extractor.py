"""Flow -> CICFlowMeter-compatible feature vector.

Feature names, order and units (microseconds, bytes of L4 payload) follow the
CSE-CIC-IDS2018 CSV schema, so a model trained on the CSVs can score flows that
this project reconstructs from PCAP files or live traffic.

Known deviations from the original Java CICFlowMeter (documented in
docs/DATA_PREPROCESSING.md): bulk-transfer features are not reproduced and
subflow features equal the whole-flow totals. Both groups are near-constant or
duplicate columns in CICIDS2018 and are removed by preprocessing anyway.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .flow_builder import Flow
from .pcap_parser import ACK, CWR, ECE, FIN, PSH, RST, SYN, URG

# Exact column order of the CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" CSVs.
CIC_FEATURE_COLUMNS: list[str] = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio",
    "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Features this extractor does not reproduce faithfully (always 0 here).
NOT_REPRODUCED = {
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
}

FLOW_META_COLUMNS = ["Src IP", "Dst IP", "Src Port", "Start Time", "End Time", "Closed By", "SYN Without ACK"]


def _rate(amount: float, duration_us: float) -> float:
    """CICFlowMeter divides by duration in seconds; zero duration yields +inf/NaN in the CSVs."""
    seconds = duration_us / 1e6
    if seconds > 0:
        return amount / seconds
    return math.inf if amount > 0 else math.nan


def flow_to_features(flow: Flow) -> dict[str, float]:
    duration = flow.duration_us
    fwd, bwd, alll = flow.fwd_len, flow.bwd_len, flow.all_len
    total_bytes = fwd.total + bwd.total
    flags = flow.flag_counts
    f = {
        "Dst Port": flow.dst_port,
        "Protocol": flow.protocol,
        "Flow Duration": duration,
        "Tot Fwd Pkts": fwd.n,
        "Tot Bwd Pkts": bwd.n,
        "TotLen Fwd Pkts": fwd.total,
        "TotLen Bwd Pkts": bwd.total,
        "Fwd Pkt Len Max": fwd.safe_max(),
        "Fwd Pkt Len Min": fwd.safe_min(),
        "Fwd Pkt Len Mean": fwd.safe_mean(),
        "Fwd Pkt Len Std": fwd.std,
        "Bwd Pkt Len Max": bwd.safe_max(),
        "Bwd Pkt Len Min": bwd.safe_min(),
        "Bwd Pkt Len Mean": bwd.safe_mean(),
        "Bwd Pkt Len Std": bwd.std,
        "Flow Byts/s": _rate(total_bytes, duration),
        "Flow Pkts/s": _rate(alll.n, duration),
        "Flow IAT Mean": flow.flow_iat.safe_mean(),
        "Flow IAT Std": flow.flow_iat.std,
        "Flow IAT Max": flow.flow_iat.safe_max(),
        "Flow IAT Min": flow.flow_iat.safe_min(),
        "Fwd IAT Tot": flow.fwd_iat.total,
        "Fwd IAT Mean": flow.fwd_iat.safe_mean(),
        "Fwd IAT Std": flow.fwd_iat.std,
        "Fwd IAT Max": flow.fwd_iat.safe_max(),
        "Fwd IAT Min": flow.fwd_iat.safe_min(),
        "Bwd IAT Tot": flow.bwd_iat.total,
        "Bwd IAT Mean": flow.bwd_iat.safe_mean(),
        "Bwd IAT Std": flow.bwd_iat.std,
        "Bwd IAT Max": flow.bwd_iat.safe_max(),
        "Bwd IAT Min": flow.bwd_iat.safe_min(),
        "Fwd PSH Flags": flow.fwd_psh,
        "Bwd PSH Flags": flow.bwd_psh,
        "Fwd URG Flags": flow.fwd_urg,
        "Bwd URG Flags": flow.bwd_urg,
        "Fwd Header Len": flow.fwd_header_bytes,
        "Bwd Header Len": flow.bwd_header_bytes,
        "Fwd Pkts/s": _rate(fwd.n, duration),
        "Bwd Pkts/s": _rate(bwd.n, duration),
        "Pkt Len Min": alll.safe_min(),
        "Pkt Len Max": alll.safe_max(),
        "Pkt Len Mean": alll.safe_mean(),
        "Pkt Len Std": alll.std,
        "Pkt Len Var": alll.var,
        "FIN Flag Cnt": flags[FIN],
        "SYN Flag Cnt": flags[SYN],
        "RST Flag Cnt": flags[RST],
        "PSH Flag Cnt": flags[PSH],
        "ACK Flag Cnt": flags[ACK],
        "URG Flag Cnt": flags[URG],
        "CWE Flag Count": flags[CWR],
        "ECE Flag Cnt": flags[ECE],
        "Down/Up Ratio": (bwd.n // fwd.n) if fwd.n else 0,
        "Pkt Size Avg": (alll.total / alll.n) if alll.n else 0.0,
        "Fwd Seg Size Avg": fwd.safe_mean(),
        "Bwd Seg Size Avg": bwd.safe_mean(),
        "Subflow Fwd Pkts": fwd.n,
        "Subflow Fwd Byts": fwd.total,
        "Subflow Bwd Pkts": bwd.n,
        "Subflow Bwd Byts": bwd.total,
        "Init Fwd Win Byts": flow.init_fwd_win,
        "Init Bwd Win Byts": flow.init_bwd_win,
        "Fwd Act Data Pkts": flow.fwd_act_data_pkts,
        "Fwd Seg Size Min": flow.fwd_min_seg if math.isfinite(flow.fwd_min_seg) else 0,
        "Active Mean": flow.active.safe_mean(),
        "Active Std": flow.active.std,
        "Active Max": flow.active.safe_max(),
        "Active Min": flow.active.safe_min(),
        "Idle Mean": flow.idle.safe_mean(),
        "Idle Std": flow.idle.std,
        "Idle Max": flow.idle.safe_max(),
        "Idle Min": flow.idle.safe_min(),
    }
    for name in NOT_REPRODUCED:
        f[name] = 0.0
    return f


def flow_metadata(flow: Flow) -> dict:
    return {
        "Src IP": flow.src_ip, "Dst IP": flow.dst_ip, "Src Port": flow.src_port,
        "Start Time": flow.start_time, "End Time": flow.last_time,
        "Closed By": flow.closed_by, "SYN Without ACK": flow.syn_without_ack,
    }


def flows_to_frame(flows, include_metadata: bool = True) -> pd.DataFrame:
    """Convert flows into a DataFrame with CIC columns in canonical order (+ metadata)."""
    rows = []
    for flow in flows:
        row = flow_to_features(flow)
        if include_metadata:
            row.update(flow_metadata(flow))
        rows.append(row)
    columns = CIC_FEATURE_COLUMNS + (FLOW_META_COLUMNS if include_metadata else [])
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    frame[CIC_FEATURE_COLUMNS] = frame[CIC_FEATURE_COLUMNS].astype(np.float64)
    return frame
