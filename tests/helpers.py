"""Synthetic packet/flow builders for tests (exact, known ground truth)."""
from __future__ import annotations

import socket
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd


def tcp_frame(src: str, dst: str, sport: int, dport: int, flags: int, payload: bytes = b"", window: int = 8192) -> bytes:
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=window, data=payload)
    tcp.off = 5
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_TCP, data=tcp)
    ip.len = 20 + len(bytes(tcp))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def udp_frame(src: str, dst: str, sport: int, dport: int, payload: bytes = b"") -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_UDP, data=udp)
    ip.len = 20 + len(bytes(udp))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> Path:
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for ts, frame in frames:
            writer.writepkt(frame, ts=ts)
    return path


def tcp_session(src="10.0.0.1", dst="10.0.0.2", sport=40000, dport=80, t0=1000.0, req=b"GET / HTTP/1.1\r\n\r\n",
                resp=b"x" * 500) -> list[tuple[float, bytes]]:
    S, A, P, F = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH, dpkt.tcp.TH_FIN
    return [
        (t0 + 0.000, tcp_frame(src, dst, sport, dport, S, window=64240)),
        (t0 + 0.010, tcp_frame(dst, src, dport, sport, S | A, window=65535)),
        (t0 + 0.020, tcp_frame(src, dst, sport, dport, A)),
        (t0 + 0.030, tcp_frame(src, dst, sport, dport, P | A, req)),
        (t0 + 0.050, tcp_frame(dst, src, dport, sport, P | A, resp)),
        (t0 + 0.060, tcp_frame(src, dst, sport, dport, F | A)),
    ]


def synthetic_cic_frame(n_benign=600, n_attack=200, seed=0) -> pd.DataFrame:
    """Random CIC-schema rows: attacks have short, SYN-heavy, high-rate flows."""
    from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    rows = []
    for label, n in (("Benign", n_benign), ("DoS-Hulk", n_attack)):
        base = rng.lognormal(3, 1, size=(n, len(CIC_FEATURE_COLUMNS)))
        df = pd.DataFrame(base, columns=CIC_FEATURE_COLUMNS)
        if label != "Benign":
            df["Flow Duration"] = rng.uniform(1, 100, n)
            df["SYN Flag Cnt"] = rng.integers(3, 10, n)
            df["Flow Pkts/s"] = rng.uniform(1e5, 1e6, n)
        else:
            df["Flow Duration"] = rng.uniform(1e5, 1e7, n)
            df["SYN Flag Cnt"] = rng.integers(0, 2, n)
            df["Flow Pkts/s"] = rng.uniform(1, 100, n)
        df["Fwd Byts/b Avg"] = 0.0  # constant column
        df["FineLabel"] = label
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["Family"] = np.where(out["FineLabel"] == "Benign", "BENIGN", "DoS")
    out["IsAttack"] = (out["Family"] != "BENIGN").astype(int)
    out["row_hash"] = np.arange(len(out), dtype=np.uint64)
    out["Timestamp"] = np.arange(len(out))
    return out
