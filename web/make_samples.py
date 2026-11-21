"""Build the demo samples listed in web/samples/samples.json.

    python web/make_samples.py

* synthetic_scan_beacon.pcap - known ground truth: normal web sessions, a SYN port scan,
  a host sweep and a 30 s beacon (same generators as the integration tests).
* ctu13_neris_slice.pcap     - 20 minutes of CTU-13 scenario 1 restricted to the labelled
  hosts (infected 147.32.84.165 + documented normal hosts).
* cicids2018_test_sample.csv - stratified rows of the held-out *test* split (CICFlowMeter
  columns + Family ground truth), never used for training or model selection.
Existing captures (CICIDS2018 Bot victim PCAP, traffic-analysis exercise) are referenced in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dpkt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from tests.helpers import tcp_frame, tcp_session, udp_frame, write_pcap  # noqa: E402

OUT = ROOT / "web" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
CFG = load_config()


def web_session(rng, src, dst, sport, dport, t0) -> list:
    """TCP session with realistic variability: RTT, request size, multi-segment response, ACKs."""
    S, A, P, F = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH, dpkt.tcp.TH_FIN
    rtt = rng.uniform(0.004, 0.12)
    cw, sw = int(rng.choice([64240, 65535, 29200])), int(rng.choice([65535, 26883, 28960, 8192]))
    t, out = t0, []
    out.append((t, tcp_frame(src, dst, sport, dport, S, window=cw)))
    t += rtt
    out.append((t, tcp_frame(dst, src, dport, sport, S | A, window=sw)))
    t += rng.uniform(0.0002, 0.002)
    out.append((t, tcp_frame(src, dst, sport, dport, A, window=cw)))
    for _ in range(int(rng.integers(1, 4))):                 # keep-alive: 1-3 requests
        t += rng.uniform(0.001, 0.4)
        out.append((t, tcp_frame(src, dst, sport, dport, P | A, b"G" * int(rng.integers(180, 900)), window=cw)))
        size = int(rng.lognormal(8.5, 1.3))
        t += rtt + rng.uniform(0.001, 0.05)
        while size > 0:
            seg = min(size, 1460)
            out.append((t, tcp_frame(dst, src, dport, sport, A | (P if seg < 1460 else 0), b"x" * seg, window=sw)))
            size -= seg
            t += rng.uniform(0.0001, 0.003)
            if rng.random() < 0.5:
                out.append((t, tcp_frame(src, dst, sport, dport, A, window=cw)))
    t += rng.uniform(0.05, 5.0)
    out.append((t, tcp_frame(src, dst, sport, dport, F | A, window=cw)))
    t += rtt
    out.append((t, tcp_frame(dst, src, dport, sport, F | A, window=sw)))
    out.append((t + 0.0005, tcp_frame(src, dst, sport, dport, A, window=cw)))
    return out


def synthetic() -> Path:
    S = dpkt.tcp.TH_SYN
    t0, frames = 1_700_000_000.0, []
    import numpy as np
    rng = np.random.default_rng(2026)
    for h in range(1, 6):                                       # 5 normal clients browsing
        for i in range(60):
            frames += web_session(rng, f"192.168.10.{h}", f"93.184.216.{(i % 7) + 1}", 40000 + i,
                                  443 if i % 3 else 80, t0 + h * 1.7 + i * 14.0 + rng.uniform(0, 6))
    for i in range(40):                                         # DNS lookups
        frames.append((t0 + i * 20.0, udp_frame("192.168.10.2", "192.168.10.254", 53000 + i, 53, b"q" * 30)))
        frames.append((t0 + i * 20.0 + 0.01, udp_frame("192.168.10.254", "192.168.10.2", 53, 53000 + i, b"a" * 90)))
    for p in range(1, 1025):                                    # SYN port scan of a server
        frames.append((t0 + 300 + p * 0.003, tcp_frame("192.168.10.66", "192.168.10.20", 55555, p, S)))
    for h in range(1, 121):                                     # host sweep on port 445
        frames.append((t0 + 500 + h * 0.01, tcp_frame("192.168.10.67", f"10.20.0.{h}", 56000, 445, S)))
    for i in range(30):                                         # C&C beacon every 30 s
        frames += tcp_session(src="192.168.10.7", dst="185.199.10.9", sport=50000 + i, dport=8080,
                              t0=t0 + 5 + i * 30.0, req=b"GET /ping", resp=b"cmd:sleep")
    return write_pcap(OUT / "synthetic_scan_beacon.pcap", sorted(frames, key=lambda f: f[0]))


def ctu_slice(minutes_from: float = 74, minutes_len: float = 20) -> Path | None:
    src = (ROOT / CFG["ctu13"]["pcap"]).resolve()
    if not src.exists():
        print("CTU-13 pcap not found, skipped:", src)
        return None
    import socket
    hosts = {socket.inet_aton(h) for h in CFG["ctu13"]["infected_hosts"] + CFG["ctu13"]["normal_hosts"]}
    out = OUT / "ctu13_neris_slice.pcap"
    n_in = n_out = 0
    with open(src, "rb") as fi, open(out, "wb") as fo:
        magic = fi.read(4)
        fi.seek(0)
        reader = dpkt.pcapng.Reader(fi) if magic == bytes([0x0A, 0x0D, 0x0D, 0x0A]) else dpkt.pcap.Reader(fi)
        writer = dpkt.pcap.Writer(fo, linktype=reader.datalink())
        start = None
        for ts, buf in reader:
            n_in += 1
            if start is None:
                start = ts
            rel = (ts - start) / 60
            if rel < minutes_from:
                continue
            if rel > minutes_from + minutes_len:
                break
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
            except Exception:
                continue
            if isinstance(ip, dpkt.ip.IP) and (ip.src in hosts or ip.dst in hosts):
                writer.writepkt(buf, ts=ts)
                n_out += 1
    print(f"CTU-13 slice: read {n_in:,} packets, kept {n_out:,}")
    return out


def test_csv(per_family: int = 1500) -> Path:
    df = pd.read_parquet(ROOT / "data" / "processed" / "split_test.parquet")
    parts = [g.sample(min(len(g), per_family), random_state=7) for _, g in df.groupby("Family")]
    sample = pd.concat(parts).sample(frac=1, random_state=7)
    sample = sample.drop(columns=[c for c in ("row_hash", "IsAttack", "Day", "RawLabel") if c in sample])
    # processed parquet keeps Timestamp as epoch/1000 (int); write it back in CICFlowMeter's format
    sample["Timestamp"] = pd.to_datetime(sample["Timestamp"].astype("int64") * 1000, unit="s").dt.strftime("%d/%m/%Y %H:%M:%S")
    out = OUT / "cicids2018_test_sample.csv"
    sample.to_csv(out, index=False)
    return out, len(sample)


def main() -> None:
    synthetic()
    ctu = ctu_slice()
    _, n_csv = test_csv()
    ext = Path(CFG["paths"]["external_pcaps"])
    samples = [
        {"id": "synthetic", "kind": "pcap", "path": "web/samples/synthetic_scan_beacon.pcap",
         "title": "Kịch bản tổng hợp: quét cổng, quét host, beacon C&C",
         "description": "Biết trước đáp án: 192.168.10.66 quét 1.024 cổng, 192.168.10.67 quét 120 host cổng 445, "
                        "192.168.10.7 gọi về 185.199.10.9:8080 mỗi 30 giây; còn lại là web/DNS bình thường.",
         "truth": "PortScan · HostSweep · Beaconing"},
        {"id": "cic_bot", "kind": "pcap",
         "path": "data/raw/cicids2018_pcap/Friday-02-03-2018/capEC2AMAZ-O4EL3NG-172.31.69.26",
         "title": "CICIDS2018 – PCAP gốc máy nạn nhân Bot (02/03/2018)",
         "description": "Gói tin thật của testbed; máy 172.31.69.26 nhiễm bot gọi về C&C qua TCP 8080. "
                        "Dùng để kiểm chứng feature parity với CICFlowMeter.",
         "truth": "Botnet (C&C → 8080)"},
        {"id": "cic_csv", "kind": "csv", "path": "web/samples/cicids2018_test_sample.csv",
         "title": f"CICIDS2018 – {n_csv:,} flow của tập test (có nhãn)".replace(",", "."),
         "description": "Các dòng CICFlowMeter lấy phân tầng từ tập test độc lập (tối đa 1.500 dòng mỗi họ). "
                        "Có cột Family nên web tính được ma trận nhầm lẫn và Macro-F1.",
         "truth": "7 họ có nhãn"},
        {"id": "exercise", "kind": "pcap", "path": str((ext / "2026-10-01-traffic-analysis-exercise.pcap").as_posix()),
         "title": "PCAP bài tập phân tích traffic (mạng khác, không nhãn)",
         "description": "42.497 gói từ một mạng hoàn toàn khác testbed – minh hoạ dịch chuyển phân phối: "
                        "cần hiệu chỉnh ngưỡng GMM trước khi triển khai.",
         "truth": "Không có nhãn"},
    ]
    if ctu is not None:
        samples.insert(2, {"id": "ctu_neris", "kind": "pcap", "path": "web/samples/ctu13_neris_slice.pcap",
                           "title": "CTU-13 kịch bản 1 – phút 74–94, các host có nhãn",
                           "description": "Máy 147.32.84.165 bắt đầu hoạt động như bot Neris (IRC C&C, spam SMTP) từ phút 75, "
                                          "cùng 6 máy bình thường. Phân tích mức host theo cửa sổ 300 s.",
                           "truth": "Botnet host 147.32.84.165"})
    for s in samples:
        p = Path(s["path"])
        p = p if p.is_absolute() else (ROOT / p).resolve()
        s["available"] = p.exists()
    (OUT / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([(s["id"], s["available"]) for s in samples]))


if __name__ == "__main__":
    main()
