"""Offline PCAP processing: packets -> CICFlowMeter-compatible flows (-> optional detection).

    python scripts/process_pcap.py file.pcap --out data/processed/file_flows.parquet
    python scripts/process_pcap.py file.pcap --detect            # also run the detection pipeline
    python scripts/process_pcap.py big.pcap --max-packets 5000000 --start 1312987000 --end 1312990000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS, flows_to_frame  # noqa: E402
from src.data_processing.flow_builder import FlowBuilder  # noqa: E402
from src.data_processing.pcap_parser import iter_pcap  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    frame[CIC_FEATURE_COLUMNS] = frame[CIC_FEATURE_COLUMNS].astype("float32")
    return frame


def pcap_to_flows(path: str | Path, max_packets: int | None = None, start: float | None = None,
                  end: float | None = None, chunk_flows: int = 500_000, log=print,
                  part_dir: str | Path | None = None) -> pd.DataFrame:
    """Reconstruct flows. With ``part_dir`` flows are streamed to parquet part files
    (float32 features) and only a summary frame of the last chunk is kept in memory."""
    cfg = load_config()["flow"]
    builder = FlowBuilder(cfg["flow_timeout_s"], cfg["idle_timeout_s"], cfg["max_active_flows"])
    frames, pending, t0, n = [], [], time.time(), 0
    written = 0
    if part_dir is not None:
        part_dir = Path(part_dir)
        part_dir.mkdir(parents=True, exist_ok=True)

    def _store(chunk):
        nonlocal written
        frame = _compact(flows_to_frame(chunk))
        if part_dir is not None:
            frame.to_parquet(part_dir / f"part-{len(list(part_dir.glob('part-*.parquet'))):05d}.parquet", index=False)
            written += len(frame)
        else:
            frames.append(frame)

    for pkt in iter_pcap(path, max_packets, start, end):
        builder.add_packet(pkt)
        n += 1
        if builder._finished:
            pending.extend(builder.pop_finished())
        if len(pending) >= chunk_flows:
            _store(pending)
            pending = []
        if n % 2_000_000 == 0:
            log(f"  {n:,} packets, {written + sum(map(len, frames)) + len(pending):,} flows, "
                f"{n / (time.time() - t0):,.0f} pkt/s", flush=True)
    builder.flush()
    pending.extend(builder.pop_finished())
    _store(pending)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["parse_stats"] = dict(iter_pcap.stats, seconds=round(time.time() - t0, 1),
                                    evicted_for_memory=builder.evicted_for_memory,
                                    flows_written=written + len(out))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pcap")
    p.add_argument("--out", help="output parquet for flows")
    p.add_argument("--part-dir", help="stream flows to parquet part files in this directory (large pcaps)")
    p.add_argument("--max-packets", type=int)
    p.add_argument("--start", type=float, help="epoch seconds")
    p.add_argument("--end", type=float, help="epoch seconds")
    p.add_argument("--detect", action="store_true", help="run detection pipeline on the flows")
    p.add_argument("--alerts", default="logs/pcap_alerts")
    a = p.parse_args()
    flows = pcap_to_flows(a.pcap, a.max_packets, a.start, a.end, part_dir=a.part_dir)
    stats = flows.attrs["parse_stats"]
    print(json.dumps(stats, indent=2))
    if a.part_dir:
        (Path(a.part_dir) / "parse_stats.json").write_text(json.dumps(stats, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        flows.to_parquet(a.out, index=False)
    if a.detect:
        from src.detection.alert_manager import AlertManager
        from src.detection.detection_pipeline import DetectionPipeline
        from src.utils.config import PROJECT_ROOT

        cfg = load_config()
        pipe = DetectionPipeline(PROJECT_ROOT / "models" / "deployment", cfg)
        am = AlertManager(a.alerts, console=False)
        alerts, suspects, scores = pipe.process_flows(flows)
        host_alerts, report = pipe.analyze_hosts(pipe.host_flows_from_frame(flows), suspects)
        for al in alerts + host_alerts:
            am.emit(al)
        am.close()
        print(json.dumps({"flow_alerts": len(alerts), "host_alerts": len(host_alerts),
                          "predicted_families": scores["predicted_family"].value_counts().to_dict(),
                          "anomalies": int(scores["is_anomaly"].sum()), "alerts_by_type": am.counts}, indent=2))


if __name__ == "__main__":
    main()
