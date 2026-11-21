"""Realtime NIDS monitor: wires capture, processor, pipeline and alert output together."""
from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

from ..detection.alert_manager import AlertManager
from ..detection.detection_pipeline import DetectionPipeline
from ..utils.config import PROJECT_ROOT, load_config
from .packet_capture import CaptureError, LiveCapture, PcapReplay
from .realtime_processor import RealtimeProcessor


def run_monitor(interface: str | None = None, pcap: str | None = None, speed: float = 0.0,
                duration_s: float | None = None, bundle_dir: str | None = None, config_path: str | None = None,
                alert_dir: str | None = None, sqlite: bool = False, max_packets: int | None = None,
                quiet: bool = False) -> dict:
    cfg = load_config(config_path)
    bundle = Path(bundle_dir) if bundle_dir else PROJECT_ROOT / cfg["paths"]["models"] / "deployment"
    pipeline = DetectionPipeline(bundle, cfg)
    out_dir = Path(alert_dir) if alert_dir else PROJECT_ROOT / cfg["realtime"]["alert_dir"]
    alerts = AlertManager(out_dir, console=not quiet, use_sqlite=sqlite)
    replay = pcap is not None
    if replay:
        source = PcapReplay(pcap, speed=speed, queue_size=cfg["realtime"]["queue_size"], max_packets=max_packets)
    else:
        source = LiveCapture(interface or cfg["realtime"]["interface"], cfg["realtime"]["bpf_filter"],
                             cfg["realtime"]["queue_size"])
    processor = RealtimeProcessor(pipeline, alerts, cfg, replay_clock=replay)
    stop = threading.Event()

    def _handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)
    try:
        source.start()
    except CaptureError as exc:
        alerts.close()
        raise SystemExit(f"capture error: {exc}")
    if duration_s:
        threading.Timer(duration_s, stop.set).start()
    print(f"NIDS monitor started ({'replay ' + str(pcap) if replay else 'interface ' + str(source.interface)}); "
          f"Ctrl+C to stop", flush=True)
    t0 = time.perf_counter()
    try:
        processor.consume(source.queue, stop, getattr(source, "finished", None))
    finally:
        source.stop()
        alerts.close()
    summary = processor.stats.summary()
    summary.update({"captured": source.stats.captured, "dropped_by_capture": source.stats.dropped,
                    "capture_errors": source.stats.errors, "mode": "replay" if replay else "live",
                    "wall_time_s": round(time.perf_counter() - t0, 2),
                    "alerts_by_type": alerts.counts, "missing_features": sorted(pipeline.missing_feature_warnings),
                    "active_flows_evicted_for_memory": processor.builder.evicted_for_memory})
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return summary
