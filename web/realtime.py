"""Realtime session for the web demo: PCAP replay (any OS) or live capture (Linux/Kali, root).

The session drives the project's own :class:`RealtimeProcessor` (same queue ->
decode -> FlowBuilder -> DetectionPipeline path as ``scripts/run_realtime.py``)
and pushes a snapshot to the browser after every detection run.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from collections import Counter, deque

from src.detection.alert_manager import AlertManager
from src.realtime.packet_capture import CaptureError, LiveCapture, PcapReplay
from src.realtime.realtime_processor import RealtimeProcessor

from .services import CFG, _f, pipeline

LIVE_ENABLED = os.environ.get("NIDS_ALLOW_LIVE", "0") == "1"


class _Tap:
    """Delegates to the pipeline and records per-run predictions for the dashboard."""

    def __init__(self, pipe) -> None:
        self._pipe = pipe
        self.families: Counter = Counter()
        self.anomalies = 0
        self.last_batch = 0

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def process_flows(self, frame):
        alerts, suspects, scores = self._pipe.process_flows(frame)
        self.last_batch = len(scores)
        if len(scores):
            self.families.update(scores["predicted_family"].tolist())
            self.anomalies += int(scores["is_anomaly"].sum())
        return alerts, suspects, scores


class Session:
    def __init__(self, mode: str, path: str | None = None, speed: float = 50.0, interface: str | None = None,
                 max_packets: int | None = None) -> None:
        if mode == "live" and not LIVE_ENABLED:
            raise CaptureError("Chế độ bắt gói trực tiếp đang tắt. Chạy server với NIDS_ALLOW_LIVE=1 "
                               "(Linux/Kali, quyền root hoặc CAP_NET_RAW).")
        self.mode = mode
        self.tap = _Tap(pipeline())
        cfg = {**CFG, "realtime": {**CFG["realtime"], "detection_interval_s": CFG["realtime"]["detection_interval_s"]}}
        self.alert_dir = tempfile.mkdtemp(prefix="nids_web_")
        self.am = AlertManager(self.alert_dir, console=False, suppress_window_s=30.0)
        self.proc = RealtimeProcessor(self.tap, self.am, cfg, replay_clock=(mode == "replay"))
        if mode == "replay":
            self.capture = PcapReplay(path, speed=speed, max_packets=max_packets)
        else:
            self.capture = LiveCapture(interface, CFG["realtime"]["bpf_filter"], CFG["realtime"]["queue_size"])
        self.updates: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recent = deque(maxlen=200)
        self.history = deque(maxlen=120)
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- loop
    def _snapshot(self, emitted: list, done: bool = False) -> dict:
        s = self.proc.stats
        elapsed = time.perf_counter() - s.started
        point = {"t": round(elapsed, 1), "packets": s.packets, "flows": s.flows_scored, "alerts": s.alerts,
                 "batch": self.tap.last_batch}
        self.history.append(point)
        new = [a.to_dict() for a in emitted]
        for a in new:
            a["confidence"] = _f(a["confidence"])
            self.recent.appendleft(a)
        cap = self.capture.stats
        return {
            "type": "done" if done else "update", "mode": self.mode,
            "stats": {"elapsed_s": round(elapsed, 1), "packets": s.packets, "invalid": s.invalid_packets,
                      "pps": round(s.packets / elapsed, 0) if elapsed else 0, "flows_scored": s.flows_scored,
                      "alerts": s.alerts, "anomalies": self.tap.anomalies, "detection_runs": s.detection_runs,
                      "captured": cap.captured, "dropped": getattr(cap, "dropped", 0),
                      "queue": self.capture.queue.qsize(), "packet_time": _f(self.proc._packet_now, 3),
                      "batch_ms": _f(s.batch_inference_ms[-1], 1) if s.batch_inference_ms else None,
                      "rss_mb": _f(s.rss_mb[-1], 0) if s.rss_mb else None,
                      "active_flows": len(self.proc.builder._active) if hasattr(self.proc.builder, "_active") else None},
            "families": dict(self.tap.families), "new_alerts": new, "point": point,
            "errors": list(getattr(cap, "errors", [])),
        }

    def _run(self) -> None:
        proc, src = self.proc, self.capture.queue
        interval = proc.interval
        last_packet_t, last_detect, last_push = None, time.perf_counter(), 0.0
        try:
            while not self.stop_event.is_set():
                got = 0
                try:
                    while got < 5000:
                        ts, frame, lt = src.get(timeout=0.2)
                        proc.handle_frame(ts, frame, lt)
                        got += 1
                        if (proc.replay_clock and last_packet_t is not None
                                and proc._packet_now - last_packet_t >= interval):
                            break      # run detection at every interval of packet time
                except queue.Empty:
                    pass
                emitted = None
                if proc.replay_clock:          # windows follow packet time when replaying
                    if last_packet_t is None:
                        last_packet_t = proc._packet_now
                    if proc._packet_now - last_packet_t >= interval:
                        emitted = proc.run_detection(proc._packet_now)
                        last_packet_t = proc._packet_now
                elif time.perf_counter() - last_detect >= interval:
                    emitted = proc.run_detection()
                    last_detect = time.perf_counter()
                now = time.perf_counter()
                if emitted is not None or now - last_push >= 1.0:
                    self.updates.put(self._snapshot(emitted or []))
                    last_push = now
                fin = getattr(self.capture, "finished", None)
                if fin is not None and fin.is_set() and src.empty():
                    break
            proc.builder.flush()
            emitted = proc.run_detection(proc._packet_now if proc.replay_clock else None)
            self.updates.put(self._snapshot(emitted, done=True))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            self.updates.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.capture.stop()
            self.am.close()

    def start(self) -> None:
        self.capture.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.capture.stop()


def interfaces() -> list[str]:
    if not LIVE_ENABLED:
        return []
    try:
        return LiveCapture.list_interfaces()
    except Exception:
        return []
