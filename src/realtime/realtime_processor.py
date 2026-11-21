"""Realtime processing loop: queue -> decode -> flows -> detection -> alerts.

* Closed flows (FIN/RST, idle timeout, active timeout) are scored every
  ``detection_interval_s`` seconds.
* A sliding window of the last ``host_window_s`` seconds of flows feeds the
  host-level botnet analysis (beaconing, host model, topology).
* Latency is measured per alerted flow as wall-clock time between the flow's
  last packet being *processed* and the alert being emitted, plus the
  inherent wait until the flow closes (reported separately).
"""
from __future__ import annotations

import queue
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psutil

from ..data_processing.feature_extractor import flows_to_frame
from ..data_processing.flow_builder import Flow, FlowBuilder
from ..data_processing.pcap_parser import decode_frame
from ..detection.alert_manager import AlertManager
from ..detection.detection_pipeline import DetectionPipeline


@dataclass
class ProcessorStats:
    packets: int = 0
    invalid_packets: int = 0
    flows_scored: int = 0
    alerts: int = 0
    detection_runs: int = 0
    processing_latency_ms: list = field(default_factory=list)
    flow_close_wait_s: list = field(default_factory=list)
    batch_inference_ms: list = field(default_factory=list)
    cpu_percent: list = field(default_factory=list)
    rss_mb: list = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def summary(self) -> dict:
        elapsed = time.perf_counter() - self.started

        def pct(values, q):
            return float(np.percentile(values, q)) if values else None

        return {
            "elapsed_s": round(elapsed, 2), "packets": self.packets, "invalid_packets": self.invalid_packets,
            "packets_per_s": round(self.packets / elapsed, 1) if elapsed else None,
            "flows_scored": self.flows_scored, "alerts": self.alerts, "detection_runs": self.detection_runs,
            "processing_latency_ms_p50": pct(self.processing_latency_ms, 50),
            "processing_latency_ms_p95": pct(self.processing_latency_ms, 95),
            "flow_close_wait_s_p50": pct(self.flow_close_wait_s, 50),
            "batch_inference_ms_mean": float(np.mean(self.batch_inference_ms)) if self.batch_inference_ms else None,
            "cpu_percent_mean": float(np.mean(self.cpu_percent)) if self.cpu_percent else None,
            "rss_mb_max": float(np.max(self.rss_mb)) if self.rss_mb else None,
        }


class RealtimeProcessor:
    def __init__(self, pipeline: DetectionPipeline, alerts: AlertManager, config: dict,
                 replay_clock: bool = False) -> None:
        flow_cfg, rt_cfg = config["flow"], config["realtime"]
        self.pipeline = pipeline
        self.alerts = alerts
        self.closed: list[tuple[Flow, float]] = []
        self.builder = FlowBuilder(flow_cfg["flow_timeout_s"], flow_cfg["idle_timeout_s"],
                                   flow_cfg["max_active_flows"], on_flow_closed=self._on_closed)
        self.interval = rt_cfg["detection_interval_s"]
        self.host_window = rt_cfg["host_window_s"]
        self.window: deque = deque()
        self.stats = ProcessorStats()
        self.replay_clock = replay_clock  # use packet time (replay) instead of wall clock for windows
        self._proc = psutil.Process()
        self._last_run = time.perf_counter()
        self._packet_now = 0.0

    def _on_closed(self, flow: Flow) -> None:
        self.closed.append((flow, time.perf_counter()))

    def handle_frame(self, ts: float, frame: bytes, linktype: int) -> None:
        pkt = decode_frame(ts, frame, linktype)
        if pkt is None:
            self.stats.invalid_packets += 1
            return
        self.stats.packets += 1
        self._packet_now = ts
        self.builder.add_packet(pkt)

    def run_detection(self, now_packet_time: float | None = None) -> list:
        now = now_packet_time if now_packet_time is not None else (self._packet_now if self.replay_clock else time.time())
        self.builder.expire_all(now)
        batch, self.closed = self.closed, []
        self.stats.detection_runs += 1
        emitted = []
        if batch:
            flows = [f for f, _ in batch]
            frame = flows_to_frame(flows)
            t0 = time.perf_counter()
            flow_alerts, suspects, _ = self.pipeline.process_flows(frame)
            infer_ms = (time.perf_counter() - t0) * 1000
            self.stats.batch_inference_ms.append(infer_ms)
            self.stats.flows_scored += len(frame)
            done = time.perf_counter()
            for (flow, closed_at) in batch:
                self.stats.processing_latency_ms.append((done - closed_at) * 1000)
                self.stats.flow_close_wait_s.append(max(0.0, now - flow.last_time))
            hf = self.pipeline.host_flows_from_frame(frame)
            self.window.extend(hf.to_dict("records"))
            cutoff = now - self.host_window
            while self.window and self.window[0]["start_time"] < cutoff:
                self.window.popleft()
            host_alerts, _ = self.pipeline.analyze_hosts(pd.DataFrame(list(self.window)), suspects)
            for alert in flow_alerts + host_alerts:
                if self.alerts.emit(alert, now=now):
                    self.stats.alerts += 1
                    emitted.append(alert)
        self.stats.cpu_percent.append(self._proc.cpu_percent(interval=None))
        self.stats.rss_mb.append(self._proc.memory_info().rss / 2**20)
        return emitted

    def consume(self, source_queue: queue.Queue, stop_event, finished_event=None, max_batch: int = 5000) -> None:
        """Drain the capture queue until ``stop_event`` is set (or replay finished and queue empty)."""
        last_detect_packet_time = None
        while not stop_event.is_set():
            got = 0
            try:
                while got < max_batch:
                    ts, frame, linktype = source_queue.get(timeout=0.2)
                    self.handle_frame(ts, frame, linktype)
                    got += 1
            except queue.Empty:
                pass
            if self.replay_clock:
                if last_detect_packet_time is None:
                    last_detect_packet_time = self._packet_now
                if self._packet_now - last_detect_packet_time >= self.interval:
                    self.run_detection(self._packet_now)
                    last_detect_packet_time = self._packet_now
            elif time.perf_counter() - self._last_run >= self.interval:
                self.run_detection()
                self._last_run = time.perf_counter()
            if finished_event is not None and finished_event.is_set() and source_queue.empty():
                break
        self.builder.flush()
        self.run_detection(self._packet_now if self.replay_clock else None)
