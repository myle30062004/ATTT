"""Bidirectional flow reconstruction following CICFlowMeter semantics.

* A flow is keyed by the 5-tuple; the first packet seen fixes the *forward*
  direction (its source is the flow initiator).
* A flow is closed when (a) it has lasted longer than ``flow_timeout`` seconds
  (CICFlowMeter "activity timeout", 120 s), (b) it has been idle for
  ``idle_timeout`` seconds, or (c) a TCP FIN/RST packet is seen (the packet is
  added to the flow first, as CICFlowMeter does).
* Packet length statistics use the L4 *payload* size, as in CICFlowMeter.
* All times inside the feature vector are in **microseconds**, like the
  CSE-CIC-IDS2018 CSVs.

The same class is used offline (PCAP files) and in the realtime monitor.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .pcap_parser import ACK, CWR, ECE, FIN, PSH, PROTO_TCP, RST, SYN, URG, PacketRecord

ACTIVITY_TIMEOUT_US = 5_000_000  # CICFlowMeter threshold for active/idle periods


class RunningStats:
    """Streaming count/sum/min/max/mean/sample-std (Welford)."""

    __slots__ = ("n", "total", "mean", "m2", "min", "max")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def safe_min(self) -> float:
        return self.min if self.n else 0.0

    def safe_max(self) -> float:
        return self.max if self.n else 0.0

    def safe_mean(self) -> float:
        return self.mean if self.n else 0.0


FlowKey = tuple


def flow_key(pkt: PacketRecord) -> FlowKey:
    a = (pkt.src_ip, pkt.src_port)
    b = (pkt.dst_ip, pkt.dst_port)
    return (a, b, pkt.protocol) if a <= b else (b, a, pkt.protocol)


@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float
    last_time: float = 0.0
    fwd_len: RunningStats = field(default_factory=RunningStats)
    bwd_len: RunningStats = field(default_factory=RunningStats)
    all_len: RunningStats = field(default_factory=RunningStats)
    flow_iat: RunningStats = field(default_factory=RunningStats)
    fwd_iat: RunningStats = field(default_factory=RunningStats)
    bwd_iat: RunningStats = field(default_factory=RunningStats)
    active: RunningStats = field(default_factory=RunningStats)
    idle: RunningStats = field(default_factory=RunningStats)
    last_fwd_time: float | None = None
    last_bwd_time: float | None = None
    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0
    fwd_psh: int = 0
    bwd_psh: int = 0
    fwd_urg: int = 0
    bwd_urg: int = 0
    flag_counts: dict = field(default_factory=lambda: {f: 0 for f in (FIN, SYN, RST, PSH, ACK, URG, CWR, ECE)})
    init_fwd_win: int = -1
    init_bwd_win: int = -1
    fwd_act_data_pkts: int = 0
    fwd_min_seg: float = math.inf
    start_active_us: float = 0.0
    end_active_us: float = 0.0
    syn_without_ack: int = 0
    closed_by: str = ""

    def __post_init__(self) -> None:
        self.last_time = self.start_time
        self.start_active_us = self.end_active_us = self.start_time * 1e6

    def is_forward(self, pkt: PacketRecord) -> bool:
        return pkt.src_ip == self.src_ip and pkt.src_port == self.src_port

    def add(self, pkt: PacketRecord) -> None:
        ts_us = pkt.timestamp * 1e6
        if self.all_len.n > 0:
            self.flow_iat.add(max(0.0, (pkt.timestamp - self.last_time) * 1e6))
            self._update_active_idle(ts_us)
        forward = self.is_forward(pkt)
        size = pkt.payload_length
        self.all_len.add(size)
        if forward:
            if self.last_fwd_time is not None:
                self.fwd_iat.add(max(0.0, (pkt.timestamp - self.last_fwd_time) * 1e6))
            self.last_fwd_time = pkt.timestamp
            self.fwd_len.add(size)
            self.fwd_header_bytes += pkt.header_length
            self.fwd_min_seg = min(self.fwd_min_seg, pkt.header_length)
            if size >= 1:
                self.fwd_act_data_pkts += 1
            if pkt.protocol == PROTO_TCP and self.init_fwd_win < 0 and pkt.window >= 0:
                self.init_fwd_win = pkt.window
            if pkt.tcp_flags & PSH:
                self.fwd_psh += 1
            if pkt.tcp_flags & URG:
                self.fwd_urg += 1
        else:
            if self.last_bwd_time is not None:
                self.bwd_iat.add(max(0.0, (pkt.timestamp - self.last_bwd_time) * 1e6))
            self.last_bwd_time = pkt.timestamp
            self.bwd_len.add(size)
            self.bwd_header_bytes += pkt.header_length
            if pkt.protocol == PROTO_TCP and self.init_bwd_win < 0 and pkt.window >= 0:
                self.init_bwd_win = pkt.window
            if pkt.tcp_flags & PSH:
                self.bwd_psh += 1
            if pkt.tcp_flags & URG:
                self.bwd_urg += 1
        if pkt.tcp_flags:
            for flag in self.flag_counts:
                if pkt.tcp_flags & flag:
                    self.flag_counts[flag] += 1
            if pkt.tcp_flags & SYN and not pkt.tcp_flags & ACK:
                self.syn_without_ack += 1
        self.last_time = max(self.last_time, pkt.timestamp)

    def _update_active_idle(self, ts_us: float) -> None:
        if ts_us - self.end_active_us > ACTIVITY_TIMEOUT_US:
            if self.end_active_us - self.start_active_us > 0:
                self.active.add(self.end_active_us - self.start_active_us)
            self.idle.add(ts_us - self.end_active_us)
            self.start_active_us = self.end_active_us = ts_us
        else:
            self.end_active_us = ts_us

    @property
    def packet_count(self) -> int:
        return self.all_len.n

    @property
    def duration_us(self) -> float:
        return (self.last_time - self.start_time) * 1e6


class FlowBuilder:
    """Incrementally groups packets into flows and emits them when they close."""

    def __init__(self, flow_timeout_s: float = 120.0, idle_timeout_s: float = 60.0,
                 max_active_flows: int = 200_000,
                 on_flow_closed: Callable[[Flow], None] | None = None) -> None:
        self.flow_timeout = flow_timeout_s
        self.idle_timeout = idle_timeout_s
        self.max_active_flows = max_active_flows
        self.on_flow_closed = on_flow_closed
        # Ordered by last activity (least recently active first) so idle
        # expiry only touches flows that actually expire.
        self.active: OrderedDict[FlowKey, Flow] = OrderedDict()
        self._finished: list[Flow] = []
        self.evicted_for_memory = 0

    def _close(self, key: FlowKey, reason: str) -> None:
        flow = self.active.pop(key)
        flow.closed_by = reason
        if self.on_flow_closed is not None:
            self.on_flow_closed(flow)
        else:
            self._finished.append(flow)

    def add_packet(self, pkt: PacketRecord) -> None:
        key = flow_key(pkt)
        flow = self.active.get(key)
        if flow is not None and pkt.timestamp - flow.start_time > self.flow_timeout:
            self._close(key, "active_timeout")
            flow = None
        if flow is None:
            if len(self.active) >= self.max_active_flows:
                self._close(next(iter(self.active)), "memory_limit")
                self.evicted_for_memory += 1
            flow = Flow(pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol, pkt.timestamp)
            self.active[key] = flow
        else:
            self.active.move_to_end(key)
        flow.add(pkt)
        if pkt.protocol == PROTO_TCP and pkt.tcp_flags & (FIN | RST):
            self._close(key, "fin_rst")
        self.expire(pkt.timestamp)

    def expire(self, now: float) -> None:
        """Close flows idle for longer than ``idle_timeout`` (oldest activity first)."""
        while self.active:
            key = next(iter(self.active))
            if now - self.active[key].last_time <= self.idle_timeout:
                break
            self._close(key, "idle_timeout")

    def expire_all(self, now: float) -> None:
        """Also close flows exceeding the active timeout (used by the realtime loop)."""
        self.expire(now)
        for key in [k for k, f in self.active.items() if now - f.start_time > self.flow_timeout]:
            self._close(key, "active_timeout")

    def flush(self) -> None:
        for key in list(self.active):
            self._close(key, "end_of_capture")

    def pop_finished(self) -> list[Flow]:
        done, self._finished = self._finished, []
        return done


def build_flows(packets: Iterable[PacketRecord], flow_timeout_s: float = 120.0,
                idle_timeout_s: float = 60.0, max_active_flows: int = 200_000) -> Iterator[Flow]:
    builder = FlowBuilder(flow_timeout_s, idle_timeout_s, max_active_flows)
    for pkt in packets:
        builder.add_packet(pkt)
        if builder._finished:
            yield from builder.pop_finished()
    builder.flush()
    yield from builder.pop_finished()
