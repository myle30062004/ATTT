"""Live packet capture with Scapy.

Scapy is used only to *capture*: each packet's raw bytes and timestamp are put
on a bounded queue and decoded later by :func:`decode_frame`, the same decoder
used for offline PCAP files (feature parity). The bounded queue caps memory;
when it is full, packets are counted as dropped instead of exhausting RAM.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..data_processing.pcap_parser import (LINKTYPE_ETHERNET, LINKTYPE_LINUX_SLL, LINKTYPE_NULL, LINKTYPE_RAW,
                                          iter_pcap)


class CaptureError(RuntimeError):
    pass


@dataclass
class CaptureStats:
    captured: int = 0
    dropped: int = 0
    errors: list = field(default_factory=list)


def _linktype_of(pkt) -> int:
    name = type(pkt).__name__
    if name == "Ether":
        return LINKTYPE_ETHERNET
    if name in ("CookedLinux", "CookedLinuxV2"):
        return LINKTYPE_LINUX_SLL
    if name == "Loopback":
        return LINKTYPE_NULL
    return LINKTYPE_RAW


class LiveCapture:
    """Background capture: ``start()``, read ``(ts, bytes, linktype)`` tuples from ``.queue``, ``stop()``."""

    def __init__(self, interface: str | None, bpf_filter: str | None = "ip", queue_size: int = 100_000) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stats = CaptureStats()
        self._sniffer = None

    @staticmethod
    def list_interfaces() -> list[str]:
        """Interface names accepted by Scapy (friendly names on Windows, kernel names on Linux)."""
        from scapy.all import conf, get_if_list
        names = {i.name for i in conf.ifaces.values()} | {i.network_name for i in conf.ifaces.values()}
        return sorted(n for n in names | set(get_if_list()) if n)

    def _on_packet(self, pkt) -> None:
        try:
            self.queue.put_nowait((float(pkt.time), bytes(pkt), _linktype_of(pkt)))
            self.stats.captured += 1
        except queue.Full:
            self.stats.dropped += 1

    def start(self) -> None:
        try:
            from scapy.all import AsyncSniffer
        except ImportError as exc:  # pragma: no cover
            raise CaptureError("Scapy is not installed") from exc
        if self.interface is not None and self.interface not in self.list_interfaces():
            raise CaptureError(f"interface {self.interface!r} not found; available: {self.list_interfaces()}")
        try:
            self._sniffer = AsyncSniffer(iface=self.interface, filter=self.bpf_filter, store=False,
                                         prn=self._on_packet)
            self._sniffer.start()
        except PermissionError as exc:
            raise CaptureError("permission denied: run as root or grant CAP_NET_RAW/CAP_NET_ADMIN "
                               "(sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python3)))") from exc
        except OSError as exc:
            raise CaptureError(f"cannot open interface {self.interface!r}: {exc}") from exc

    def stop(self) -> None:
        if self._sniffer is not None and self._sniffer.running:
            self._sniffer.stop()


class PcapReplay:
    """Feeds a PCAP through the realtime path (same queue interface) for testing without root.

    ``speed`` = 0 replays as fast as possible; 1.0 = original timing.
    """

    def __init__(self, path: str | Path, speed: float = 0.0, queue_size: int = 100_000,
                 max_packets: int | None = None) -> None:
        self.path = Path(path)
        self.speed = speed
        self.max_packets = max_packets
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stats = CaptureStats()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.finished = threading.Event()

    def _run(self) -> None:
        import time
        import dpkt
        try:
            with open(self.path, "rb") as fh:
                magic = fh.read(4)
                fh.seek(0)
                reader = dpkt.pcapng.Reader(fh) if magic == b"\x0a\x0d\x0d\x0a" else dpkt.pcap.Reader(fh)
                linktype = reader.datalink()
                first_ts, wall0 = None, time.perf_counter()
                for ts, frame in reader:
                    if self._stop.is_set():
                        break
                    if first_ts is None:
                        first_ts = ts
                    if self.speed > 0:
                        delay = (ts - first_ts) / self.speed - (time.perf_counter() - wall0)
                        if delay > 0:
                            time.sleep(delay)
                    self.queue.put((float(ts), bytes(frame), linktype))  # blocking: replay never drops
                    self.stats.captured += 1
                    if self.max_packets and self.stats.captured >= self.max_packets:
                        break
        except Exception as exc:  # surface reader errors to the monitor
            self.stats.errors.append(repr(exc))
        finally:
            self.finished.set()

    def start(self) -> None:
        if not self.path.exists():
            raise CaptureError(f"pcap not found: {self.path}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


__all__ = ["LiveCapture", "PcapReplay", "CaptureError", "iter_pcap"]
