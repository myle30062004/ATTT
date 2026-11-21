"""Packet decoding shared by offline PCAP processing and live capture.

Every packet source (dpkt for files, Scapy for live sniffing) hands raw frame
bytes to :func:`decode_frame`, so offline and realtime features are computed
from identical packet records.

Payload length is derived from the IP total-length field, not from the
captured length, so header-truncated captures (e.g. CTU-13 ``*.truncated.pcap``)
still give correct byte counts.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import dpkt

PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1

FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80

LINKTYPE_NULL, LINKTYPE_ETHERNET, LINKTYPE_RAW, LINKTYPE_LOOP = 0, 1, 101, 108
LINKTYPE_LINUX_SLL, LINKTYPE_IPV4, LINKTYPE_IPV6 = 113, 228, 229


@dataclass(slots=True)
class PacketRecord:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    ip_length: int
    payload_length: int
    header_length: int
    tcp_flags: int = 0
    window: int = -1

    def has_flag(self, flag: int) -> bool:
        return bool(self.tcp_flags & flag)


def _decode_transport(ts: float, src: str, dst: str, proto: int, ip_total: int,
                      ip_hdr_len: int, seg: bytes) -> PacketRecord | None:
    """Decode the L4 header. ``seg`` may be truncated; lengths come from the IP header."""
    l4_total = ip_total - ip_hdr_len
    if l4_total < 0:
        return None
    if proto == PROTO_TCP:
        if len(seg) < 14:
            return None
        sport, dport = struct.unpack_from("!HH", seg, 0)
        data_offset = (seg[12] >> 4) * 4
        if data_offset < 20:
            return None
        flags = seg[13]
        window = struct.unpack_from("!H", seg, 14)[0] if len(seg) >= 16 else -1
        payload = max(0, l4_total - data_offset)
        return PacketRecord(ts, src, dst, sport, dport, proto, ip_total, payload, data_offset, flags, window)
    if proto == PROTO_UDP:
        if len(seg) < 4:
            return None
        sport, dport = struct.unpack_from("!HH", seg, 0)
        return PacketRecord(ts, src, dst, sport, dport, proto, ip_total, max(0, l4_total - 8), 8)
    # ICMP and other protocols: no ports, whole L4 counted as payload after an 8-byte header for ICMP.
    hdr = 8 if proto == PROTO_ICMP else 0
    return PacketRecord(ts, src, dst, 0, 0, proto, ip_total, max(0, l4_total - hdr), hdr)


def decode_ip(ts: float, buf: bytes) -> PacketRecord | None:
    if not buf:
        return None
    version = buf[0] >> 4
    if version == 4:
        if len(buf) < 20:
            return None
        ihl = (buf[0] & 0x0F) * 4
        if ihl < 20 or len(buf) < ihl:
            return None
        total_len = struct.unpack_from("!H", buf, 2)[0]
        frag = struct.unpack_from("!H", buf, 6)[0]
        proto = buf[9]
        src = socket.inet_ntop(socket.AF_INET, buf[12:16])
        dst = socket.inet_ntop(socket.AF_INET, buf[16:20])
        if total_len == 0:  # TSO / missing length: fall back to captured length
            total_len = len(buf)
        if frag & 0x1FFF:  # non-first fragment carries no L4 header
            return PacketRecord(ts, src, dst, 0, 0, proto, total_len, max(0, total_len - ihl), 0)
        return _decode_transport(ts, src, dst, proto, total_len, ihl, buf[ihl:])
    if version == 6:
        if len(buf) < 40:
            return None
        payload_len = struct.unpack_from("!H", buf, 4)[0]
        proto = buf[6]
        src = socket.inet_ntop(socket.AF_INET6, buf[8:24])
        dst = socket.inet_ntop(socket.AF_INET6, buf[24:40])
        return _decode_transport(ts, src, dst, proto, payload_len + 40, 40, buf[40:])
    return None


def decode_frame(ts: float, frame: bytes, linktype: int = LINKTYPE_ETHERNET) -> PacketRecord | None:
    """Decode one captured frame into a :class:`PacketRecord`; returns ``None`` for non-IP or malformed frames."""
    try:
        if linktype == LINKTYPE_ETHERNET:
            if len(frame) < 14:
                return None
            eth_type = struct.unpack_from("!H", frame, 12)[0]
            offset = 14
            while eth_type in (0x8100, 0x88A8) and len(frame) >= offset + 4:  # VLAN tags
                eth_type = struct.unpack_from("!H", frame, offset + 2)[0]
                offset += 4
            if eth_type not in (0x0800, 0x86DD):
                return None
            return decode_ip(ts, frame[offset:])
        if linktype in (LINKTYPE_NULL, LINKTYPE_LOOP):  # BSD/Npcap loopback: 4-byte address family
            if len(frame) < 5:
                return None
            return decode_ip(ts, frame[4:])
        if linktype == LINKTYPE_LINUX_SLL:
            if len(frame) < 16:
                return None
            return decode_ip(ts, frame[16:])
        if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4, LINKTYPE_IPV6, 12, 14):
            return decode_ip(ts, frame)
    except (struct.error, ValueError, OSError):
        return None
    return None


def _open_reader(fh):
    magic = fh.read(4)
    fh.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


def iter_pcap(path: str | Path, max_packets: int | None = None,
              start_time: float | None = None, end_time: float | None = None) -> Iterator[PacketRecord]:
    """Stream :class:`PacketRecord` objects from a pcap/pcapng file.

    Malformed frames are skipped (counted in :data:`iter_pcap.stats` of the last call).
    """
    stats = {"frames": 0, "decoded": 0, "skipped": 0}
    iter_pcap.stats = stats
    with open(path, "rb") as fh:
        reader = _open_reader(fh)
        linktype = reader.datalink()
        for ts, frame in reader:
            ts = float(ts)
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                break
            stats["frames"] += 1
            record = decode_frame(ts, frame, linktype)
            if record is None:
                stats["skipped"] += 1
                continue
            stats["decoded"] += 1
            yield record
            if max_packets is not None and stats["decoded"] >= max_packets:
                break


iter_pcap.stats = {}
