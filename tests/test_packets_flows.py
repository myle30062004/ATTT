"""PCAP parser, flow reconstruction and CIC feature extraction."""
import math

import dpkt
import numpy as np
import pytest

from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS, flow_to_features, flows_to_frame
from src.data_processing.flow_builder import FlowBuilder, RunningStats, build_flows
from src.data_processing.pcap_parser import (LINKTYPE_ETHERNET, PROTO_TCP, PROTO_UDP, SYN, decode_frame,
                                             iter_pcap)
from tests.helpers import tcp_frame, tcp_session, udp_frame, write_pcap


def test_decode_tcp_uses_ip_length_not_captured_length():
    frame = tcp_frame("1.1.1.1", "2.2.2.2", 1234, 80, dpkt.tcp.TH_SYN, b"A" * 100)
    truncated = frame[:54]  # header-only capture like CTU-13 *.truncated.pcap
    pkt = decode_frame(1.0, truncated, LINKTYPE_ETHERNET)
    assert pkt is not None
    assert (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port) == ("1.1.1.1", "2.2.2.2", 1234, 80)
    assert pkt.protocol == PROTO_TCP and pkt.payload_length == 100 and pkt.header_length == 20
    assert pkt.has_flag(SYN)


def test_decode_udp_and_invalid_frames():
    pkt = decode_frame(1.0, udp_frame("1.1.1.1", "8.8.8.8", 5353, 53, b"q" * 30))
    assert pkt.protocol == PROTO_UDP and pkt.payload_length == 30 and pkt.dst_port == 53
    assert decode_frame(1.0, b"") is None
    assert decode_frame(1.0, b"\x00" * 10) is None                       # too short
    arp = b"\xff" * 6 + b"\x00" * 6 + b"\x08\x06" + b"\x00" * 28
    assert decode_frame(1.0, arp) is None                                  # non-IP
    bad_ip = b"\xff" * 6 + b"\x00" * 6 + b"\x08\x00" + b"\x45\x00"        # truncated IP header
    assert decode_frame(1.0, bad_ip) is None


def test_unknown_protocol_is_kept_without_ports():
    ip = dpkt.ip.IP(src=b"\x01\x01\x01\x01", dst=b"\x02\x02\x02\x02", p=47, data=b"\x00" * 24)  # GRE
    ip.len = 44
    eth = dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    pkt = decode_frame(0.0, bytes(eth))
    assert pkt.protocol == 47 and pkt.src_port == 0 and pkt.dst_port == 0


def test_iter_pcap_and_bidirectional_flow(tmp_path):
    path = write_pcap(tmp_path / "s.pcap", tcp_session())
    packets = list(iter_pcap(path))
    assert len(packets) == 6
    flows = list(build_flows(packets))
    assert len(flows) == 1
    f = flows[0]
    feats = flow_to_features(f)
    assert f.src_ip == "10.0.0.1" and feats["Dst Port"] == 80
    assert feats["Tot Fwd Pkts"] == 4 and feats["Tot Bwd Pkts"] == 2
    assert feats["TotLen Fwd Pkts"] == 18 and feats["TotLen Bwd Pkts"] == 500
    assert feats["Flow Duration"] == pytest.approx(60_000, rel=1e-6)   # microseconds
    assert feats["Init Fwd Win Byts"] == 64240 and feats["Init Bwd Win Byts"] == 65535
    assert feats["SYN Flag Cnt"] == 2 and feats["FIN Flag Cnt"] == 1
    assert feats["Fwd Act Data Pkts"] == 1
    assert feats["Flow IAT Max"] == pytest.approx(20_000, rel=1e-6)
    assert feats["Down/Up Ratio"] == 0                                   # 2 // 4
    assert f.closed_by == "fin_rst"


def test_incomplete_flow_and_timeouts():
    S = dpkt.tcp.TH_SYN
    b = FlowBuilder(flow_timeout_s=120, idle_timeout_s=10)
    for t, fr in [(0.0, tcp_frame("1.1.1.1", "2.2.2.2", 1, 22, S)), (30.0, tcp_frame("3.3.3.3", "2.2.2.2", 5, 22, S))]:
        b.add_packet(decode_frame(t, fr))
    done = b.pop_finished()
    assert len(done) == 1 and done[0].closed_by == "idle_timeout"
    feats = flow_to_features(done[0])
    assert feats["Tot Bwd Pkts"] == 0 and feats["Flow Duration"] == 0
    assert math.isnan(feats["Flow Byts/s"])          # 0 bytes / 0 s, as CICFlowMeter
    assert feats["Init Bwd Win Byts"] == -1
    b.flush()
    assert len(b.pop_finished()) == 1


def test_max_active_flows_evicts():
    b = FlowBuilder(max_active_flows=2, idle_timeout_s=1e9)
    for i in range(5):
        b.add_packet(decode_frame(float(i), udp_frame("1.1.1.1", "2.2.2.2", 1000 + i, 53)))
    assert len(b.active) == 2 and b.evicted_for_memory == 3


def test_running_stats_matches_numpy():
    data = np.random.default_rng(1).normal(size=500)
    rs = RunningStats()
    for x in data:
        rs.add(float(x))
    assert rs.mean == pytest.approx(data.mean())
    assert rs.std == pytest.approx(data.std(ddof=1))
    assert rs.min == data.min() and rs.max == data.max()


def test_frame_has_canonical_columns_in_order(tmp_path):
    frames = tcp_session() + tcp_session(sport=40001, t0=2000.0)
    flows = list(build_flows(iter_pcap(write_pcap(tmp_path / "x.pcap", frames))))
    df = flows_to_frame(flows)
    assert list(df.columns[: len(CIC_FEATURE_COLUMNS)]) == CIC_FEATURE_COLUMNS
    assert len(df) == 2
    empty = flows_to_frame([])
    assert empty.empty and list(empty.columns[: len(CIC_FEATURE_COLUMNS)]) == CIC_FEATURE_COLUMNS


def test_loopback_linktype_null():
    ip_bytes = tcp_frame("127.0.0.1", "127.0.0.1", 5000, 8080, dpkt.tcp.TH_SYN)[14:]
    pkt = decode_frame(1.0, b"\x02\x00\x00\x00" + ip_bytes, 0)
    assert pkt is not None and pkt.dst_port == 8080
