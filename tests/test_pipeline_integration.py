"""End-to-end: synthetic training -> bundle -> PCAP -> flows -> detection -> alerts (offline and replay)."""
import json

import dpkt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data_processing.feature_extractor import flows_to_frame
from src.data_processing.flow_builder import build_flows
from src.data_processing.pcap_parser import iter_pcap
from src.data_processing.preprocessing import FlowCleaner, LogScaler
from src.detection.alert_manager import AlertManager
from src.detection.bundle import save_bundle
from src.detection.decision_engine import DecisionEngine, FlowVerdict
from src.detection.detection_pipeline import DetectionPipeline, MissingModelError, check_feature_parity
from src.models.gaussian_detector import MultivariateGaussianDetector, select_epsilon
from src.utils.config import load_config
from tests.helpers import tcp_frame, tcp_session, write_pcap

S, A = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK


def _traffic(t0: float) -> list:
    frames = []
    for i in range(40):                                        # normal web sessions
        frames += tcp_session(src="10.0.0.1", sport=41000 + i, t0=t0 + i * 3.0)
    for p in range(1, 301):                                    # SYN scan from 10.0.0.66
        frames.append((t0 + 200 + p * 0.002, tcp_frame("10.0.0.66", "10.0.0.2", 55555, p, S)))
    for i in range(20):                                        # beacon every 30 s
        frames += tcp_session(src="10.0.0.7", dst="6.6.6.6", sport=50000 + i, dport=8080, t0=t0 + i * 30.0,
                              req=b"ping", resp=b"cmd")
    return sorted(frames, key=lambda f: f[0])


def _flows_frame(tmp_path, name, t0):
    pcap = write_pcap(tmp_path / name, _traffic(t0))
    return pcap, flows_to_frame(build_flows(iter_pcap(pcap), idle_timeout_s=5))


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    _, train = _flows_frame(tmp, "train.pcap", 1000.0)
    train["Family"] = np.where(train["Src IP"] == "10.0.0.66", "PortScan", "BENIGN")
    cleaner = FlowCleaner().fit(train)
    feats = [c for c in ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Bwd Pkts", "SYN Flag Cnt",
                         "Flow IAT Mean", "Init Bwd Win Byts", "Fwd Pkt Len Max"] if c in cleaner.kept_columns]
    x = cleaner.transform(train)[feats]
    classes = ["BENIGN", "PortScan"]
    y = (train["Family"] == "PortScan").astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=30, random_state=0).fit(x, y)
    benign = x[y == 0]
    scaler = LogScaler("standard").fit(benign)
    gauss = MultivariateGaussianDetector(1e-2).fit(scaler.transform(benign))
    gauss.log_epsilon = select_epsilon(gauss.log_density(scaler.transform(x)), y)["log_epsilon"]
    return save_bundle(tmp / "deployment", cleaner=cleaner, classifier=clf, classifier_features=feats,
                       class_names=classes, gaussian=gauss, gaussian_scaler=scaler, gaussian_features=feats,
                       importances=dict(zip(feats, clf.feature_importances_)))


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(MissingModelError):
        DetectionPipeline(tmp_path / "nope")


def test_offline_pipeline_detects_scan_and_beacon(tmp_path, bundle_dir):
    cfg = load_config()
    pipe = DetectionPipeline(bundle_dir, cfg)
    _, frame = _flows_frame(tmp_path, "test.pcap", 5000.0)
    assert check_feature_parity(frame) == []
    alerts, suspects, scores = pipe.process_flows(frame)
    # unanswered SYNs are single-packet flows: not scored per flow (CICFlowMeter never emits them) ...
    assert len(scores) == int(((frame["Tot Fwd Pkts"] + frame["Tot Bwd Pkts"]) >= 2).sum())
    benign_fp = [a for a in alerts if a.src_ip == "10.0.0.1"]
    assert len(benign_fp) <= 4
    host_alerts, report = pipe.analyze_hosts(pipe.host_flows_from_frame(frame), suspects)
    # ... but the scan is caught by the host-level rule
    scan = [a for a in host_alerts if a.src_ip == "10.0.0.66"]
    assert any(a.attack_type == "PortScan" and a.detection_module == "HostScanRule" for a in scan)
    assert scan[0].relevant_features["unique_dst_ports"] == 300
    beacons = report["beacons"]
    assert beacons.loc[beacons.src_ip == "10.0.0.7", "is_beacon"].any()
    assert any(a.attack_type == "Beaconing" and a.src_ip == "10.0.0.7" for a in host_alerts)
    sample = scan[0].to_dict()
    for key in ("timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "detection_module",
                "attack_type", "confidence", "severity", "reason", "relevant_features"):
        assert key in sample


def test_pipeline_handles_empty_and_missing_columns(bundle_dir):
    pipe = DetectionPipeline(bundle_dir, load_config())
    alerts, suspects, _ = pipe.process_flows(flows_to_frame([]))
    assert alerts == [] and suspects == set()
    _, report = pipe.analyze_hosts(pd.DataFrame())
    assert report["beacons"].empty


def test_decision_engine_uncertain_and_unknown():
    eng = DecisionEngine(uncertain_threshold=0.6)
    meta = {"timestamp": "t", "src_ip": "a", "dst_ip": "b", "src_port": 1, "dst_port": 2, "protocol": 6}
    alert, to_host = eng.decide_flow(FlowVerdict(False, 1.0, "DoS", 0.45, {}, {}), meta)
    assert alert.attack_type == "Uncertain" and to_host
    alert, to_host = eng.decide_flow(FlowVerdict(True, 99.0, "BENIGN", 0.7, {}, {}), meta)
    assert alert.attack_type == "Unknown" and alert.detection_module == "GaussianAnomaly" and to_host
    alert, to_host = eng.decide_flow(FlowVerdict(False, 1.0, "BENIGN", 0.99, {}, {}), meta)
    assert alert is None and not to_host
    alert, _ = eng.decide_flow(FlowVerdict(True, 50.0, "Botnet", 0.95, {}, {}), meta)
    assert alert.severity == "CRITICAL"


def test_alert_manager_outputs_and_suppression(tmp_path):
    eng = DecisionEngine()
    meta = {"timestamp": "t", "src_ip": "a", "dst_ip": "b", "src_port": 1, "dst_port": 2, "protocol": 6}
    alert, _ = eng.decide_flow(FlowVerdict(True, 9.0, "DoS", 0.9, {}, {"x": 1.0}), meta)
    am = AlertManager(tmp_path, console=False, use_sqlite=True, suppress_window_s=10)
    assert am.emit(alert, now=0.0) and not am.emit(alert, now=5.0) and am.emit(alert, now=20.0)
    am.close()
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "1 similar alerts suppressed" in json.loads(lines[1])["reason"]
    assert len(pd.read_csv(tmp_path / "alerts.csv")) == 2


def test_realtime_replay_matches_offline(tmp_path, bundle_dir):
    from src.realtime.monitor import run_monitor

    pcap, frame = _flows_frame(tmp_path, "replay.pcap", 9000.0)
    summary = run_monitor(pcap=str(pcap), bundle_dir=str(bundle_dir), alert_dir=str(tmp_path / "alerts"),
                          quiet=True)
    assert summary["packets"] == sum(1 for _ in iter_pcap(pcap))
    assert summary["flows_scored"] == len(frame)                 # same flow reconstruction as offline
    assert summary["alerts"] > 0 and summary["dropped_by_capture"] == 0
    assert summary["processing_latency_ms_p50"] is not None
