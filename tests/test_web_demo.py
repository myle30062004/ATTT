"""Web demo: API smoke tests on the real deployment bundle (skipped when it is absent)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "deployment" / "bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="deployment bundle not built")


@pytest.fixture(scope="module")
def client():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from web.app import app
    with TestClient(app) as c:
        yield c


def test_info_and_results(client):
    info = client.get("/api/info").json()
    assert info["class_names"][0] == "BENIGN" and len(info["classifier_features"]) == 24
    res = client.get("/api/results").json()
    assert {"validation", "test", "anomaly"} <= set(res)


def test_single_flow_roundtrip(client):
    flow = client.get("/api/flow/random", params={"family": "Botnet"}).json()
    out = client.post("/api/flow/predict", json={"features": flow["features"]}).json()
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["predicted"] in out["probabilities"]
    assert len(out["contributions"]) == 10


def test_synthetic_pcap_detects_scan_sweep_and_beacon(tmp_path):
    from web import make_samples, services

    pcap = make_samples.synthetic()

    class Job:
        def update(self, *_):
            pass

    r = services.analyze_pcap(Job(), pcap, {"id": "synthetic"})
    kinds = {(g["src_ip"], g["attack_type"]) for g in r["alert_groups"]}
    assert ("192.168.10.66", "PortScan") in kinds
    assert ("192.168.10.67", "HostSweep") in kinds
    assert ("192.168.10.7", "Beaconing") in kinds
    # realistic background web traffic should stay quiet at flow level
    benign_alerts = [g for g in r["alert_groups"] if g["src_ip"].startswith("192.168.10.") and
                     g["src_ip"].split(".")[-1] in {"1", "2", "3", "4", "5"} and g["detection_module"] != "GraphTopology"]
    assert not benign_alerts
