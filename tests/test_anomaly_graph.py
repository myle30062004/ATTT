"""Gaussian anomaly detection, host aggregation, beaconing and graph topology."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from src.data_processing.host_aggregator import HOST_FEATURES, host_features, shannon_entropy
from src.graph.beaconing_detector import detect_beaconing
from src.graph.graph_builder import build_graph, node_metrics
from src.graph.topology_analyzer import analyze_topology, gini
from src.models.gaussian_detector import (MultivariateGaussianDetector, UnivariateGaussianDetector,
                                          select_epsilon)


def test_univariate_density_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal([0, 5], [1, 2], size=(5000, 2))
    det = UnivariateGaussianDetector().fit(x)
    q = np.array([[0.5, 4.0]])
    expected = norm.logpdf(0.5, x[:, 0].mean(), x[:, 0].std()) + norm.logpdf(4.0, x[:, 1].mean(), x[:, 1].std())
    assert det.log_density(q)[0] == pytest.approx(expected, rel=1e-9)


def test_multivariate_density_matches_scipy_and_handles_singular_cov():
    rng = np.random.default_rng(1)
    x = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=4000)
    det = MultivariateGaussianDetector(regularization=1e-9).fit(x)
    q = np.array([[1.0, -1.0]])
    ref = multivariate_normal(x.mean(0), np.cov(x, rowvar=False)).logpdf(q)
    assert det.log_density(q)[0] == pytest.approx(ref, rel=1e-6)
    # correlated anomaly: each coordinate normal, but the combination is improbable
    uni = UnivariateGaussianDetector().fit(x)
    assert det.log_density(q)[0] < uni.log_density(q)[0]
    singular = np.c_[x[:, 0], x[:, 0] * 2.0, x[:, 1]]              # perfectly collinear column
    det2 = MultivariateGaussianDetector(regularization=1e-3).fit(singular)
    assert np.isfinite(det2.log_density(singular[:10])).all()


def test_no_underflow_in_high_dimension():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 60))
    det = UnivariateGaussianDetector().fit(x)
    far = np.full((1, 60), 30.0)
    assert np.isfinite(det.log_density(far)).all()                 # p(x) itself would underflow to 0


def test_epsilon_selection_separates_and_respects_fpr():
    rng = np.random.default_rng(3)
    normal = rng.normal(size=(3000, 3))
    attack = rng.normal(4, 1, size=(300, 3))
    det = UnivariateGaussianDetector().fit(normal[:2000])
    xv = np.vstack([normal[2000:], attack])
    yv = np.r_[np.zeros(1000), np.ones(300)]
    best = select_epsilon(det.log_density(xv), yv)
    det.log_epsilon = best["log_epsilon"]
    pred = det.predict(xv)
    assert best["score"] > 0.9 and pred[yv == 1].mean() > 0.9 and pred[yv == 0].mean() < 0.05
    capped = select_epsilon(det.log_density(xv), yv, max_fpr=0.001)
    assert capped["fpr"] <= 0.001


def test_detector_requires_samples():
    with pytest.raises(ValueError):
        UnivariateGaussianDetector().fit(np.zeros((1, 3)))


def _flows(rows):
    return pd.DataFrame(rows, columns=["src_ip", "dst_ip", "dst_port", "protocol", "start_time", "duration",
                                       "packets", "bytes"])


def test_host_features_scanner_vs_beacon():
    scan = [("10.0.0.5", "10.0.0.9", p, 6, i * 0.01, 0.0, 1, 0) for i, p in enumerate(range(1, 201))]
    beacon = [("10.0.0.7", "6.6.6.6", 443, 6, i * 60.0, 0.5, 6, 300) for i in range(30)]
    hf = host_features(_flows(scan + beacon)).set_index("src_ip")
    assert list(hf.columns) == HOST_FEATURES
    assert hf.loc["10.0.0.5", "unique_dst_ports"] == 200 and hf.loc["10.0.0.5", "dst_port_entropy"] > 7
    assert hf.loc["10.0.0.7", "periodicity_score"] == pytest.approx(1.0)
    assert hf.loc["10.0.0.7", "repeated_connections"] == 29
    assert shannon_entropy(pd.Series([1, 1, 1])) == 0.0
    assert host_features(_flows([])).empty


def test_beaconing_detection_with_jitter_and_whitelist():
    rng = np.random.default_rng(4)
    t = np.cumsum(60 + rng.normal(0, 1.5, 40))
    beacon = [("10.0.0.7", "6.6.6.6", 443, 6, ts, 0.5, 6, 300) for ts in t]
    ntp = [("10.0.0.8", "1.1.1.1", 123, 17, i * 64.0, 0.0, 2, 96) for i in range(40)]
    human = [("10.0.0.9", "5.5.5.5", 80, 6, ts, 1.0, 10, 5000) for ts in np.cumsum(rng.exponential(60, 40))]
    burst = [("10.0.0.10", "4.4.4.4", 80, 6, i * 0.001, 0.0, 1, 0) for i in range(50)]   # merged burst
    res = detect_beaconing(_flows(beacon + ntp + human + burst)).set_index("src_ip")
    assert bool(res.loc["10.0.0.7", "is_beacon"]) and res.loc["10.0.0.7", "cv"] < 0.1
    assert not bool(res.loc["10.0.0.8", "is_beacon"])          # whitelisted NTP
    assert not bool(res.loc["10.0.0.9", "is_beacon"])          # Poisson-like human traffic, CV ~ 1
    assert "10.0.0.10" not in res.index                        # bursts merged -> too few connections


def test_graph_metrics_and_topology_centralized_vs_p2p():
    bots = [f"10.0.0.{i}" for i in range(1, 9)]
    central = [(b, "6.6.6.6", 6667, 6, float(i), 1.0, 5, 100) for b in bots for i in range(10)]
    g = build_graph(_flows(central))
    m = node_metrics(g).set_index("ip")
    assert m.loc["6.6.6.6", "in_degree"] == 8 and m.loc["6.6.6.6", "pagerank"] == m["pagerank"].max()
    topo = analyze_topology(g, bots)
    assert topo["verdict"] == "centralized" and "6.6.6.6" in topo["indicators"]["controllers"]
    # same 8 hosts plus 50 other hosts that all use the same resolver: the resolver is popular,
    # so it is not evidence of a controller
    normal_rows = [(f"10.1.0.{i}", "9.9.9.9", 53, 17, 0.0, 0.1, 2, 100) for i in range(50)]
    shared_dns = [(b, "9.9.9.9", 53, 17, 0.0, 0.1, 2, 100) for b in bots]
    g_dns = build_graph(_flows(normal_rows + shared_dns))
    assert analyze_topology(g_dns, bots)["verdict"] == "no_coordinated_structure"
    rng = np.random.default_rng(5)
    peers = [f"8.8.{i // 250}.{i % 250}" for i in range(400)]
    p2p_rows = []
    for b in bots:
        for p in rng.choice(peers, 80, replace=False):
            p2p_rows.append((b, p, 5000, 17, 0.0, 0.1, 1, 60))
        for other in bots:
            if other != b:
                p2p_rows.append((b, other, 5000, 17, 0.0, 0.1, 1, 60))
    topo2 = analyze_topology(build_graph(_flows(p2p_rows)), bots)
    assert topo2["verdict"] == "p2p"
    assert analyze_topology(g, ["10.0.0.1"])["verdict"] == "single_host_insufficient_evidence"
    assert gini([1, 1, 1]) == pytest.approx(0.0) and gini([0, 0, 10]) > 0.6


def test_gaussian_mixture_handles_multimodal_normal_traffic():
    from src.models.gaussian_detector import GaussianMixtureDetector, threshold_at_fpr
    rng = np.random.default_rng(7)
    normal = np.vstack([rng.normal(-4, 0.5, size=(2000, 2)), rng.normal(4, 0.5, size=(2000, 2))])
    between = rng.normal(0, 0.3, size=(200, 2))          # attacks between the two normal modes
    gmm = GaussianMixtureDetector(n_components=2).fit(normal)
    single = MultivariateGaussianDetector().fit(normal)
    thr_g = threshold_at_fpr(gmm.log_density(normal), 0.01)
    thr_s = threshold_at_fpr(single.log_density(normal), 0.01)
    assert (gmm.log_density(between) < thr_g).mean() > 0.95      # mixture flags them
    assert (single.log_density(between) < thr_s).mean() < 0.05   # single Gaussian misses them
