"""Multi-stage detection pipeline shared by offline PCAP analysis and realtime monitoring.

    packets -> flows (FlowBuilder) -> CIC features (feature_extractor)
            -> FlowCleaner (train-fitted) -> Gaussian anomaly score
            -> attack classifier (every flow) -> DecisionEngine -> alerts
    flows (sliding window) -> host features + beaconing + graph topology
            -> host-level botnet model / heuristics -> alerts

The *deployment bundle* (``models/deployment``) is produced by
``scripts/train.py --stage bundle`` and contains everything needed:
cleaner, feature lists, scalers, Gaussian detector, classifier, class names,
and optionally a host-level botnet model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data_processing.feature_extractor import CIC_FEATURE_COLUMNS
from ..data_processing.host_aggregator import HOST_FEATURES, flows_from_cic_frame, host_features
from ..graph.beaconing_detector import detect_beaconing
from ..graph.graph_builder import build_graph
from ..graph.topology_analyzer import analyze_topology
from .decision_engine import Alert, DecisionEngine, FlowVerdict


class MissingModelError(FileNotFoundError):
    pass


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="milliseconds")


class DetectionPipeline:
    def __init__(self, bundle_dir: str | Path, config: dict | None = None) -> None:
        bundle_dir = Path(bundle_dir)
        path = bundle_dir / "bundle.joblib"
        if not path.exists():
            raise MissingModelError(f"deployment bundle not found at {path}; run scripts/train.py --stage bundle")
        bundle = joblib.load(path)
        self.meta = json.loads((bundle_dir / "bundle.json").read_text())
        self.cleaner = bundle["cleaner"]
        self.clf = bundle["classifier"]
        self.clf_features: list[str] = bundle["classifier_features"]
        self.clf_scaler = bundle.get("classifier_scaler")
        self.class_names: list[str] = bundle["class_names"]
        self.gauss = bundle["gaussian"]
        self.gauss_features: list[str] = bundle["gaussian_features"]
        self.gauss_scaler = bundle["gaussian_scaler"]
        self.importances: dict = bundle.get("importances", {})
        self.host_model = bundle.get("host_model")
        self.host_model_features: list[str] = bundle.get("host_model_features", HOST_FEATURES)
        cfg = config or {}
        self.engine = DecisionEngine(cfg.get("classifier", {}).get("uncertain_threshold", 0.6))
        self.botnet_cfg = cfg.get("botnet", {})
        self.min_packets = int(cfg.get("flow", {}).get("min_packets_for_scoring", 2))
        self.missing_feature_warnings: set[str] = set()
        self.skipped_short_flows = 0

    def scorable(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Flows that flow-level models may score: CICFlowMeter never emits flows with fewer
        than two packets, so such flows are outside the training distribution. They are
        still used for host-level analysis (e.g. unanswered SYNs of a port scan)."""
        if frame.empty or "Tot Fwd Pkts" not in frame:
            return frame
        keep = (frame["Tot Fwd Pkts"] + frame["Tot Bwd Pkts"]) >= self.min_packets
        self.skipped_short_flows += int((~keep).sum())
        return frame.loc[keep]

    # ---------------------------------------------------------------- flows
    def _check_features(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.cleaner.kept_columns if c not in frame.columns]
        if missing:
            self.missing_feature_warnings.update(missing)

    def score_flows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return anomaly score/flag, predicted family and confidence for each flow."""
        out = pd.DataFrame(index=frame.index)
        if frame.empty:
            for c in ("anomaly_score", "is_anomaly", "predicted_family", "confidence"):
                out[c] = pd.Series(dtype=float)
            return out
        self._check_features(frame)
        clean = self.cleaner.transform(frame)
        xg = self.gauss_scaler.transform(clean[self.gauss_features])
        log_p = self.gauss.log_density(xg)
        xc = clean[self.clf_features]
        if self.clf_scaler is not None:
            xc = self.clf_scaler.transform(xc)
        proba = self.clf.predict_proba(xc)
        pred = proba.argmax(1)
        out["anomaly_score"] = -log_p
        out["is_anomaly"] = log_p < self.gauss.log_epsilon
        out["predicted_family"] = [self.class_names[i] for i in pred]
        out["confidence"] = proba.max(1)
        out.attrs["gauss_input"] = xg
        out.attrs["clean"] = clean
        return out

    def _anomaly_explanation(self, z_row: np.ndarray, raw_row: pd.Series, top: int = 5) -> dict:
        idx = np.argsort(-np.abs(z_row))[:top]
        return {self.gauss_features[i]: {"value": float(raw_row[self.gauss_features[i]]),
                                         "z_vs_normal": round(float(z_row[i]), 2)} for i in idx}

    def _classifier_explanation(self, raw_row: pd.Series, top: int = 5) -> dict:
        names = list(self.importances)[:top] or self.clf_features[:top]
        return {n: float(raw_row[n]) for n in names if n in raw_row.index}

    def process_flows(self, frame: pd.DataFrame) -> tuple[list[Alert], set[str], pd.DataFrame]:
        """Score flows and convert verdicts into alerts. ``frame`` needs CIC features + metadata columns.

        Only flows with >= ``min_packets_for_scoring`` packets are scored (see :meth:`scorable`);
        the returned ``scores`` frame covers those flows only."""
        frame = self.scorable(frame)
        scores = self.score_flows(frame)
        alerts: list[Alert] = []
        suspects: set[str] = set()
        if frame.empty:
            return alerts, suspects, scores
        xg = scores.attrs["gauss_input"]
        clean = scores.attrs["clean"]
        variances = self.gauss.var if hasattr(self.gauss, "var") else np.diag(self.gauss.cov)
        z = (xg - self.gauss.mu) / np.sqrt(variances)
        flagged = np.flatnonzero(scores["is_anomaly"].to_numpy() | (scores["predicted_family"] != "BENIGN").to_numpy())
        for i in flagged:
            row = frame.iloc[i]
            verdict = FlowVerdict(
                is_anomaly=bool(scores["is_anomaly"].iat[i]), anomaly_score=float(scores["anomaly_score"].iat[i]),
                predicted_family=scores["predicted_family"].iat[i], confidence=float(scores["confidence"].iat[i]),
                anomaly_features=self._anomaly_explanation(z[i], clean.iloc[i]),
                classifier_features=self._classifier_explanation(clean.iloc[i]))
            meta = {"timestamp": _iso(row.get("Start Time", 0.0)), "src_ip": row.get("Src IP", "?"),
                    "dst_ip": row.get("Dst IP", "?"), "src_port": row.get("Src Port", 0),
                    "dst_port": row.get("Dst Port", 0), "protocol": row.get("Protocol", 0)}
            alert, to_host = self.engine.decide_flow(verdict, meta)
            if alert is not None:
                alerts.append(alert)
            if to_host:
                suspects.add(meta["src_ip"])
        return alerts, suspects, scores

    # ---------------------------------------------------------------- hosts
    def analyze_hosts(self, host_flows: pd.DataFrame, suspects: set[str] | None = None) -> tuple[list[Alert], dict]:
        """Host-level botnet analysis over a window of flows (host-aggregator schema)."""
        alerts: list[Alert] = []
        report: dict = {"beacons": pd.DataFrame(), "host_scores": pd.DataFrame(), "topology": {}}
        if host_flows.empty:
            return alerts, report
        now = _iso(host_flows["start_time"].max())
        cfg = self.botnet_cfg
        beacons = detect_beaconing(host_flows, min_connections=cfg.get("beacon_min_connections", 8),
                                   cv_threshold=cfg.get("beacon_cv_threshold", 0.25),
                                   min_period_s=cfg.get("beacon_min_period_s", 2.0),
                                   whitelist_ports=tuple(cfg.get("beacon_whitelist_ports", [53, 123])))
        report["beacons"] = beacons
        flagged_hosts = set(suspects or set())
        for b in beacons[beacons["is_beacon"]].itertuples(index=False):
            flagged_hosts.add(b.src_ip)
            alerts.append(self.engine.host_alert(
                b.src_ip, now, "BeaconingDetector", "Beaconing", b.beacon_score, "MEDIUM",
                f"{b.n_connections} connections to {b.dst_ip}:{b.dst_port} every ~{b.interval_median:.1f}s "
                f"(CV={b.cv:.3f}, robust CV={b.robust_cv:.3f}); periodic call-home pattern, may also be a "
                f"legitimate scheduled service.",
                {"interval_mean_s": round(b.interval_mean, 3), "cv": round(b.cv, 4),
                 "n_connections": int(b.n_connections), "span_s": round(b.span_s, 1)},
                dst_ip=b.dst_ip, dst_port=b.dst_port))
        hosts = host_features(host_flows)
        min_ports = cfg.get("scan_min_unique_ports", 50)
        min_ips = cfg.get("sweep_min_unique_ips", 50)
        min_failed = cfg.get("scan_min_failed_ratio", 0.5)
        for h in hosts.itertuples(index=False):
            kind = None
            if h.unique_dst_ports >= min_ports and h.failed_ratio >= min_failed:
                kind = "PortScan"
            elif h.unique_dst_ips >= min_ips and h.failed_ratio >= min_failed:
                kind = "HostSweep"
            if kind:
                flagged_hosts.add(h.src_ip)
                alerts.append(self.engine.host_alert(
                    h.src_ip, now, "HostScanRule", kind, float(h.failed_ratio), "HIGH",
                    f"{h.unique_dst_ports} distinct destination ports / {h.unique_dst_ips} hosts in the window, "
                    f"{h.failed_ratio:.0%} of {h.total_connections} connections unanswered, "
                    f"{h.syn_without_ack} SYN without ACK, port entropy {h.dst_port_entropy:.2f} bits.",
                    {"unique_dst_ports": int(h.unique_dst_ports), "unique_dst_ips": int(h.unique_dst_ips),
                     "failed_ratio": round(float(h.failed_ratio), 3), "syn_without_ack": int(h.syn_without_ack),
                     "connections_per_sec": round(float(h.connections_per_sec), 2)}))
        if self.host_model is not None and not hosts.empty:
            proba = self.host_model.predict_proba(hosts[self.host_model_features].astype(np.float64))[:, 1]
            hosts["botnet_score"] = proba
            for h in hosts.loc[hosts["botnet_score"] >= 0.5].itertuples(index=False):
                flagged_hosts.add(h.src_ip)
                alerts.append(self.engine.host_alert(
                    h.src_ip, now, "HostBehaviourModel", "Botnet", h.botnet_score,
                    "HIGH" if h.botnet_score >= 0.8 else "MEDIUM",
                    "Host-level behaviour resembles CTU-13 botnet hosts (connection fan-out, failure ratio, "
                    "destination entropy, timing regularity).",
                    {f: float(getattr(h, f)) for f in ("unique_dst_ips", "unique_dst_ports", "failed_ratio",
                                                       "dst_ip_entropy", "periodicity_score")}))
        report["host_scores"] = hosts
        if len(flagged_hosts) >= 2:
            topo = analyze_topology(build_graph(host_flows), sorted(flagged_hosts))
            report["topology"] = topo
            if topo["verdict"] in ("centralized", "p2p"):
                alerts.append(self.engine.host_alert(
                    ",".join(sorted(flagged_hosts))[:200], now, "GraphTopology", f"Botnet-{topo['verdict']}",
                    max(topo["votes"].values()) / 3.0, "HIGH",
                    f"Suspicious hosts show {topo['verdict']} communication structure "
                    f"(votes {topo['votes']}).", {k: v for k, v in topo["indicators"].items()
                                                  if k not in ("top_shared_destinations",)}))
        return alerts, report

    @staticmethod
    def host_flows_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return flows_from_cic_frame(frame)


def check_feature_parity(frame: pd.DataFrame) -> list[str]:
    """Columns of the canonical CIC schema missing from ``frame`` (empty list = parity)."""
    return [c for c in CIC_FEATURE_COLUMNS if c not in frame.columns]
