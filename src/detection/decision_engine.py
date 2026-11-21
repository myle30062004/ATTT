"""Combines anomaly detection, attack classification and botnet analysis into alerts.

Decision rules per flow (``a`` = Gaussian anomaly flag, ``c`` = classifier
family, ``p`` = classifier confidence = max class probability):

=====================  ===========================  =================================
anomaly ``a``          classifier                   decision
=====================  ===========================  =================================
any                    attack, p >= tau             ATTACK alert (family = c)
any                    attack, p <  tau             UNCERTAIN alert -> host to botnet analysis
yes                    BENIGN                       UNKNOWN anomaly alert -> host to botnet analysis
no                     BENIGN                       no alert (monitoring counters only)
=====================  ===========================  =================================

The classifier runs on *every* flow, not only on anomalies: the Gaussian
model misses attacks that look statistically normal, so "normal" is never
treated as proof of safety.

Confidence semantics: ``p`` is the classifier's estimated class probability
(a model score calibrated only on CICIDS2018-like traffic), and the anomaly
score is ``-log p(x)`` under the normal-traffic density. Neither is the
probability that an attack is happening; reports call them *scores*.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITIES = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
FAMILY_SEVERITY = {
    "DoS": "HIGH", "DDoS": "HIGH", "BruteForce": "HIGH", "WebAttack": "HIGH",
    "Infiltration": "CRITICAL", "Botnet": "CRITICAL", "PortScan": "MEDIUM",
}
PROTOCOL_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP"}


def bump(severity: str, steps: int = 1) -> str:
    i = min(len(SEVERITIES) - 1, max(0, SEVERITIES.index(severity) + steps))
    return SEVERITIES[i]


@dataclass
class Alert:
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    detection_module: str
    attack_type: str
    confidence: float
    severity: str
    reason: str
    anomaly_score: float | None = None
    relevant_features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowVerdict:
    is_anomaly: bool
    anomaly_score: float
    predicted_family: str
    confidence: float
    anomaly_features: dict
    classifier_features: dict


class DecisionEngine:
    def __init__(self, uncertain_threshold: float = 0.6, benign_label: str = "BENIGN") -> None:
        self.tau = uncertain_threshold
        self.benign = benign_label

    def decide_flow(self, verdict: FlowVerdict, meta: dict) -> tuple[Alert | None, bool]:
        """Return (alert or None, send_host_to_botnet_analysis)."""
        common = dict(
            timestamp=meta["timestamp"], src_ip=meta["src_ip"], dst_ip=meta["dst_ip"],
            src_port=int(meta.get("src_port", 0)), dst_port=int(meta.get("dst_port", 0)),
            protocol=PROTOCOL_NAMES.get(int(meta.get("protocol", 0)), str(meta.get("protocol"))),
            anomaly_score=round(float(verdict.anomaly_score), 3),
        )
        is_attack = verdict.predicted_family != self.benign
        if is_attack and verdict.confidence >= self.tau:
            severity = FAMILY_SEVERITY.get(verdict.predicted_family, "MEDIUM")
            if verdict.confidence < 0.8:
                severity = bump(severity, -1)
            agree = " Gaussian model also flags the flow as anomalous." if verdict.is_anomaly else ""
            return Alert(**common, detection_module="AttackClassifier", attack_type=verdict.predicted_family,
                         confidence=round(verdict.confidence, 4), severity=severity,
                         reason=f"Classifier assigns {verdict.predicted_family} with score "
                                f"{verdict.confidence:.2f}.{agree}",
                         relevant_features=verdict.classifier_features), False
        if is_attack:
            return Alert(**common, detection_module="AttackClassifier", attack_type="Uncertain",
                         confidence=round(verdict.confidence, 4), severity="LOW",
                         reason=f"Most likely {verdict.predicted_family} but score {verdict.confidence:.2f} "
                                f"< {self.tau}; type not asserted, host queued for behaviour analysis.",
                         relevant_features=verdict.classifier_features), True
        if verdict.is_anomaly:
            return Alert(**common, detection_module="GaussianAnomaly", attack_type="Unknown",
                         confidence=round(1.0 - verdict.confidence, 4), severity="MEDIUM" if verdict.confidence < 0.9 else "LOW",
                         reason="Flow is improbable under the model of normal traffic but matches no known "
                                "attack class; most atypical features listed.",
                         relevant_features=verdict.anomaly_features), True
        return None, False

    @staticmethod
    def host_alert(src_ip: str, timestamp: str, module: str, attack_type: str, score: float,
                   severity: str, reason: str, features: dict, dst_ip: str = "*", dst_port: int = 0) -> Alert:
        return Alert(timestamp=timestamp, src_ip=src_ip, dst_ip=dst_ip, src_port=0, dst_port=dst_port,
                     protocol="*", detection_module=module, attack_type=attack_type,
                     confidence=round(float(score), 4), severity=severity, reason=reason,
                     relevant_features=features)
