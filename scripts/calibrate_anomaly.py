"""Recalibrate the Gaussian anomaly threshold (epsilon) on normal traffic of the deployment network.

    python scripts/calibrate_anomaly.py normal_traffic.pcap --target-fpr 0.01 [--bundle models/deployment]

The mixture model learned what "normal" looks like in the CICIDS2018 testbed.
On another network (or on PCAP-reconstructed flows, whose features match
CICFlowMeter only approximately) the same epsilon gives a different
false-positive rate: 16.6% instead of 1% on a Bot victim's benign flows of the
CICIDS2018 testbed itself (reports/metrics/feature_parity_check.json).
This script scores the scorable flows (>= 2 packets) of a capture that is
known to contain only normal traffic and moves epsilon to the requested
quantile of their log-density. The model parameters are not changed; the old
and new epsilon are recorded in bundle.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.process_pcap import pcap_to_flows  # noqa: E402
from src.detection.detection_pipeline import DetectionPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pcap", help="capture of NORMAL traffic from the deployment network")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--bundle", default=str(ROOT / "models" / "deployment"))
    p.add_argument("--min-flows", type=int, default=500)
    a = p.parse_args()
    pipe = DetectionPipeline(a.bundle, load_config())
    flows = pipe.scorable(pcap_to_flows(a.pcap, log=lambda *x, **k: None))
    if len(flows) < a.min_flows:
        raise SystemExit(f"only {len(flows)} scorable flows; need at least {a.min_flows} for a stable quantile")
    clean = pipe.cleaner.transform(flows)
    log_p = pipe.gauss.log_density(pipe.gauss_scaler.transform(clean[pipe.gauss_features]))
    old = float(pipe.gauss.log_epsilon)
    new = float(np.quantile(log_p, a.target_fpr))
    before = float((log_p < old).mean())
    bundle_path = Path(a.bundle) / "bundle.joblib"
    bundle = joblib.load(bundle_path)
    bundle["gaussian"].log_epsilon = new
    joblib.dump(bundle, bundle_path)
    meta_path = Path(a.bundle) / "bundle.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("calibrations", []).append({
        "utc": datetime.now(timezone.utc).isoformat(), "pcap": str(a.pcap), "scorable_flows": int(len(flows)),
        "target_fpr": a.target_fpr, "old_log_epsilon": old, "new_log_epsilon": new,
        "flagged_share_before": before})
    meta["gaussian_log_epsilon"] = new
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta["calibrations"][-1], indent=2))


if __name__ == "__main__":
    main()
