"""Feature-parity check: this project's PCAP flow extractor vs. CICFlowMeter (CICIDS2018).

    python scripts/fetch_cic_pcap_member.py --day Friday-02-03-2018 --member "pcap/capEC2AMAZ-O4EL3NG-172.31.69.26"
    python scripts/parity_check.py data/raw/cicids2018_pcap/Friday-02-03-2018/capEC2AMAZ-O4EL3NG-172.31.69.26 \
        data/raw/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv --victim 172.31.69.26

Row-level matching is impossible for this day: the CSV has no IP columns, its
timestamps use a 12-hour clock with 1-second resolution, and Bot C&C flows
repeat with identical packet/byte counts every few seconds. Parity is
therefore checked three ways on a raw capture of a documented Bot victim:

1. Distribution parity: for the host's C&C flows (TCP to port 8080) vs. the
   CSV rows labelled Bot, compare per-feature medians and the share of our
   flows whose "size signature" (packet counts, byte totals, max lengths,
   header lengths, initial windows) occurs exactly among CSV Bot rows.
2. End-to-end detection: share of the host's C&C flows that the deployed
   classifier (trained on CSV features only) labels Botnet.
3. Anomaly calibration: share of the host's other flows that the Gaussian
   model flags, compared with its 1% target FPR. Same testbed and day as the
   training data, so a large excess would indicate a feature mismatch rather
   than a change of network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.process_pcap import pcap_to_flows  # noqa: E402
from src.data_processing.dataset_loader import iter_cicids_chunks  # noqa: E402
from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS, NOT_REPRODUCED  # noqa: E402
from src.detection.detection_pipeline import DetectionPipeline  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402

SIGNATURE = ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
             "Bwd Pkt Len Max", "Fwd Header Len", "Bwd Header Len", "Init Fwd Win Byts", "Init Bwd Win Byts"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pcap")
    p.add_argument("csv")
    p.add_argument("--victim", required=True)
    p.add_argument("--cnc-port", type=int, default=8080)
    a = p.parse_args()
    ours = pcap_to_flows(a.pcap, log=lambda *x, **k: None)
    ref = pd.concat(iter_cicids_chunks(Path(a.csv)), ignore_index=True)
    bot_ref = ref[ref["FineLabel"] == "Bot"]
    cnc = ours[(ours["Src IP"] == a.victim) & (ours["Dst Port"] == a.cnc_port) & (ours["Protocol"] == 6)]
    other = ours[~ours.index.isin(cnc.index)]
    report = {"pcap": Path(a.pcap).name, "our_flows": int(len(ours)), "victim_cnc_flows": int(len(cnc)),
              "other_flows": int(len(other)), "csv_bot_rows": int(len(bot_ref))}

    # 1. distribution parity
    sig_ref = set(map(tuple, bot_ref[SIGNATURE].to_numpy()))
    sig_ours = list(map(tuple, cnc[SIGNATURE].to_numpy()))
    report["cnc_size_signature_found_in_csv_bot_rows"] = float(np.mean([s in sig_ref for s in sig_ours])) if sig_ours else None
    rows = []
    for f in CIC_FEATURE_COLUMNS:
        o = cnc[f].replace([np.inf, -np.inf], np.nan).dropna()
        r = bot_ref[f].replace([np.inf, -np.inf], np.nan).dropna()
        mo, mr = (float(o.median()) if len(o) else np.nan), (float(r.median()) if len(r) else np.nan)
        rows.append({"feature": f, "median_ours": mo, "median_cicflowmeter": mr,
                     "relative_diff": abs(mo - mr) / max(abs(mr), 1.0) if np.isfinite(mo) and np.isfinite(mr) else np.nan,
                     "not_reproduced_by_design": f in NOT_REPRODUCED})
    dist = pd.DataFrame(rows).sort_values("relative_diff", ascending=False)
    dist.to_csv(ROOT / "reports" / "metrics" / "feature_parity_distribution.csv", index=False)
    report["features_median_within_5pct"] = int((dist["relative_diff"] <= 0.05).sum())
    report["features_compared"] = int(dist["relative_diff"].notna().sum())

    # 1b. signature-conditioned parity: for the most frequent size signatures shared by our C&C flows
    # and CSV Bot rows, compare every other feature's median within the same signature.
    key = ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Header Len",
           "Bwd Header Len", "Init Fwd Win Byts"]
    shared = (cnc.groupby(key).size().rename("ours").to_frame()
              .join(bot_ref.groupby(key).size().rename("csv"), how="inner")
              .sort_values("ours", ascending=False).head(3))
    per_sig = {}
    for sig in shared.index:
        o = cnc[(cnc[key] == pd.Series(sig, index=key)).all(axis=1)]
        r = bot_ref[(bot_ref[key] == pd.Series(sig, index=key)).all(axis=1)]
        res = {}
        for f in CIC_FEATURE_COLUMNS:
            mo = float(o[f].replace([np.inf, -np.inf], np.nan).median())
            mr = float(r[f].replace([np.inf, -np.inf], np.nan).median())
            if np.isnan(mo) and np.isnan(mr):
                res[f] = True
            elif np.isnan(mo) or np.isnan(mr):
                res[f] = False
            else:
                res[f] = abs(mo - mr) <= 0.10 * abs(mr) + 1e-6
        per_sig[str(dict(zip(key, sig)))] = {"ours": int(shared.loc[sig, "ours"]), "csv": int(shared.loc[sig, "csv"]),
                                              "agree": res}
    agree_all = [f for f in CIC_FEATURE_COLUMNS if f not in NOT_REPRODUCED
                 and all(v["agree"][f] for v in per_sig.values())]
    report["signatures_compared"] = {k: {"ours": v["ours"], "csv": v["csv"],
                                         "disagreeing_features": [f for f, ok in v["agree"].items()
                                                                  if not ok and f not in NOT_REPRODUCED]}
                                     for k, v in per_sig.items()}
    report["parity_safe_features"] = agree_all
    (ROOT / "models" / "preprocessing").mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "preprocessing" / "parity_safe_features.json").write_text(json.dumps(agree_all, indent=2))

    # 2 + 3. end-to-end with the deployed bundle
    pipe = DetectionPipeline(PROJECT_ROOT / "models" / "deployment", load_config())
    # score exactly what the pipeline scores: flows with >= 2 packets (CICFlowMeter never emits 1-packet flows)
    s_cnc = pipe.score_flows(pipe.scorable(cnc))
    s_oth = pipe.score_flows(pipe.scorable(other))
    report["other_flows_scored"] = int(len(s_oth))
    report["other_flows_single_packet_not_scored"] = int(len(other) - len(s_oth))
    report["cnc_predicted_families"] = s_cnc["predicted_family"].value_counts(normalize=True).round(4).to_dict()
    report["cnc_classified_botnet_share"] = float((s_cnc["predicted_family"] == "Botnet").mean())
    report["cnc_flagged_anomalous_share"] = float(s_cnc["is_anomaly"].mean())
    report["other_flows_predicted_families"] = s_oth["predicted_family"].value_counts(normalize=True).round(4).to_dict()
    report["other_flows_flagged_anomalous_share"] = float(s_oth["is_anomaly"].mean())
    report["gaussian_target_fpr"] = 0.01
    out = ROOT / "reports" / "metrics" / "feature_parity_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(dist.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
