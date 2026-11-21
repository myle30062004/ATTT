"""Score a lab run: alerts (alerts.jsonl) against ground truth written by traffic_generators.py.

    python scripts/lab/evaluate_lab_run.py --alerts logs/lab/alerts/alerts.jsonl --truth logs/lab/truth.json \
        --summary logs/lab/alerts/run_summary.json --out reports/experiments/lab_results.json

Per scenario:
* detected: at least one alert whose source is the attacker (or, for host-level
  alerts, whose src list contains it) inside [start - 5 s, end + flow timeout];
* first-alert latency: alert *emission* time is not stored separately, so latency
  is measured as (timestamp of the first matching alert's flow start) - scenario
  start, plus the processor's measured processing latency (run_summary);
* precision: share of alerts in the scenario window that point at the attacker;
* alert types: which modules / attack types fired.
Alerts outside every scenario window are counted as background alerts (the
lab host's own normal traffic), giving an empirical false-alert count.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alerts", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--summary")
    p.add_argument("--out")
    p.add_argument("--grace", type=float, default=130.0, help="seconds after scenario end still attributed to it")
    a = p.parse_args()
    truth = json.loads(Path(a.truth).read_text())
    rows = [json.loads(line) for line in Path(a.alerts).read_text().splitlines() if line.strip()]
    alerts = pd.DataFrame(rows)
    if alerts.empty:
        alerts = pd.DataFrame(columns=["timestamp", "src_ip", "dst_ip", "attack_type", "detection_module", "severity"])
    alerts["t"] = alerts["timestamp"].map(lambda s: datetime.fromisoformat(s).timestamp())
    results, attributed = [], pd.Series(False, index=alerts.index)
    for run in truth:
        win = (alerts["t"] >= run["start"] - 5) & (alerts["t"] <= run["end"] + a.grace)
        mine = win & alerts["src_ip"].astype(str).str.contains(run["attacker_ip"], regex=False)
        attributed |= mine
        first = alerts.loc[mine, "t"].min() if mine.any() else None
        results.append({
            "scenario": run["scenario"], "attacker_ip": run["attacker_ip"], "duration_s": round(run["end"] - run["start"], 1),
            "detected": bool(mine.any()), "alerts_from_attacker": int(mine.sum()),
            "alerts_in_window": int(win.sum()),
            "precision_in_window": float(mine.sum() / win.sum()) if win.any() else None,
            "first_alert_offset_s": round(first - run["start"], 2) if first is not None else None,
            "attack_types": alerts.loc[mine, "attack_type"].value_counts().to_dict(),
            "modules": alerts.loc[mine, "detection_module"].value_counts().to_dict(),
            "severity": alerts.loc[mine, "severity"].value_counts().to_dict(),
        })
    background = alerts.loc[~attributed]
    report = {"scenarios": results, "background_alerts": int(len(background)),
              "background_alert_types": background["attack_type"].value_counts().to_dict() if len(background) else {}}
    if a.summary and Path(a.summary).exists():
        report["processor"] = json.loads(Path(a.summary).read_text())
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
