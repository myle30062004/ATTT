"""Run the three lab scenarios against a lab server while the realtime monitor captures live traffic.

    python scripts/lab/run_lab.py --interface "Loopback Pseudo-Interface 1" --target 127.0.0.1     # Windows + Npcap
    sudo python3 scripts/lab/run_lab.py --interface lo --target 127.0.0.1                          # Kali
    sudo python3 scripts/lab/run_lab.py --interface eth0 --target 192.168.56.101 --no-server       # lab VM target

Order: warm-up (normal browsing-like requests) -> port scan -> pause -> DoS
simulation -> pause -> beaconing. Each scenario writes ground truth; afterwards
evaluate_lab_run.py scores the alerts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
GEN = str(ROOT / "scripts" / "lab" / "traffic_generators.py")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interface", required=True)
    p.add_argument("--target", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--out", default=str(ROOT / "logs" / "lab"))
    p.add_argument("--no-server", action="store_true", help="target already runs a server")
    p.add_argument("--pause", type=float, default=75.0, help="quiet gap between scenarios (> idle timeout)")
    p.add_argument("--beacon-count", type=int, default=30)
    p.add_argument("--beacon-interval", type=float, default=5.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = out / "truth.json"
    if truth.exists():
        truth.unlink()
    procs = []
    if not a.no_server:
        procs.append(subprocess.Popen([PY, GEN, "server", "--port", str(a.port), "--bind", a.target]))
        time.sleep(1.5)
    expected = 60 + 3 * a.pause + 40 + a.beacon_count * a.beacon_interval + 200
    monitor = subprocess.Popen([PY, str(ROOT / "scripts" / "run_realtime.py"), "--interface", a.interface,
                                "--alerts", str(out / "alerts"), "--duration", str(expected), "--quiet"])
    time.sleep(8)
    try:
        # normal warm-up traffic: a few spaced requests
        run([PY, GEN, "beacon", "--target", a.target, "--port", str(a.port), "--interval", "7", "--jitter", "6",
             "--count", "6"])
        time.sleep(a.pause)
        run([PY, GEN, "portscan", "--target", a.target, "--ports", "1-1000", "--rate", "400", "--truth", str(truth)])
        time.sleep(a.pause)
        run([PY, GEN, "dos", "--target", a.target, "--port", str(a.port), "--rate", "400", "--duration", "30",
             "--truth", str(truth)])
        time.sleep(a.pause)
        run([PY, GEN, "beacon", "--target", a.target, "--port", str(a.port), "--interval", str(a.beacon_interval),
             "--jitter", "0.3", "--count", str(a.beacon_count), "--truth", str(truth)])
        monitor.wait()
    finally:
        for pr in procs:
            pr.terminate()
        if monitor.poll() is None:
            monitor.terminate()
            monitor.wait(30)
    run([PY, str(ROOT / "scripts" / "lab" / "evaluate_lab_run.py"), "--alerts", str(out / "alerts" / "alerts.jsonl"),
         "--truth", str(truth), "--summary", str(out / "alerts" / "run_summary.json"),
         "--out", str(ROOT / "reports" / "experiments" / "lab_results.json")])
    print(json.dumps({"truth": str(truth)}, indent=2))


if __name__ == "__main__":
    main()
