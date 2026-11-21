"""Realtime NIDS entry point.

Live capture (Kali Linux, needs root or CAP_NET_RAW):
    sudo python3 scripts/run_realtime.py --interface eth0
    sudo python3 scripts/run_realtime.py --interface eth0 --duration 600 --sqlite

Replay a PCAP through the identical realtime path (no root needed):
    python scripts/run_realtime.py --pcap capture.pcap --speed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime.monitor import run_monitor  # noqa: E402
from src.realtime.packet_capture import LiveCapture  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", help="network interface to sniff (default: config realtime.interface)")
    p.add_argument("--pcap", help="replay this pcap instead of live capture")
    p.add_argument("--speed", type=float, default=0.0, help="replay speed factor (0 = as fast as possible)")
    p.add_argument("--duration", type=float, help="stop after N seconds")
    p.add_argument("--max-packets", type=int, help="stop replay after N packets")
    p.add_argument("--bundle", help="deployment bundle directory (default models/deployment)")
    p.add_argument("--alerts", help="alert output directory (default config realtime.alert_dir)")
    p.add_argument("--sqlite", action="store_true", help="also store alerts in SQLite")
    p.add_argument("--quiet", action="store_true", help="no console alerts")
    p.add_argument("--list-interfaces", action="store_true")
    a = p.parse_args()
    if a.list_interfaces:
        print("\n".join(LiveCapture.list_interfaces()))
        return
    run_monitor(interface=a.interface, pcap=a.pcap, speed=a.speed, duration_s=a.duration, bundle_dir=a.bundle,
                alert_dir=a.alerts, sqlite=a.sqlite, max_packets=a.max_packets, quiet=a.quiet)


if __name__ == "__main__":
    main()
