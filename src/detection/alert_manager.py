"""Alert output: console, JSON lines, CSV history and optional SQLite.

Repeated alerts with the same (src, dst, type, module) inside
``suppress_window_s`` are aggregated into a counter instead of being written
again, so a flood (e.g. DoS) does not produce millions of identical lines.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from pathlib import Path

from .decision_engine import Alert

CSV_FIELDS = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "detection_module",
              "attack_type", "confidence", "severity", "anomaly_score", "reason", "relevant_features"]
SEVERITY_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class AlertManager:
    def __init__(self, out_dir: str | Path, console: bool = True, use_sqlite: bool = False,
                 suppress_window_s: float = 30.0, min_console_severity: str = "LOW") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.out_dir / "alerts.jsonl"
        self.csv_path = self.out_dir / "alerts.csv"
        self.console = console
        self.min_console = SEVERITY_ORDER[min_console_severity]
        self.suppress_window = suppress_window_s
        self._last_seen: dict[tuple, float] = {}
        self.suppressed: dict[tuple, int] = {}
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._db = None
        if use_sqlite:
            self._db = sqlite3.connect(self.out_dir / "alerts.db", check_same_thread=False)
            self._db.execute(f"CREATE TABLE IF NOT EXISTS alerts ({', '.join(f'{c} TEXT' for c in CSV_FIELDS)})")
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    def emit(self, alert: Alert, now: float | None = None) -> bool:
        """Write the alert unless it duplicates one emitted within the suppression window."""
        now = time.time() if now is None else now
        key = (alert.src_ip, alert.dst_ip, alert.attack_type, alert.detection_module)
        with self._lock:
            last = self._last_seen.get(key)
            if last is not None and now - last < self.suppress_window:
                self.suppressed[key] = self.suppressed.get(key, 0) + 1
                return False
            self._last_seen[key] = now
            repeats = self.suppressed.pop(key, 0)
            record = alert.to_dict()
            if repeats:
                record["reason"] += f" ({repeats} similar alerts suppressed since last report)"
            self.counts[alert.attack_type] = self.counts.get(alert.attack_type, 0) + 1
            with open(self.json_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            row = {k: record.get(k) for k in CSV_FIELDS}
            row["relevant_features"] = json.dumps(record.get("relevant_features", {}), default=str)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
            if self._db is not None:
                self._db.execute(f"INSERT INTO alerts VALUES ({', '.join('?' * len(CSV_FIELDS))})",
                                 [str(row[k]) for k in CSV_FIELDS])
                self._db.commit()
            if self.console and SEVERITY_ORDER[alert.severity] >= self.min_console:
                print(f"[{alert.severity:8s}] {alert.timestamp} {alert.detection_module:17s} "
                      f"{alert.attack_type:12s} {alert.src_ip}:{alert.src_port} -> {alert.dst_ip}:{alert.dst_port} "
                      f"score={alert.confidence:.2f} | {alert.reason}", flush=True)
            return True

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
