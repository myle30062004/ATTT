"""Controlled lab traffic for testing the NIDS. FOR YOUR OWN LAB ONLY.

Every generator refuses destinations that are not loopback or private
(RFC 1918 / unique-local) addresses, caps rate and duration, and writes a
ground-truth record (attacker IP, target, start/end time) to ``--truth`` so
the run can be scored with ``evaluate_lab_run.py``.

    python scripts/lab/traffic_generators.py server  --port 8080
    python scripts/lab/traffic_generators.py beacon  --target 127.0.0.1 --port 8080 --interval 10 --jitter 0.5 --count 40
    python scripts/lab/traffic_generators.py portscan --target 127.0.0.1 --ports 1-1024 --rate 300
    python scripts/lab/traffic_generators.py dos     --target 127.0.0.1 --port 8080 --rate 400 --duration 30

On Kali the scan / DoS scenarios can equally use standard tools against a lab VM
(e.g. ``nmap -sS 192.168.56.101``, ``hping3 -S -p 80 --faster -c 20000 192.168.56.101``).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import random
import socket
import socketserver
import threading
import time
from pathlib import Path

MAX_RATE = 2000          # connections per second
MAX_DURATION = 600       # seconds


def assert_lab_target(target: str) -> str:
    ip = ipaddress.ip_address(socket.gethostbyname(target))
    if not (ip.is_loopback or ip.is_private) or ip.is_multicast or ip.is_unspecified:
        raise SystemExit(f"refusing non-lab destination {ip}: only loopback/private addresses are allowed")
    return str(ip)


def write_truth(path: str | None, record: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(p.read_text()) if p.exists() else []
    runs.append(record)
    p.write_text(json.dumps(runs, indent=2))


def local_source_ip(target: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((target, 9))
        return s.getsockname()[0]


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(2)
            data = self.request.recv(4096)
            if data:
                self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        except OSError:
            pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 1024


def run_server(port: int, bind: str = "127.0.0.1") -> None:
    with _Server((bind, port), _Handler) as srv:
        print(f"lab server listening on {bind}:{port} (Ctrl+C to stop)", flush=True)
        srv.serve_forever()


def _connect(target: str, port: int, payload: bytes | None, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((target, port), timeout=timeout) as s:
            if payload:
                s.sendall(payload)
                s.recv(1024)
        return True
    except OSError:
        return False


def beacon(target: str, port: int, interval: float, jitter: float, count: int, truth: str | None) -> None:
    target = assert_lab_target(target)
    start = time.time()
    ok = 0
    for i in range(count):
        ok += _connect(target, port, b"GET /beacon?id=lab HTTP/1.1\r\nHost: c2\r\n\r\n")
        if i < count - 1:
            time.sleep(max(0.1, interval + random.uniform(-jitter, jitter)))
    end = time.time()
    write_truth(truth, {"scenario": "beaconing", "attacker_ip": local_source_ip(target), "target_ip": target,
                        "target_port": port, "start": start, "end": end, "connections": count,
                        "successful": ok, "interval_s": interval, "jitter_s": jitter})
    print(f"beacon done: {ok}/{count} successful connections")


def portscan(target: str, ports: str, rate: float, truth: str | None) -> None:
    target = assert_lab_target(target)
    lo, hi = (int(x) for x in ports.split("-"))
    rate = min(rate, MAX_RATE)
    start, open_ports = time.time(), []
    for port in range(lo, hi + 1):
        t0 = time.perf_counter()
        if _connect(target, port, None, timeout=0.3):
            open_ports.append(port)
        time.sleep(max(0.0, 1.0 / rate - (time.perf_counter() - t0)))
    end = time.time()
    write_truth(truth, {"scenario": "portscan", "attacker_ip": local_source_ip(target), "target_ip": target,
                        "start": start, "end": end, "ports_scanned": hi - lo + 1, "open_ports": open_ports})
    print(f"scan done: {hi - lo + 1} ports, open: {open_ports}")


def dos(target: str, port: int, rate: float, duration: float, workers: int, truth: str | None) -> None:
    target = assert_lab_target(target)
    rate, duration = min(rate, MAX_RATE), min(duration, MAX_DURATION)
    stop = time.time() + duration
    sent = [0] * workers
    request = b"GET / HTTP/1.1\r\nHost: lab\r\n\r\n"

    def worker(idx: int) -> None:
        per_worker = rate / workers
        while time.time() < stop:
            t0 = time.perf_counter()
            _connect(target, port, request, timeout=1.0)
            sent[idx] += 1
            time.sleep(max(0.0, 1.0 / per_worker - (time.perf_counter() - t0)))

    start = time.time()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    write_truth(truth, {"scenario": "dos", "attacker_ip": local_source_ip(target), "target_ip": target,
                        "target_port": port, "start": start, "end": end, "connections": sum(sent),
                        "rate_cap": rate})
    print(f"dos simulation done: {sum(sent)} connections in {end - start:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("server"); s.add_argument("--port", type=int, default=8080); s.add_argument("--bind", default="127.0.0.1")
    b = sub.add_parser("beacon")
    for sp in (b,):
        sp.add_argument("--target", required=True); sp.add_argument("--port", type=int, default=8080)
    b.add_argument("--interval", type=float, default=10); b.add_argument("--jitter", type=float, default=0.5)
    b.add_argument("--count", type=int, default=40); b.add_argument("--truth")
    ps = sub.add_parser("portscan"); ps.add_argument("--target", required=True); ps.add_argument("--ports", default="1-1024")
    ps.add_argument("--rate", type=float, default=300); ps.add_argument("--truth")
    d = sub.add_parser("dos"); d.add_argument("--target", required=True); d.add_argument("--port", type=int, default=8080)
    d.add_argument("--rate", type=float, default=300); d.add_argument("--duration", type=float, default=30)
    d.add_argument("--workers", type=int, default=16); d.add_argument("--truth")
    a = p.parse_args()
    if a.cmd == "server":
        run_server(a.port, a.bind)
    elif a.cmd == "beacon":
        beacon(a.target, a.port, a.interval, a.jitter, a.count, a.truth)
    elif a.cmd == "portscan":
        portscan(a.target, a.ports, a.rate, a.truth)
    else:
        dos(a.target, a.port, a.rate, a.duration, a.workers, a.truth)


if __name__ == "__main__":
    main()
