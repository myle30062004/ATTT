"""Resumable dataset downloader for CSE-CIC-IDS2018 (public S3 bucket) and CTU-13 labels.

Each file is downloaded in chunks with HTTP Range requests so an interrupted
transfer resumes where it stopped. Integrity is checked against the size the
server reports (S3 multipart ETags are not plain MD5s, so size is the reliable
check); a SHA-256 is recorded in a manifest for later verification.

Usage:
    python scripts/download_dataset.py cicids2018
    python scripts/download_dataset.py ctu13 --scenarios 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CICIDS_BUCKET = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
CICIDS_PREFIX = "Processed Traffic Data for ML Algorithms/"
CTU_BASE = "https://mcfp.felk.cvut.cz/publicDatasets"
CTU_SCENARIOS = {
    42: "capture20110810.binetflow", 43: "capture20110811.binetflow",
    44: "capture20110812.binetflow", 45: "capture20110815.binetflow",
    46: "capture20110815-2.binetflow", 47: "capture20110816.binetflow",
    48: "capture20110816-2.binetflow", 49: "capture20110816-3.binetflow",
    50: "capture20110817.binetflow", 51: "capture20110818.binetflow",
    52: "capture20110818-2.binetflow", 53: "capture20110819.binetflow",
    54: "capture20110815-3.binetflow",
}
CHUNK = 1 << 20


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def download(url: str, dest: Path, expected_size: int | None = None, retries: int = 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = expected_size if expected_size is not None else remote_size(url)
    if dest.exists() and dest.stat().st_size == total:
        print(f"[skip] {dest.name} already complete ({total:,} bytes)")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        if have == total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab") as fh:
                last = time.time()
                while True:
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    fh.write(buf)
                    have += len(buf)
                    if time.time() - last > 15:
                        print(f"  {dest.name}: {have / total:6.1%} ({have / 1e6:,.0f} MB)", flush=True)
                        last = time.time()
        except Exception as exc:  # network errors: resume on next attempt
            print(f"  attempt {attempt} interrupted: {exc!r}", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if have == total:
            break
    if not part.exists() or part.stat().st_size != total:
        raise RuntimeError(f"download incomplete for {url}")
    part.replace(dest)
    print(f"[done] {dest.name} ({total:,} bytes)")
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK * 8), b""):
            h.update(buf)
    return h.hexdigest()


def update_manifest(directory: Path, entries: dict) -> None:
    manifest = directory / "MANIFEST.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data.update(entries)
    manifest.write_text(json.dumps(data, indent=2))


def list_cicids_files() -> list[tuple[str, int]]:
    query = urllib.parse.urlencode({"list-type": "2", "prefix": CICIDS_PREFIX})
    with urllib.request.urlopen(f"{CICIDS_BUCKET}/?{query}", timeout=60) as resp:
        tree = ET.fromstring(resp.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    files = []
    for item in tree.findall("s3:Contents", ns):
        key = item.find("s3:Key", ns).text
        size = int(item.find("s3:Size", ns).text)
        if key.endswith(".csv") and size > 0:
            files.append((key, size))
    return sorted(files, key=lambda kv: kv[1])


def fetch_cicids(out_dir: Path) -> None:
    files = list_cicids_files()
    print(f"CICIDS2018: {len(files)} CSV files, {sum(s for _, s in files) / 1e9:.2f} GB total")
    entries = {}
    for key, size in files:
        url = f"{CICIDS_BUCKET}/{urllib.parse.quote(key)}"
        path = download(url, out_dir / Path(key).name, expected_size=size)
        entries[path.name] = {"source": url, "size": size}
    for name, meta in entries.items():
        meta["sha256"] = sha256(out_dir / name)
    update_manifest(out_dir, entries)


def fetch_ctu(out_dir: Path, scenarios: list[int]) -> None:
    entries = {}
    for sid in scenarios:
        fname = CTU_SCENARIOS[sid]
        url = f"{CTU_BASE}/CTU-Malware-Capture-Botnet-{sid}/detailed-bidirectional-flow-labels/{fname}"
        path = download(url, out_dir / f"scenario_{sid}" / fname)
        entries[f"scenario_{sid}/{fname}"] = {"source": url, "size": path.stat().st_size, "sha256": sha256(path)}
    update_manifest(out_dir, entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["cicids2018", "ctu13"])
    parser.add_argument("--scenarios", type=int, nargs="+", default=[42])
    args = parser.parse_args(argv)
    if args.dataset == "cicids2018":
        fetch_cicids(ROOT / "data" / "raw" / "cicids2018")
    else:
        fetch_ctu(ROOT / "data" / "raw" / "ctu13", args.scenarios)
    return 0


if __name__ == "__main__":
    sys.exit(main())
