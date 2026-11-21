"""Fetch single PCAP members from the CSE-CIC-IDS2018 raw-traffic ZIPs without downloading the whole archive.

Each day's raw capture is one ZIP of ~40+ GB on the public S3 bucket. ZIP
archives keep their directory at the end, so with HTTP Range requests we can
list the members and stream-decompress just one of them.

    python scripts/fetch_cic_pcap_member.py --day Friday-02-03-2018 --list
    python scripts/fetch_cic_pcap_member.py --day Friday-02-03-2018 --member pcap/capEC2AMAZ-... --out data/raw/cicids2018_pcap/

Used for the feature-parity check (scripts/parity_check.py).
"""
from __future__ import annotations

import argparse
import io
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BUCKET = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP Range requests (with a small read-ahead cache)."""

    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache_start, self._cache = -1, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs = self._cache_start
        if cs <= self.pos and self.pos + n <= cs + len(self._cache):
            out = self._cache[self.pos - cs:self.pos - cs + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            self._cache_start = self.pos
            self._cache = self._fetch(self.pos, min(self.size, self.pos + self.block))
            out = self._cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_day(day: str) -> zipfile.ZipFile:
    key = f"Original Network Traffic and Log data/{day}/pcap.zip"
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(f"{BUCKET}/{urllib.parse.quote(key)}"), 1 << 20))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day", required=True)
    p.add_argument("--list", action="store_true")
    p.add_argument("--member")
    p.add_argument("--out", default="data/raw/cicids2018_pcap")
    a = p.parse_args()
    zf = open_day(a.day)
    if a.list:
        for info in sorted(zf.infolist(), key=lambda i: i.compress_size):
            print(f"{info.compress_size:>14,} {info.file_size:>14,} {info.filename}")
        return
    out = Path(a.out) / a.day
    out.mkdir(parents=True, exist_ok=True)
    target = out / Path(a.member).name
    with zf.open(a.member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
