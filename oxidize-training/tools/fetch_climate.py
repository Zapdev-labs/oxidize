#!/usr/bin/env python3
"""Download NOAA nClimDiv statewide/divisional monthly climate and CPC indices."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

CLIMDIV_INDEX = "https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/"
FILES = {
    "climdiv-tmpcst.txt": r"climdiv-tmpcst-v1\.0\.0-\d+",
    "climdiv-pcpnst.txt": r"climdiv-pcpnst-v1\.0\.0-\d+",
    "climdiv-pdsist.txt": r"climdiv-pdsist-v1\.0\.0-\d+",
    "climdiv-tmpcdv.txt": r"climdiv-tmpcdv-v1\.0\.0-\d+",
    "climdiv-pcpndv.txt": r"climdiv-pcpndv-v1\.0\.0-\d+",
}
INDICES = {
    "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "ao.ascii.txt": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/"
        "monthly.ao.index.b50.current.ascii"
    ),
    "nao.ascii.txt": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.nao.monthly.b5001.current.ascii"
    ),
    "pna.ascii.txt": (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
        "norm.pna.monthly.b5001.current.ascii"
    ),
}
PDO_URLS = [
    "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    "https://psl.noaa.gov/pdo/data/pdo.timeseries.sst.txt",
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "oxidize-training-weather/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def latest_climdiv_names(html: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for dest, pat in FILES.items():
        matches = re.findall(pat, html)
        if not matches:
            raise RuntimeError(f"no climdiv file matching {pat}")
        found[dest] = sorted(matches)[-1]
    return found


def write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/climate")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    listing = fetch(CLIMDIV_INDEX).decode("utf-8", "replace")
    names = latest_climdiv_names(listing)
    for dest, name in names.items():
        write_bytes(out / dest, fetch(CLIMDIV_INDEX + name))

    for dest, url in INDICES.items():
        write_bytes(out / dest, fetch(url))

    last_err: Exception | None = None
    for url in PDO_URLS:
        try:
            write_bytes(out / "pdo.dat", fetch(url))
            last_err = None
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err:
        raise last_err
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
