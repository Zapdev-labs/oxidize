#!/usr/bin/env python3
"""Cache one Wikipedia current-events digest per calendar day.

Usage:
  fetch_news.py --csv data/spy.csv --out-dir data/news
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

UA = {"User-Agent": "oxidize-training/0.1 (https://github.com/dih/oxidize; news corpus)"}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

EVENTS = {
    "2021-01-06": "US Capitol riot during Electoral College count.",
    "2021-01-20": "Biden inaugurated. Markets watch fiscal and Fed path.",
    "2022-02-24": "Russia invades Ukraine. Energy and defense reprice.",
    "2022-03-16": "Fed starts hiking cycle. Risk assets sold.",
    "2022-11-08": "US midterm elections. Gridlock bid for duration.",
    "2023-03-10": "SVB fails. Regional banks and liquidity stress.",
    "2023-10-07": "Hamas attacks Israel. Oil and defense jump.",
    "2024-07-13": "Trump survives assassination attempt in Pennsylvania.",
    "2024-11-05": "US presidential election. Trump wins. Tariff and tax regime repriced.",
    "2025-01-20": "US presidential inauguration. Policy path for tariffs and spending.",
    "2025-04-02": "Tariff headlines hit risk assets. China and Mexico supply chains in focus.",
}


ATTEMPTS = 6


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    delay = 1.0
    for _ in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 12.0)
    raise urllib.error.URLError(f"throttled after {ATTEMPTS} attempts: {url}")


def clean_wiki(s: str) -> str:
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[https?:[^\s\]]+\s+([^\]]*)\]", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\{\{[^}]+\}\}", "", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" \t-*")
    return s


def parse_wikitext(wt: str) -> list[str]:
    lines = []
    for raw in wt.splitlines():
        if not raw.lstrip().startswith("*"):
            continue
        t = clean_wiki(raw)
        if len(t) < 60:
            continue
        if t.lower().startswith("armed conflicts") or t.endswith(":"):
            continue
        lines.append(t)
        if len(lines) >= 6:
            break
    return lines


def fetch_day(day: str) -> str:
    dt = datetime.strptime(day, "%Y-%m-%d")
    title = f"Portal:Current_events/{dt.year}_{MONTHS[dt.month - 1]}_{dt.day}"
    url = (
        "https://en.wikipedia.org/w/api.php?action=parse&page="
        + urllib.parse.quote(title)
        + "&prop=wikitext&format=json"
    )
    try:
        data = json.loads(get(url).decode("utf-8", "replace"))
    except Exception as e:
        return f"NEWS unavailable ({e})."
    if "error" in data:
        return EVENTS.get(day, "Quiet political tape. No major current-events page.")
    wt = data.get("parse", {}).get("wikitext", {}).get("*", "")
    items = parse_wikitext(wt)
    extra = EVENTS.get(day)
    if extra:
        items = [extra] + items
    if not items:
        items = [EVENTS.get(day, "No headline tape. Macro is the tape.")]
    text = " ".join(items)
    if len(text) > 420:
        text = text[:417] + "..."
    return text


def dates_from_csv(path: Path) -> list[str]:
    days = []
    with path.open() as f:
        next(f, None)
        for line in f:
            d = line.split(",", 1)[0].strip()
            if len(d) >= 10:
                days.append(d[:10])
    return days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/spy.csv")
    ap.add_argument("--out-dir", default="data/news")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = dates_from_csv(Path(args.csv))
    if not days:
        print("no dates", file=sys.stderr)
        return 1
    n = 0
    failed = 0
    for i, day in enumerate(days):
        dest = out / f"{day}.txt"
        if dest.exists() and dest.stat().st_size > 8:
            continue
        try:
            text = fetch_day(day)
        except Exception as e:
            print(f"fail {day} {e}", file=sys.stderr)
            failed += 1
            time.sleep(max(args.sleep, 1.0))
            continue
        if text.startswith("NEWS unavailable"):
            print(f"skip {day} {text[:80]}", file=sys.stderr)
            failed += 1
            time.sleep(max(args.sleep, 1.0))
            continue
        dest.write_text(text + "\n", encoding="utf-8")
        n += 1
        if n % 25 == 0 or i == 0:
            print(f"{day} {n} fetched / {i + 1}/{len(days)}", file=sys.stderr)
        time.sleep(args.sleep)
    print(f"news cache {out} days={len(days)} new={n} failed={failed}", file=sys.stderr)
    # Silently caching nothing would let corpus generation label every day
    # "Quiet political tape.", so surface an upstream outage as a failure.
    if failed and not n:
        print(f"fetched no news for {failed} uncached days", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
