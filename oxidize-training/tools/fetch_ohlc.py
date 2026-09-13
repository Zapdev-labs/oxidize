#!/usr/bin/env python3
"""Daily OHLC via Yahoo chart API. Usage: fetch_ohlc.py SPY data/spy.csv [range]"""
import json
import sys
import urllib.request
from datetime import datetime, timezone

def main():
    if len(sys.argv) < 3:
        print("usage: fetch_ohlc.py SYMBOL out.csv [5y]", file=sys.stderr)
        return 1
    sym = sys.argv[1].upper()
    out = sys.argv[2]
    rng = sys.argv[3] if len(sys.argv) > 3 else "5y"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range={rng}&events=div%7Csplit"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    o, h, l, c, v = q["open"], q["high"], q["low"], q["close"], q["volume"]
    rows = []
    for i, t in enumerate(ts):
        if None in (o[i], h[i], l[i], c[i]):
            continue
        d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        vol = 0 if v[i] is None else int(v[i])
        rows.append((d, o[i], h[i], l[i], c[i], vol))
    if len(rows) < 50:
        print("too few bars", file=sys.stderr)
        return 1
    with open(out, "w") as f:
        f.write("Date,Open,High,Low,Close,Volume\n")
        for d, oo, hh, ll, cc, vv in rows:
            f.write(f"{d},{oo:.4f},{hh:.4f},{ll:.4f},{cc:.4f},{vv}\n")
    print(f"wrote {len(rows)} bars to {out}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
