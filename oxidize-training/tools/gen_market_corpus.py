#!/usr/bin/env python3
"""Build a next-day-return trading corpus with political news.

Each line is either a single-name ticket labeled by the next close, or a
daily PICKS line ranking names the model should buy and sell.

Usage:
  gen_market_corpus.py --data-dir data --news-dir data/news --out data/corpus.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "SMH",
    "XLF",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "JPM",
    "XOM",
    "NFLX",
    "COST",
]


def load_ohlc(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(
                    {
                        "d": row["Date"][:10],
                        "o": float(row["Open"]),
                        "h": float(row["High"]),
                        "l": float(row["Low"]),
                        "c": float(row["Close"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    if len(rows) >= 2 and rows[0]["d"] > rows[-1]["d"]:
        rows.reverse()
    return rows


def rsi_atr(rows: list[dict]) -> None:
    p = 14
    ag = al = atr_s = 0.0
    ema = rows[0]["c"]
    for i, b in enumerate(rows):
        if i == 0:
            b["rsi"] = 50.0
            b["atr"] = b["h"] - b["l"]
            b["ema"] = ema
            b["r5"] = 0.0
            continue
        ch = b["c"] - rows[i - 1]["c"]
        g = ch if ch > 0 else 0.0
        ls = -ch if ch < 0 else 0.0
        tr = max(b["h"] - b["l"], abs(b["h"] - rows[i - 1]["c"]), abs(b["l"] - rows[i - 1]["c"]))
        if i < p:
            ag += g
            al += ls
            atr_s += tr
            b["rsi"] = 50.0
            b["atr"] = atr_s / i
        elif i == p:
            ag = (ag + g) / p
            al = (al + ls) / p
            atr_s = (atr_s + tr) / p
            b["rsi"] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
            b["atr"] = atr_s
        else:
            ag = (ag * (p - 1) + g) / p
            al = (al * (p - 1) + ls) / p
            atr_s = (atr_s * (p - 1) + tr) / p
            b["rsi"] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
            b["atr"] = atr_s
        ema = (2.0 / 21.0) * b["c"] + (19.0 / 21.0) * ema
        b["ema"] = ema
        if i >= 5 and rows[i - 5]["c"] > 0:
            b["r5"] = 100.0 * (b["c"] / rows[i - 5]["c"] - 1.0)
        else:
            b["r5"] = 0.0


def load_news(news_dir: Path, day: str) -> str:
    p = news_dir / f"{day}.txt"
    if not p.exists():
        return "Quiet political tape."
    t = p.read_text(encoding="utf-8", errors="replace").strip().replace("\n", " ")
    if len(t) > 180:
        t = t[:177] + "..."
    return t or "Quiet political tape."


def action_from_ret(ret: float, thr: float) -> str:
    if ret > thr:
        return "BUY"
    if ret < -thr:
        return "SELL"
    return "HOLD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--news-dir", default="data/news")
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--thr", type=float, default=0.006)
    args = ap.parse_args()
    data = Path(args.data_dir)
    news_dir = Path(args.news_dir)
    series: dict[str, dict[str, dict]] = {}
    dates: set[str] = set()
    for sym in UNIVERSE:
        path = data / f"{sym.lower()}.csv"
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        rows = load_ohlc(path)
        if len(rows) < 60:
            continue
        rsi_atr(rows)
        series[sym] = {r["d"]: r for r in rows}
        dates.update(series[sym].keys())
    if "SPY" not in series:
        print("need SPY", file=sys.stderr)
        return 1
    ordered = sorted(dates)
    tickets = {"BUY": [], "SELL": [], "HOLD": []}
    picks_lines = []
    for i in range(40, len(ordered) - 1):
        day = ordered[i]
        nxt = ordered[i + 1]
        news = load_news(news_dir, day)
        spy = series["SPY"].get(day)
        spy_r5 = spy["r5"] if spy else 0.0
        ranked = []
        for sym, byday in series.items():
            a = byday.get(day)
            b = byday.get(nxt)
            if not a or not b or a["c"] <= 0:
                continue
            nxt_ret = b["c"] / a["c"] - 1.0
            act = action_from_ret(nxt_ret, args.thr)
            vs = a["r5"] - spy_r5
            line = (
                f"{sym} {day} close {int(round(a['c']))} RSI {int(round(a['rsi']))} "
                f"ATR {max(1, int(round(a['atr'])))} r5d {a['r5']:+.1f} vsSPY {vs:+.1f}. "
                f"NEWS: {news} ACTION={act}"
            )
            tickets[act].append(line)
            ranked.append((nxt_ret, sym, act, a))
        if len(ranked) >= 4:
            ranked.sort(key=lambda x: x[0], reverse=True)
            buys = [x[1] for x in ranked[:2]]
            sells = [x[1] for x in ranked[-2:]]
            hold = ranked[len(ranked) // 2][1]
            picks_lines.append(
                f"PICKS {day} {buys[0]} BUY {buys[1]} BUY {hold} HOLD {sells[0]} SELL {sells[1]} SELL. NEWS: {news}"
            )
    rng = random.Random(7)
    n_move = max(len(tickets["BUY"]), len(tickets["SELL"]))
    hold = tickets["HOLD"]
    rng.shuffle(hold)
    hold = hold[: max(n_move, 1)]
    lines = tickets["BUY"] + tickets["SELL"] + hold + picks_lines
    rng.shuffle(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(
        f"wrote {len(lines)} docs BUY={len(tickets['BUY'])} SELL={len(tickets['SELL'])} "
        f"HOLD={len(hold)} PICKS={len(picks_lines)} -> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
