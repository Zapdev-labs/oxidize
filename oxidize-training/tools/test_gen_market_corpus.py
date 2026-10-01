#!/usr/bin/env python3
"""Check gen_market_corpus PICKS lines against synthetic OHLC data.

Ranking alone used to pick the two best and two worst names regardless of the
return threshold, so an all-down day labelled the least-negative names BUY while
their own single-name tickets said SELL. Run with: python3 tools/test_gen_market_corpus.py
"""
from __future__ import annotations

import datetime
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "gen_market_corpus.py"
SYMS = ["spy", "qqq", "iwm", "tlt", "gld", "smh", "xlf", "aapl", "msft", "nvda"]


def write_csvs(data: Path, drift: float) -> None:
    """One CSV per symbol, drifted so most days lean one way."""
    rng = random.Random(11)
    data.mkdir(parents=True, exist_ok=True)
    start = datetime.date(2023, 1, 2)
    for sym in SYMS:
        px = 100.0
        rows = ["Date,Open,High,Low,Close,Volume"]
        for i in range(160):
            day = start + datetime.timedelta(days=i)
            px *= 1.0 + drift + rng.gauss(0.0, 0.012)
            rows.append(
                f"{day},{px * 0.998:.4f},{px * 1.01:.4f},{px * 0.99:.4f},{px:.4f},1000"
            )
        (data / f"{sym}.csv").write_text("\n".join(rows) + "\n")


def run(data: Path, out: Path) -> str:
    proc = subprocess.run(
        [
            sys.executable, str(GEN),
            "--data-dir", str(data),
            "--news-dir", str(data / "news"),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"generator failed: {proc.stderr}")
    return out.read_text(encoding="utf-8")


def check_picks(corpus: str) -> int:
    """Every PICKS line names five distinct symbols in a fixed role order."""
    n = 0
    for line in corpus.splitlines():
        if not line.startswith("PICKS "):
            continue
        n += 1
        parts = line.split()
        syms = parts[2:12:2]
        acts = [a.rstrip(".") for a in parts[3:12:2]]
        assert acts == ["BUY", "BUY", "HOLD", "SELL", "SELL"], f"bad roles: {line}"
        assert len(set(syms)) == 5, f"symbol used twice: {line}"
    return n


def check_agreement(corpus: str) -> int:
    """A PICKS role must match the same symbol's ticket action for that day.

    HOLD tickets are downsampled, so only compare roles whose ticket survived.
    """
    tickets: dict[tuple[str, str], str] = {}
    for line in corpus.splitlines():
        if line.startswith("PICKS ") or "ACTION=" not in line:
            continue
        parts = line.split()
        tickets[(parts[0], parts[1])] = line.rsplit("ACTION=", 1)[1]
    compared = 0
    for line in corpus.splitlines():
        if not line.startswith("PICKS "):
            continue
        parts = line.split()
        day = parts[1]
        for sym, act in zip(parts[2:12:2], (a.rstrip(".") for a in parts[3:12:2])):
            want = tickets.get((sym, day))
            if want is None:
                continue
            compared += 1
            assert want == act, f"{sym} {day}: PICKS says {act}, ticket says {want}"
    return compared


def main() -> int:
    # A one-sided tape is what rank-only selection got wrong: it labelled the
    # least-negative names BUY on a down day.
    total = 0
    for drift in (-0.004, 0.0, 0.004):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            write_csvs(data, drift)
            corpus = run(data, Path(tmp) / "corpus.txt")
            npicks = check_picks(corpus)
            compared = check_agreement(corpus)
            total += npicks
            print(
                f"drift {drift:+.3f}: {npicks} PICKS lines, {compared} roles agree",
                file=sys.stderr,
            )
    assert total, "no PICKS lines produced on any tape"
    print("gen_market_corpus: ok", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
