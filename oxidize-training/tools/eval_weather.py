#!/usr/bin/env python3
"""Score winter outlooks: per-state ENSO analog on NOAA nClimDiv, plus optional oxidize-training GPT."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CODE_ACTION = {
    "1": "COLD-DRY",
    "2": "COLD-NEAR",
    "3": "COLD-WET",
    "4": "NEAR-DRY",
    "5": "NEAR-NEAR",
    "6": "NEAR-WET",
    "7": "WARM-DRY",
    "8": "WARM-NEAR",
    "9": "WARM-WET",
}


def parse_action(text: str) -> str | None:
    for ch in text.strip():
        if ch in CODE_ACTION:
            return CODE_ACTION[ch]
    up = text.upper().replace(" ", "")
    m = re.search(r"(COLD|NEAR|WARM)-(DRY|NEAR|WET)", up)
    return m.group(0) if m else None


def tercile(x: float, p33: float, p67: float, low: str, high: str) -> str:
    if x <= p33:
        return low
    if x >= p67:
        return high
    return "NEAR"


def analog_label(query: dict, pool: list[dict], thresholds: dict[str, dict], k: int = 5) -> tuple[str, float, float]:
    loc = query["loc"]
    th = thresholds[loc]
    local = [r for r in pool if r["loc"] == loc]
    if len(local) < 5:
        local = pool
    oni_q = float(query["oni"])
    same = [r for r in local if oni_q * float(r["oni"]) > 0 and abs(float(r["oni"])) >= 0.5]
    cand = same if abs(oni_q) >= 1.0 and len(same) >= 4 else local
    sigma = 0.45 if cand is same else 0.7
    wsum = 0.0
    dt = 0.0
    dp = 0.0
    for r in cand:
        d_oni = (oni_q - float(r["oni"])) / sigma
        d_pdo = (float(query["pdo"]) - float(r["pdo"])) / 1.2
        d_pdsi = (float(query.get("pdsi", 0.0)) - float(r.get("pdsi", 0.0))) / 2.0
        w = math.exp(-0.5 * (d_oni * d_oni + 0.25 * d_pdo * d_pdo + 0.35 * d_pdsi * d_pdsi))
        wsum += w
        dt += w * float(r["djf_t"])
        dp += w * float(r["djf_p"])
    dt = dt / wsum
    dp = dp / wsum
    label = f"{tercile(dt, th['t33'], th['t67'], 'COLD', 'WARM')}-{tercile(dp, th['p33'], th['p67'], 'DRY', 'WET')}"
    return label, dt, dp


def run_complete(bin_path: str, ckpt: str, vocab: str, prompts: list[str], tokens: int, temp: float) -> list[str] | None:
    ckpt_p = Path(ckpt)
    vocab_p = Path(vocab)
    if not ckpt_p.exists() or not vocab_p.exists():
        return None
    proc = subprocess.run(
        [
            bin_path,
            "complete",
            "--ckpt",
            ckpt,
            "--vocab",
            vocab,
            "--tokens",
            str(tokens),
            "--temp",
            str(temp),
            "--seed",
            "1",
        ],
        input="\n".join(prompts) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    lines = [ln.rstrip("\n") for ln in proc.stdout.splitlines()]
    if len(lines) < len(prompts):
        return None
    return lines[: len(prompts)]


def summarize(rows: list[dict], pred_key: str = "pred") -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": 0.0, "temp_accuracy": 0.0, "precip_accuracy": 0.0, "persistence_accuracy": 0.0}
    acc = sum(1 for r in rows if r[pred_key] == r["label"]) / n
    temp_acc = sum(1 for r in rows if r[pred_key].split("-")[0] == r["label"].split("-")[0]) / n
    pcpn_acc = sum(1 for r in rows if r[pred_key].split("-")[1] == r["label"].split("-")[1]) / n
    persist = sum(1 for r in rows if r["persist"] == r["label"]) / n
    hss_t = (temp_acc - 1 / 3) / (1 - 1 / 3)
    hss_p = (pcpn_acc - 1 / 3) / (1 - 1 / 3)
    return {
        "n": n,
        "accuracy": acc,
        "temp_accuracy": temp_acc,
        "precip_accuracy": pcpn_acc,
        "heidke_temp": hss_t,
        "heidke_precip": hss_p,
        "persistence_accuracy": persist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./bin/oxidize-training")
    ap.add_argument("--ckpt", default="out/weather.bin")
    ap.add_argument("--vocab", default="data/weather/vocab.bin")
    ap.add_argument("--test", default="data/weather/test.jsonl")
    ap.add_argument("--forecast", default="data/weather/forecast_2027.jsonl")
    ap.add_argument("--train", default="data/weather/analog_train.jsonl")
    ap.add_argument("--thresholds", default="data/weather/thresholds.json")
    ap.add_argument("--out", default="out/weather")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    test = [json.loads(ln) for ln in Path(args.test).read_text().splitlines() if ln.strip()]
    fc = [json.loads(ln) for ln in Path(args.forecast).read_text().splitlines() if ln.strip()]
    train = [json.loads(ln) for ln in Path(args.train).read_text().splitlines() if ln.strip()]
    thresholds = json.loads(Path(args.thresholds).read_text())

    analog_rows = []
    for r in test:
        pred, dt, dp = analog_label(r, train, thresholds, args.k)
        analog_rows.append(
            {
                "loc": r["loc"],
                "year": r["year"],
                "label": r["label"],
                "pred": pred,
                "persist": r.get("pers_cls", "NEAR-NEAR"),
                "djf_t_hat": dt,
                "djf_p_hat": dp,
            }
        )
    analog_metrics = summarize(analog_rows)

    gpt_metrics = None
    completions = run_complete(args.bin, args.ckpt, args.vocab, [r["prompt"] for r in test], args.tokens, args.temp)
    gpt_rows = []
    if completions:
        for r, comp in zip(test, completions):
            pred = parse_action(comp) or "NEAR-NEAR"
            gpt_rows.append(
                {
                    "loc": r["loc"],
                    "year": r["year"],
                    "label": r["label"],
                    "pred": pred,
                    "persist": r.get("pers_cls", "NEAR-NEAR"),
                    "completion": comp,
                }
            )
        gpt_metrics = summarize(gpt_rows)
        with (out / "gpt_predictions.jsonl").open("w", encoding="utf-8") as f:
            for r in gpt_rows:
                f.write(json.dumps(r) + "\n")

    (out / "test_metrics.json").write_text(
        json.dumps({"analog": analog_metrics, "gpt": gpt_metrics}, indent=2),
        encoding="utf-8",
    )
    with (out / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in analog_rows:
            f.write(json.dumps(r) + "\n")

    outlook = []
    for r in fc:
        pred, dt, dp = analog_label(r, train, thresholds, args.k)
        tcls, pcls = pred.split("-")
        outlook.append(
            {
                "state": r["loc"],
                "winter": "2026-27",
                "enso_phase": r["phase"],
                "oni_jja": r["oni"],
                "outlook": pred,
                "temperature": tcls,
                "precipitation": pcls,
                "djf_t_anomaly_f": round(dt, 2),
                "djf_p_anomaly_in": round(dp, 2),
            }
        )
    (out / "outlook_2026-27.json").write_text(json.dumps(outlook, indent=2), encoding="utf-8")
    with (out / "outlook_2026-27.tsv").open("w", encoding="utf-8") as f:
        f.write("state\twinter\tenso\toni_jja\ttemperature\tprecipitation\toutlook\tdjf_t_anom_F\tdjf_p_anom_in\n")
        for r in outlook:
            f.write(
                f"{r['state']}\t{r['winter']}\t{r['enso_phase']}\t{r['oni_jja']:.2f}\t"
                f"{r['temperature']}\t{r['precipitation']}\t{r['outlook']}\t"
                f"{r['djf_t_anomaly_f']}\t{r['djf_p_anomaly_in']}\n"
            )

    counts = Counter(r["outlook"] for r in outlook)
    print(
        json.dumps(
            {
                "analog_test": analog_metrics,
                "gpt_test": gpt_metrics,
                "forecast_counts": dict(counts),
                "forecast_n": len(outlook),
            },
            indent=2,
        )
    )
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
