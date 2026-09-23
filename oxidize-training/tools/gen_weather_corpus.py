#!/usr/bin/env python3
"""Build a 50-state DJF winter corpus for oxidize-training from NOAA nClimDiv + CPC indices."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

STATE_BY_CODE = {
    1: "AL",
    2: "AZ",
    3: "AR",
    4: "CA",
    5: "CO",
    6: "CT",
    7: "DE",
    8: "FL",
    9: "GA",
    10: "ID",
    11: "IL",
    12: "IN",
    13: "IA",
    14: "KS",
    15: "KY",
    16: "LA",
    17: "ME",
    18: "MD",
    19: "MA",
    20: "MI",
    21: "MN",
    22: "MS",
    23: "MO",
    24: "MT",
    25: "NE",
    26: "NV",
    27: "NH",
    28: "NJ",
    29: "NM",
    30: "NY",
    31: "NC",
    32: "ND",
    33: "OH",
    34: "OK",
    35: "OR",
    36: "PA",
    37: "RI",
    38: "SC",
    39: "SD",
    40: "TN",
    41: "TX",
    42: "UT",
    43: "VT",
    44: "VA",
    45: "WA",
    46: "WV",
    47: "WI",
    48: "WY",
    49: "HI",
    50: "AK",
}
REGION_BY_CODE = {
    101: "R-NE",
    102: "R-ENC",
    103: "R-C",
    104: "R-SE",
    105: "R-WNC",
    106: "R-S",
    107: "R-SW",
    108: "R-NW",
    109: "R-W",
    110: "R-US",
}
TEMP_MISS = -99.9
PCPN_MISS = -9.99
CLIMO_START, CLIMO_END = 1991, 2020
TRAIN_YEARS = range(1997, 2022)  # DJF 1996/97 through 2020/21
TEST_YEARS = range(2022, 2027)  # DJF 2021/22 through 2025/26
FORECAST_YEAR = 2027  # DJF 2026/27
MONTHS = list(range(1, 13))


ACTION_CODE = {
    "COLD-DRY": "1",
    "COLD-NEAR": "2",
    "COLD-WET": "3",
    "NEAR-DRY": "4",
    "NEAR-NEAR": "5",
    "NEAR-WET": "6",
    "WARM-DRY": "7",
    "WARM-NEAR": "8",
    "WARM-WET": "9",
}
CODE_ACTION = {v: k for k, v in ACTION_CODE.items()}


def fmt_num(x: float) -> str:
    v = max(-9.9, min(9.9, x))
    return f"{v:+.1f}"


def is_missing(val: float, kind: str) -> bool:
    if math.isnan(val):
        return True
    if kind == "t":
        return val <= TEMP_MISS + 0.05
    if kind == "pdsi":
        return val <= -99.0
    return val <= PCPN_MISS + 0.05


def parse_fixed_monthly(path: Path, kind: str, statewide: bool) -> dict[str, dict[int, dict[int, float]]]:
    """loc -> year -> month -> value."""
    out: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if len(line) < 94:
            continue
        try:
            if statewide:
                code = int(line[0:3])
                year = int(line[6:10])
                if code in STATE_BY_CODE:
                    loc = STATE_BY_CODE[code]
                elif code in REGION_BY_CODE:
                    loc = REGION_BY_CODE[code]
                else:
                    continue
            else:
                st = int(line[0:2])
                div = int(line[2:4])
                year = int(line[6:10])
                usps = STATE_BY_CODE.get(st)
                if not usps or div <= 0:
                    continue
                loc = f"{usps}{div:02d}"
            for i, month in enumerate(MONTHS):
                chunk = line[10 + i * 7 : 17 + i * 7].strip()
                val = float(chunk)
                if is_missing(val, kind):
                    continue
                out[loc][year][month] = val
        except ValueError:
            continue
    return out


def parse_oni(path: Path) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        season, year_s, _total, anom = parts[0], parts[1], parts[2], parts[3]
        try:
            out[(int(year_s), season.upper())] = float(anom)
        except ValueError:
            continue
    return out


def parse_monthly_index(path: Path) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            year, month, val = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        if 1 <= month <= 12:
            out[(year, month)] = val
    return out


def parse_pdo(path: Path) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month, raw in enumerate(parts[1:13], start=1):
            try:
                val = float(raw)
            except ValueError:
                continue
            if val > 90 or val < -90:
                continue
            out[(year, month)] = val
    return out


def season_months(name: str) -> list[tuple[int, int]]:
    """Return (year_delta, month) relative to the January year of DJF."""
    if name == "DJF":
        return [(-1, 12), (0, 1), (0, 2)]
    if name == "SON":
        return [(-1, 9), (-1, 10), (-1, 11)]
    if name == "JJA":
        return [(-1, 6), (-1, 7), (-1, 8)]
    raise KeyError(name)


def season_value(series: dict[int, dict[int, float]], jan_year: int, name: str) -> float | None:
    vals = []
    for dy, month in season_months(name):
        y = jan_year + dy
        if month in series.get(y, {}):
            vals.append(series[y][month])
    if len(vals) < 3:
        return None
    return mean(vals)


def climo_season(series: dict[int, dict[int, float]], name: str) -> float | None:
    vals = []
    for year in range(CLIMO_START, CLIMO_END + 1):
        v = season_value(series, year, name)
        if v is not None:
            vals.append(v)
    if len(vals) < 10:
        return None
    return mean(vals)


def index_season(idx: dict[tuple[int, int], float], jan_year: int, name: str) -> float | None:
    vals = []
    for dy, month in season_months(name):
        key = (jan_year + dy, month)
        if key in idx:
            vals.append(idx[key])
    if len(vals) < 2:
        return None
    return mean(vals)


def oni_for_view(oni: dict[tuple[int, str], float], jan_year: int, view: str) -> float | None:
    season = "JJA" if view == "JJA" else "SON"
    return oni.get((jan_year - 1, season))


def phase_of(oni: float) -> str:
    if oni >= 0.5:
        return "EL_NINO"
    if oni <= -0.5:
        return "LA_NINA"
    return "NEUTRAL"


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def tercile(x: float, p33: float, p67: float, low: str, high: str) -> str:
    if x <= p33:
        return low
    if x >= p67:
        return high
    return "NEAR"


def ticket_line(
    loc: str,
    year: int,
    view: str,
    phase: str,
    oni: float,
    pdo: float,
    ao: float,
    nao: float,
    pna: float,
    pers_t: float,
    pers_cls: str,
    pred_t: float,
    pred_p: float,
    label: str,
) -> str:
    pred = "JJA" if view == "JJA" else "SON"
    return (
        f"ST {loc} Y{year} VIEW={view} PHASE={phase} "
        f"ENSO{fmt_num(oni)} PDO{fmt_num(pdo)} AO{fmt_num(ao)} NAO{fmt_num(nao)} PNA{fmt_num(pna)} "
        f"PERS={ACTION_CODE.get(pers_cls, '5')} PERS_T{fmt_num(pers_t)} {pred}_T{fmt_num(pred_t)} {pred}_P{fmt_num(pred_p)} "
        f"ACTION={ACTION_CODE[label]}"
    )


def prompt_line(
    loc: str,
    year: int,
    view: str,
    phase: str,
    oni: float,
    pdo: float,
    ao: float,
    nao: float,
    pna: float,
    pers_t: float,
    pers_cls: str,
    pred_t: float,
    pred_p: float,
) -> str:
    pred = "JJA" if view == "JJA" else "SON"
    return (
        f"ST {loc} Y{year} VIEW={view} PHASE={phase} "
        f"ENSO{fmt_num(oni)} PDO{fmt_num(pdo)} AO{fmt_num(ao)} NAO{fmt_num(nao)} PNA{fmt_num(pna)} "
        f"PERS={ACTION_CODE.get(pers_cls, '5')} PERS_T{fmt_num(pers_t)} {pred}_T{fmt_num(pred_t)} {pred}_P{fmt_num(pred_p)} "
        f"ACTION="
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--climate-dir", default="data/climate")
    ap.add_argument("--out-dir", default="data/weather")
    args = ap.parse_args()
    root = Path(args.climate_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    temp_st = parse_fixed_monthly(root / "climdiv-tmpcst.txt", "t", True)
    pcpn_st = parse_fixed_monthly(root / "climdiv-pcpnst.txt", "p", True)
    pdsi_st = parse_fixed_monthly(root / "climdiv-pdsist.txt", "pdsi", True) if (root / "climdiv-pdsist.txt").exists() else {}
    temp_dv = parse_fixed_monthly(root / "climdiv-tmpcdv.txt", "t", False)
    pcpn_dv = parse_fixed_monthly(root / "climdiv-pcpndv.txt", "p", False)
    oni = parse_oni(root / "oni.ascii.txt")
    ao = parse_monthly_index(root / "ao.ascii.txt")
    nao = parse_monthly_index(root / "nao.ascii.txt")
    pna = parse_monthly_index(root / "pna.ascii.txt")
    pdo = parse_pdo(root / "pdo.dat")

    locs: dict[str, tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], str]] = {}
    for loc, series in temp_st.items():
        kind = "state" if loc in STATE_BY_CODE.values() else "region"
        if loc in pcpn_st:
            locs[loc] = (series, pcpn_st[loc], kind)
    for loc, series in temp_dv.items():
        if loc in pcpn_dv:
            locs[loc] = (series, pcpn_dv[loc], "div")

    climo_t = {loc: climo_season(t, "DJF") for loc, (t, _p, _k) in locs.items()}
    climo_p = {loc: climo_season(p, "DJF") for loc, (_t, p, _k) in locs.items()}
    climo_jja_t = {loc: climo_season(t, "JJA") for loc, (t, _p, _k) in locs.items()}
    climo_jja_p = {loc: climo_season(p, "JJA") for loc, (_t, p, _k) in locs.items()}
    climo_son_t = {loc: climo_season(t, "SON") for loc, (t, _p, _k) in locs.items()}
    climo_son_p = {loc: climo_season(p, "SON") for loc, (_t, p, _k) in locs.items()}

    records: list[dict] = []
    for loc, (t_s, p_s, kind) in locs.items():
        ct, cp = climo_t[loc], climo_p[loc]
        if ct is None or cp is None:
            continue
        for year in range(1996, FORECAST_YEAR + 1):
            djt = season_value(t_s, year, "DJF")
            djp = season_value(p_s, year, "DJF")
            labeled = year < FORECAST_YEAR
            if labeled and (djt is None or djp is None):
                continue
            djt_a = None if djt is None else djt - ct
            djp_a = None if djp is None else djp - cp
            prev_t = season_value(t_s, year - 1, "DJF")
            pers_t = 0.0 if prev_t is None else prev_t - ct
            for view in ("JJA", "SON"):
                pred_t = season_value(t_s, year, view)
                pred_p = season_value(p_s, year, view)
                if pred_t is None or pred_p is None:
                    continue
                c_pred_t = climo_jja_t[loc] if view == "JJA" else climo_son_t[loc]
                c_pred_p = climo_jja_p[loc] if view == "JJA" else climo_son_p[loc]
                if c_pred_t is None or c_pred_p is None:
                    continue
                oni_v = oni_for_view(oni, year, view)
                pdo_v = index_season(pdo, year, view)
                ao_v = index_season(ao, year, view)
                nao_v = index_season(nao, year, view)
                pna_v = index_season(pna, year, view)
                if None in (oni_v, pdo_v, ao_v, nao_v, pna_v):
                    continue
                pdsi_v = season_value(pdsi_st.get(loc, {}), year, view)
                if pdsi_v is None:
                    pdsi_v = 0.0
                records.append(
                    {
                        "loc": loc,
                        "kind": kind,
                        "year": year,
                        "view": view,
                        "oni": oni_v,
                        "pdo": pdo_v,
                        "ao": ao_v,
                        "nao": nao_v,
                        "pna": pna_v,
                        "pdsi": pdsi_v,
                        "pred_t": pred_t - c_pred_t,
                        "pred_p": pred_p - c_pred_p,
                        "pers_t": pers_t,
                        "djf_t": djt_a,
                        "djf_p": djp_a,
                    }
                )

    train_t: dict[str, list[float]] = defaultdict(list)
    train_p: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["year"] in TRAIN_YEARS and r["view"] == "JJA":
            if r["djf_t"] is not None:
                train_t[r["loc"]].append(r["djf_t"])
            if r["djf_p"] is not None:
                train_p[r["loc"]].append(r["djf_p"])
    thresholds: dict[str, dict[str, float]] = {}
    for loc, tvals in train_t.items():
        pvals = train_p.get(loc, [])
        if len(tvals) < 12 or len(pvals) < 12:
            continue
        thresholds[loc] = {
            "t33": percentile(tvals, 1 / 3),
            "t67": percentile(tvals, 2 / 3),
            "p33": percentile(pvals, 1 / 3),
            "p67": percentile(pvals, 2 / 3),
        }

    prev_label: dict[tuple[str, int, str], str] = {}
    labeled_rows: list[dict] = []
    for r in records:
        th = thresholds.get(r["loc"])
        if not th:
            continue
        pers_cls = tercile(r["pers_t"], th["t33"], th["t67"], "COLD", "WARM") + "-NEAR"
        prev = prev_label.get((r["loc"], r["year"] - 1, r["view"]))
        if prev:
            pers_cls = prev
        label = None
        if r["djf_t"] is not None and r["djf_p"] is not None:
            tcls = tercile(r["djf_t"], th["t33"], th["t67"], "COLD", "WARM")
            pcls = tercile(r["djf_p"], th["p33"], th["p67"], "DRY", "WET")
            label = f"{tcls}-{pcls}"
            prev_label[(r["loc"], r["year"], r["view"])] = label
        row = dict(r)
        row["pers_cls"] = pers_cls
        row["label"] = label
        row["phase"] = phase_of(r["oni"])
        labeled_rows.append(row)

    train_lines = []
    test_rows = []
    forecast_rows = []
    for r in labeled_rows:
        if r["year"] in TRAIN_YEARS and r["label"]:
            line = ticket_line(
                    r["loc"],
                    r["year"],
                    r["view"],
                    r["phase"],
                    r["oni"],
                    r["pdo"],
                    r["ao"],
                    r["nao"],
                    r["pna"],
                    r["pers_t"],
                    r["pers_cls"],
                    r["pred_t"],
                    r["pred_p"],
                    r["label"],
                )
            repeats = 8 if r["kind"] == "state" else 3 if r["kind"] == "region" else 1
            train_lines.extend([line] * repeats)
        if r["year"] in TEST_YEARS and r["kind"] == "state" and r["view"] == "JJA" and r["label"]:
            test_rows.append(r)
        if r["year"] == FORECAST_YEAR and r["kind"] == "state" and r["view"] == "JJA":
            forecast_rows.append(r)

    rng = random.Random(42)
    rng.shuffle(train_lines)
    corpus = out / "corpus.txt"
    corpus.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    def dump_prompt_rows(rows: list[dict], path: Path, include_label: bool) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda x: (x["year"], x["loc"])):
                rec = {
                    "loc": r["loc"],
                    "year": r["year"],
                    "view": r["view"],
                    "phase": r["phase"],
                    "pers_cls": r["pers_cls"],
                    "oni": r["oni"],
                    "pdo": r["pdo"],
                    "ao": r["ao"],
                    "nao": r["nao"],
                    "pna": r["pna"],
                    "pdsi": r.get("pdsi", 0.0),
                    "pers_t": r["pers_t"],
                    "pred_t": r["pred_t"],
                    "pred_p": r["pred_p"],
                    "prompt": prompt_line(
                        r["loc"],
                        r["year"],
                        r["view"],
                        r["phase"],
                        r["oni"],
                        r["pdo"],
                        r["ao"],
                        r["nao"],
                        r["pna"],
                        r["pers_t"],
                        r["pers_cls"],
                        r["pred_t"],
                        r["pred_p"],
                    ),
                }
                if include_label:
                    rec["label"] = r["label"]
                    rec["djf_t"] = r["djf_t"]
                    rec["djf_p"] = r["djf_p"]
                f.write(json.dumps(rec) + "\n")

    dump_prompt_rows(test_rows, out / "test.jsonl", True)
    dump_prompt_rows(forecast_rows, out / "forecast_2027.jsonl", False)
    analog_train = [
        r
        for r in labeled_rows
        if r["kind"] == "state" and r["view"] == "JJA" and r["year"] in TRAIN_YEARS and r["label"]
    ]
    dump_prompt_rows(analog_train, out / "analog_train.jsonl", True)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")

    states_train = {r["loc"] for r in labeled_rows if r["kind"] == "state" and r["year"] in TRAIN_YEARS}
    print(
        f"wrote {len(train_lines)} train tickets, {len(test_rows)} test prompts, "
        f"{len(forecast_rows)} forecast prompts, states={len(states_train)} -> {out}",
        file=sys.stderr,
    )
    if len(states_train) != 50:
        print(f"warning: expected 50 states, got {sorted(states_train)}", file=sys.stderr)
    missing_fc = sorted(set(STATE_BY_CODE.values()) - {r["loc"] for r in forecast_rows})
    if missing_fc:
        print(f"warning: forecast missing {missing_fc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
