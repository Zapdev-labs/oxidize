#!/usr/bin/env python3
"""Count product source lines for the shrink refactor gate.

Counted extensions: rs, go, py, c, h, cpp, hpp, cc, cxx, ts, sh
Skipped directories: VCS/build/vendor/third-party criterion headers.
Markdown is not counted.

Usage:
  python3 scripts/loc_count.py
  python3 scripts/loc_count.py --json scripts/loc_baseline.json
  python3 scripts/loc_count.py --compare scripts/loc_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "target",
    "vendor",
    ".claire",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    ".uv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "criterion",
}

EXTS = {
    ".rs": "rust",
    ".go": "go",
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".ts": "ts",
    ".sh": "sh",
}


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        p = Path(dirpath)
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in EXTS:
                continue
            yield p / name, EXTS[ext]


def count(root: Path) -> dict:
    by_lang = Counter()
    by_top = Counter()
    files = []
    for path, lang in iter_files(root):
        try:
            with open(path, "rb") as fh:
                n = sum(1 for _ in fh)
        except OSError:
            continue
        rel = str(path.relative_to(root))
        by_lang[lang] += n
        top = rel.split("/", 1)[0]
        by_top[top] += n
        files.append({"path": rel, "lang": lang, "lines": n})
    files.sort(key=lambda x: (-x["lines"], x["path"]))
    return {
        "total": int(sum(by_lang.values())),
        "by_lang": dict(by_lang.most_common()),
        "by_top": dict(by_top.most_common()),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write full count JSON")
    parser.add_argument("--compare", type=Path, help="baseline JSON to diff against")
    args = parser.parse_args()
    data = count(ROOT)
    print(f"total {data['total']}")
    print("by_lang")
    for lang, n in data["by_lang"].items():
        print(f"  {lang:8} {n}")
    print("by_top")
    for top, n in list(data["by_top"].items())[:20]:
        print(f"  {top:28} {n}")
    if args.json:
        slim = {
            "total": data["total"],
            "by_lang": data["by_lang"],
            "by_top": data["by_top"],
        }
        args.json.write_text(json.dumps(slim, indent=2) + "\n")
        print(f"wrote {args.json}")
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        old = int(baseline["total"])
        new = int(data["total"])
        delta = old - new
        print(f"baseline {old}")
        print(f"head     {new}")
        print(f"removed  {delta}")
        if delta < 10000:
            print(f"predicate FAIL: need 10000, have {delta}")
            return 1
        print("predicate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
