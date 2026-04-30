#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

from ci_benchmark_regression import _collect_means


def _collect_ratio_rows(
    means: dict[str, float],
    left_prefix: str,
    right_prefix: str,
    threshold: float,
    category: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark_name in sorted(means):
        if not benchmark_name.startswith(left_prefix):
            continue
        suffix = benchmark_name[len(left_prefix) :]
        baseline_name = f"{right_prefix}{suffix}"
        baseline_mean = means.get(baseline_name)
        ratio = None
        status = "missing-baseline"
        if baseline_mean is not None and baseline_mean > 0:
            ratio = means[benchmark_name] / baseline_mean
            status = "regression" if ratio > threshold else "ok"
        elif baseline_mean is not None:
            status = "invalid-baseline"

        rows.append(
            {
                "category": category,
                "name": suffix,
                "mapped_mean_ns": means[benchmark_name],
                "baseline_mean_ns": baseline_mean,
                "ratio": ratio,
                "threshold": threshold,
                "status": status,
            }
        )
    return rows


def build_dashboard_data(
    criterion_dir: Path, loader_threshold: float, memory_threshold: float
) -> dict[str, object]:
    means = _collect_means(criterion_dir)
    rows = _collect_ratio_rows(
        means,
        "loader/mapped_gguf/",
        "loader/llama_cpp_baseline/",
        loader_threshold,
        "loader",
    )
    rows.extend(
        _collect_ratio_rows(
            means,
            "memory/loader/mapped_gguf/",
            "memory/loader/llama_cpp_baseline/",
            memory_threshold,
            "memory",
        )
    )
    rows.sort(key=lambda row: (str(row["category"]), str(row["name"])))
    return {
        "criterion_dir": str(criterion_dir),
        "benchmark_count": len(rows),
        "rows": rows,
    }


def _format_float(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def render_html_dashboard(data: dict[str, object]) -> str:
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "unknown"))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{_format_float(row.get('mapped_mean_ns'), 0)}</td>"
            f"<td>{_format_float(row.get('baseline_mean_ns'), 0)}</td>"
            f"<td>{_format_float(row.get('ratio'), 3)}</td>"
            f"<td>{_format_float(row.get('threshold'), 3)}</td>"
            f"<td class='status-{html.escape(status)}'>{html.escape(status)}</td>"
            "</tr>"
        )
    table_body = "\n".join(table_rows) if table_rows else "<tr><td colspan='7'>No benchmark rows found.</td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Dashboard</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .status-ok {{ color: #096b2d; font-weight: 600; }}
    .status-regression, .status-missing-baseline, .status-invalid-baseline {{ color: #a1260d; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Benchmark Dashboard</h1>
  <p>Criterion directory: <code>{html.escape(str(data.get("criterion_dir", "")))}</code></p>
  <p>Total benchmark rows: {html.escape(str(data.get("benchmark_count", 0)))}</p>
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th>Benchmark</th>
        <th>Mapped Mean (ns)</th>
        <th>Baseline Mean (ns)</th>
        <th>Ratio</th>
        <th>Threshold</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard from Criterion output.")
    parser.add_argument("--criterion-dir", default="target/criterion", type=Path)
    parser.add_argument("--output-dir", default="target/benchmark-dashboard", type=Path)
    parser.add_argument("--max-loader-ratio", type=float, default=1.25)
    parser.add_argument("--max-memory-ratio", type=float, default=1.40)
    args = parser.parse_args()

    data = build_dashboard_data(
        args.criterion_dir,
        loader_threshold=args.max_loader_ratio,
        memory_threshold=args.max_memory_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    html_path = args.output_dir / "index.html"
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    html_path.write_text(render_html_dashboard(data), encoding="utf-8")
    print(f"Wrote dashboard files: {summary_path} and {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
