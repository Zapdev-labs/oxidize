#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_means(criterion_dir: Path) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for benchmark_path in criterion_dir.glob("**/new/benchmark.json"):
        benchmark = _read_json(benchmark_path)
        full_id = benchmark.get("full_id")
        if not isinstance(full_id, str) or not full_id:
            continue

        estimates_path = benchmark_path.with_name("estimates.json")
        if not estimates_path.exists():
            continue

        estimates = _read_json(estimates_path)
        mean = estimates.get("mean", {}).get("point_estimate")
        if isinstance(mean, (int, float)):
            means[full_id] = float(mean)
    return means


def _check_ratio(
    means: Dict[str, float],
    left_prefix: str,
    right_prefix: str,
    max_ratio: float,
) -> list[str]:
    errors: list[str] = []
    for name, left_value in means.items():
        if not name.startswith(left_prefix):
            continue
        suffix = name[len(left_prefix) :]
        right_name = f"{right_prefix}{suffix}"
        right_value = means.get(right_name)
        if right_value is None:
            errors.append(f"missing baseline benchmark '{right_name}'")
            continue
        if right_value <= 0:
            errors.append(f"invalid baseline mean for '{right_name}'")
            continue

        ratio = left_value / right_value
        if ratio > max_ratio:
            errors.append(
                f"regression in '{suffix}': ratio {ratio:.3f} exceeds {max_ratio:.3f}"
            )
    return errors


def run_regression_check(
    criterion_dir: Path,
    loader_ratio: float,
    memory_ratio: float,
) -> list[str]:
    means = _collect_means(criterion_dir)
    if not means:
        return [f"no criterion benchmark estimates found in '{criterion_dir}'"]

    errors: list[str] = []
    errors.extend(
        _check_ratio(
            means,
            "loader/mapped_gguf/",
            "loader/llama_cpp_baseline/",
            loader_ratio,
        )
    )
    errors.extend(
        _check_ratio(
            means,
            "memory/loader/mapped_gguf/",
            "memory/loader/llama_cpp_baseline/",
            memory_ratio,
        )
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail CI when benchmark ratios exceed configured thresholds."
    )
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        type=Path,
        help="Path to Criterion output directory.",
    )
    parser.add_argument(
        "--max-loader-ratio",
        type=float,
        default=1.25,
        help="Maximum allowed mapped/baseline mean ratio for loader benchmarks.",
    )
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=1.40,
        help="Maximum allowed mapped/baseline mean ratio for memory benchmarks.",
    )
    args = parser.parse_args()

    errors = run_regression_check(
        args.criterion_dir,
        loader_ratio=args.max_loader_ratio,
        memory_ratio=args.max_memory_ratio,
    )
    if errors:
        print("Benchmark regression check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Benchmark regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
