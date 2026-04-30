import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_benchmark_regression import run_regression_check


def _write_benchmark(root: Path, directory_name: str, full_id: str, mean: float) -> None:
    bench_dir = root / directory_name / "new"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "benchmark.json").write_text(
        json.dumps({"directory_name": directory_name, "full_id": full_id}),
        encoding="utf-8",
    )
    (bench_dir / "estimates.json").write_text(
        json.dumps({"mean": {"point_estimate": mean}}),
        encoding="utf-8",
    )


class RegressionCheckTests(unittest.TestCase):
    def test_passes_when_ratios_stay_within_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_benchmark(
                root,
                "loader_mapped",
                "loader/mapped_gguf/fixture/valid-v3.gguf",
                120.0,
            )
            _write_benchmark(
                root,
                "loader_baseline",
                "loader/llama_cpp_baseline/fixture/valid-v3.gguf",
                100.0,
            )
            _write_benchmark(
                root,
                "memory_mapped",
                "memory/loader/mapped_gguf/fixture/valid-v3.gguf",
                130.0,
            )
            _write_benchmark(
                root,
                "memory_baseline",
                "memory/loader/llama_cpp_baseline/fixture/valid-v3.gguf",
                100.0,
            )

            errors = run_regression_check(root, loader_ratio=1.25, memory_ratio=1.4)
            self.assertEqual(errors, [])

    def test_reports_regression_when_loader_ratio_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_benchmark(
                root,
                "loader_mapped",
                "loader/mapped_gguf/fixture/valid-v3.gguf",
                151.0,
            )
            _write_benchmark(
                root,
                "loader_baseline",
                "loader/llama_cpp_baseline/fixture/valid-v3.gguf",
                100.0,
            )

            errors = run_regression_check(root, loader_ratio=1.25, memory_ratio=1.4)
            self.assertEqual(len(errors), 1)
            self.assertIn("regression in 'fixture/valid-v3.gguf'", errors[0])

    def test_reports_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_benchmark(
                root,
                "loader_mapped",
                "loader/mapped_gguf/fixture/valid-v3.gguf",
                101.0,
            )

            errors = run_regression_check(root, loader_ratio=1.25, memory_ratio=1.4)
            self.assertEqual(len(errors), 1)
            self.assertIn("missing baseline benchmark", errors[0])


if __name__ == "__main__":
    unittest.main()
