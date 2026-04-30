import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_benchmark_dashboard import build_dashboard_data, render_html_dashboard
from test_ci_benchmark_regression import _write_benchmark


class BenchmarkDashboardTests(unittest.TestCase):
    def test_builds_rows_with_ratio_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_benchmark(
                root,
                "loader_mapped",
                "loader/mapped_gguf/fixture/valid-v3.gguf",
                130.0,
            )
            _write_benchmark(
                root,
                "loader_baseline",
                "loader/llama_cpp_baseline/fixture/valid-v3.gguf",
                100.0,
            )

            data = build_dashboard_data(root, loader_threshold=1.25, memory_threshold=1.4)
            self.assertEqual(data["benchmark_count"], 1)
            row = data["rows"][0]
            self.assertEqual(row["status"], "regression")
            self.assertEqual(row["name"], "fixture/valid-v3.gguf")

    def test_marks_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_benchmark(
                root,
                "loader_mapped",
                "loader/mapped_gguf/fixture/valid-v3.gguf",
                101.0,
            )

            data = build_dashboard_data(root, loader_threshold=1.25, memory_threshold=1.4)
            row = data["rows"][0]
            self.assertEqual(row["status"], "missing-baseline")
            self.assertIsNone(row["ratio"])

    def test_renders_html_with_rows(self) -> None:
        data = {
            "criterion_dir": "target/criterion",
            "benchmark_count": 1,
            "rows": [
                {
                    "category": "loader",
                    "name": "fixture/valid-v3.gguf",
                    "mapped_mean_ns": 120.0,
                    "baseline_mean_ns": 100.0,
                    "ratio": 1.2,
                    "threshold": 1.25,
                    "status": "ok",
                }
            ],
        }

        html = render_html_dashboard(data)
        self.assertIn("<h1>Benchmark Dashboard</h1>", html)
        self.assertIn("fixture/valid-v3.gguf", html)
        self.assertIn("status-ok", html)

    def test_dashboard_data_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
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
            data = build_dashboard_data(root, loader_threshold=1.25, memory_threshold=1.4)
            encoded = json.dumps(data)
            self.assertIn("memory", encoded)


if __name__ == "__main__":
    unittest.main()
