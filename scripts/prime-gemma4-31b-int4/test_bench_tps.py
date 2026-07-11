"""Behavioral tests for process-per-GPU oxidize-c TPS aggregation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Literal

import httpx2
import pytest


BENCH_PATH = Path(__file__).with_name("bench_tps.py")
sys.path.insert(0, str(BENCH_PATH.parent))
SPEC = importlib.util.spec_from_file_location("bench_tps", BENCH_PATH)
assert SPEC is not None
assert SPEC.loader is not None
bench_tps = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_tps
SPEC.loader.exec_module(bench_tps)


def record(
    endpoint_index: int,
    *,
    tokens: int | None = 600,
    error: str | None = None,
    started_at: float = 10.1,
    finished_at: float = 10.9,
    phase: Literal["warmup", "measure"] = "measure",
) -> bench_tps.RequestRecord:
    return bench_tps.RequestRecord(
        endpoint_index=endpoint_index,
        endpoint_url=f"http://127.0.0.1:{8080 + endpoint_index}/v1",
        expected_worker=str(endpoint_index),
        client_index=0,
        phase=phase,
        started_at=started_at,
        finished_at=finished_at,
        completion_tokens=tokens,
        text="generated output" if error is None else "",
        error=error,
        server_identity="oxidize-c" if error is None else None,
        worker_identity=str(endpoint_index) if error is None else None,
    )


def summarize(records: list[bench_tps.RequestRecord], *, duration: float = 1.0) -> bench_tps.BenchmarkSummary:
    return bench_tps.summarize_window(
        records=records,
        expected_workers=("0", "1"),
        started_at=10.0,
        ended_at=10.0 + duration,
        target_tps=1000.0,
    )


def response_body(tokens: int = 8, text: str = "hello") -> bytes:
    return json.dumps(
        {"usage": {"completion_tokens": tokens}, "choices": [{"text": text}]},
    ).encode()


def test_two_urls_aggregate_verified_generated_tokens() -> None:
    # Given: one valid measured completion from each static GPU worker.
    records = [record(0), record(1)]

    # When: the shared one-second measurement window is summarized.
    summary = summarize(records)

    # Then: aggregate TPS and coverage are reported across both endpoints.
    assert summary["aggregate_generated_tps"] == 1200.0
    assert summary["valid_generated_tokens"] == 1200
    assert summary["errors"] == 0
    assert summary["gpu_workers_covered"] == ["0", "1"]
    assert summary["duration_s"] == 1.0
    assert summary["passed"] is True


def test_wrong_static_worker_header_is_rejected() -> None:
    # Given: endpoint 1 responds with endpoint 0's static worker header.
    # When: the completion evidence is verified.
    result = bench_tps.verify_completion(
        status=200,
        body=response_body(),
        server_identity="oxidize-c",
        worker_identity="0",
        required_identity="oxidize-c",
        expected_worker="1",
    )

    # Then: client request identity cannot masquerade as GPU ownership.
    assert result == "missing or invalid X-Oxidize-Benchmark-Worker: '0'; expected '1'"


def test_http_adapter_uses_static_response_identity_without_client_echo() -> None:
    # Given: a wire-level endpoint fake with static oxidize-c ownership headers.
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "X-Oxidize-Benchmark-Worker" not in request.headers
        assert "X-Oxidize-Engine" not in request.headers
        return httpx2.Response(
            200,
            headers={
                "X-Oxidize-Engine": "oxidize-c",
                "X-Oxidize-Benchmark-Worker": "1",
            },
            json={
                "usage": {"completion_tokens": 8},
                "choices": [{"message": {"content": "generated"}}],
            },
        )

    spec = bench_tps.WorkerSpec(1, "http://worker-1/v1", "1", 0)
    with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
        # When: one real HTTP adapter call is made through the fake transport.
        result = bench_tps.core.request_completion(
            client=client,
            spec=spec,
            model="gemma4",
            prompt="hello",
            max_tokens=8,
            phase="measure",
            required_identity="oxidize-c",
        )

    # Then: only static response headers prove worker coverage.
    assert result.error is None
    assert result.worker_identity == "1"
    assert result.completion_tokens == 8
    assert result.text == "generated"


def test_missing_url_fails_worker_coverage_even_above_tps_target() -> None:
    # Given: endpoint 0 alone reports enough tokens to exceed the aggregate target.
    records = [record(0, tokens=2000)]

    # When: two configured endpoints are summarized.
    summary = summarize(records)

    # Then: absent endpoint 1 keeps the benchmark fail-closed.
    assert summary["aggregate_generated_tps"] == 2000.0
    assert summary["gpu_workers_covered"] == ["0"]
    assert summary["passed"] is False


def test_error_response_fails_otherwise_valid_measurement() -> None:
    # Given: both endpoints generated valid tokens but one request also failed.
    records = [record(0), record(1), record(1, tokens=None, error="HTTP status 500")]

    # When: measurement evidence is summarized.
    summary = summarize(records)

    # Then: any measured request error rejects the run.
    assert summary["valid_generated_tokens"] == 1200
    assert summary["errors"] == 1
    assert summary["passed"] is False


def test_threshold_requires_target_tokens_for_full_duration() -> None:
    # Given: the 120-second run is one generated token below its 120,000 minimum.
    records = [
        record(0, tokens=60_000, started_at=10.1, finished_at=129.8),
        record(1, tokens=59_999, started_at=10.2, finished_at=129.9),
    ]

    # When: the fixed criterion window is summarized.
    summary = summarize(records, duration=120.0)

    # Then: reported volume and TPS cannot round up to a pass.
    assert summary["valid_generated_tokens"] == 119_999
    assert summary["required_generated_tokens"] == 120_000
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--urls", "http://127.0.0.1:8080/v1,"],
        ["--urls", "http://127.0.0.1:8080/v1,http://127.0.0.1:8080/v1"],
        ["--urls", "not-a-url"],
        ["--urls", "http://127.0.0.1:8080/v1", "--concurrency-per-gpu", "0"],
        ["--urls", "http://127.0.0.1:8080/v1", "--duration-seconds", "0"],
    ],
)
def test_malformed_cli_arguments_are_rejected(argv: list[str]) -> None:
    # Given: malformed or ambiguous process-per-GPU arguments.
    # When/Then: parsing exits before any benchmark request can start.
    with pytest.raises(SystemExit) as raised:
        bench_tps.parse_config(argv)
    assert raised.value.code == 2


def test_cli_default_completion_fits_ctx128() -> None:
    # Given: the benchmark CLI is invoked without an explicit completion limit.
    # When: its configuration is parsed.
    config = bench_tps.parse_config([])

    # Then: the default leaves prompt capacity within the server's 128-token context.
    assert config.max_tokens == 64


def test_worker_specs_assign_each_client_loop_to_one_url() -> None:
    # Given: two GPU endpoints and two client loops per endpoint.
    # When: deterministic worker assignments are built.
    specs = bench_tps.worker_specs(
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8081/v1"),
        concurrency_per_gpu=2,
    )

    # Then: endpoint and client indices are stable and never load-balanced implicitly.
    assert specs == [
        bench_tps.WorkerSpec(0, "http://127.0.0.1:8080/v1", "0", 0),
        bench_tps.WorkerSpec(0, "http://127.0.0.1:8080/v1", "0", 1),
        bench_tps.WorkerSpec(1, "http://127.0.0.1:8081/v1", "1", 0),
        bench_tps.WorkerSpec(1, "http://127.0.0.1:8081/v1", "1", 1),
    ]


def test_cli_runs_one_warmup_then_one_measurement_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a deterministic in-memory phase runner for two endpoints.
    calls: list[str] = []

    def fake_run_phase(**kwargs: str | int | float | tuple[str, ...]) -> tuple[list[bench_tps.RequestRecord], float, float]:
        phase = kwargs["phase"]
        assert isinstance(phase, str)
        calls.append(phase)
        if phase == "warmup":
            return [record(0, phase="warmup"), record(1, phase="warmup")], 5.0, 6.0
        return [record(0), record(1)], 10.0, 11.0

    monkeypatch.setattr(bench_tps, "run_phase", fake_run_phase)
    ledger = tmp_path / "requests.jsonl"
    summary_path = tmp_path / "summary.json"

    # When: the required process-per-GPU CLI is executed.
    exit_code = bench_tps.main(
        [
            "--urls",
            "http://127.0.0.1:8080/v1,http://127.0.0.1:8081/v1",
            "--concurrency-per-gpu",
            "2",
            "--warmup-seconds",
            "1",
            "--duration-seconds",
            "1",
            "--target-tps",
            "1000",
            "--ledger",
            str(ledger),
            "--summary",
            str(summary_path),
        ],
    )

    # Then: exactly one shared warmup and one shared measurement produce durable evidence.
    summary = json.loads(summary_path.read_text())
    ledger_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert exit_code == 0
    assert calls == ["warmup", "measure"]
    assert summary["aggregate_generated_tps"] == 1200.0
    assert summary["gpu_workers_covered"] == ["0", "1"]
    assert len(ledger_rows) == 4
    assert {row["endpoint_url"] for row in ledger_rows} == {
        "http://127.0.0.1:8080/v1",
        "http://127.0.0.1:8081/v1",
    }
