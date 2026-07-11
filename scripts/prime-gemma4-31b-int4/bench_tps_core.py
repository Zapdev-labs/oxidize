from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import socket
import threading
import time
from typing import Callable, Final, Literal, Sequence, TypedDict

import httpx2


Phase = Literal["warmup", "measure"]
ENGINE_HEADER: Final = "X-Oxidize-Engine"
WORKER_HEADER: Final = "X-Oxidize-Benchmark-Worker"


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    endpoint_index: int
    endpoint_url: str
    expected_worker: str
    client_index: int


@dataclass(frozen=True, slots=True)
class RequestRecord:
    endpoint_index: int
    endpoint_url: str
    expected_worker: str
    client_index: int
    phase: Phase
    started_at: float
    finished_at: float
    completion_tokens: int | None
    text: str
    error: str | None
    server_identity: str | None = None
    worker_identity: str | None = None
    http_status: int = 0


class BenchmarkSummary(TypedDict):
    aggregate_generated_tps: float
    valid_generated_tokens: int
    required_generated_tokens: int
    errors: int
    gpu_workers_covered: list[str]
    gpu_workers_expected: list[str]
    duration_s: float
    target_tps: float
    passed: bool


RequestFunction = Callable[..., RequestRecord]


def create_client(total_loops: int, timeout_s: float) -> httpx2.Client:
    limits = httpx2.Limits(
        max_connections=max(200, total_loops),
        max_keepalive_connections=40,
        keepalive_expiry=30.0,
    )
    timeout = httpx2.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=10.0)
    # Retries are deliberately disabled so every transport failure reaches the ledger.
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=0,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    return httpx2.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
    )


def worker_specs(endpoints: Sequence[str], concurrency_per_gpu: int) -> list[WorkerSpec]:
    return [
        WorkerSpec(endpoint_index, endpoint_url, str(endpoint_index), client_index)
        for endpoint_index, endpoint_url in enumerate(endpoints)
        for client_index in range(concurrency_per_gpu)
    ]


def verify_completion(
    *,
    status: int,
    body: bytes,
    server_identity: str | None,
    worker_identity: str | None,
    required_identity: str,
    expected_worker: str,
) -> tuple[int, str] | str:
    if status != 200:
        return f"HTTP status {status}"
    if server_identity != required_identity:
        return f"missing or invalid {ENGINE_HEADER}: {server_identity!r}"
    if worker_identity != expected_worker:
        return (
            f"missing or invalid {WORKER_HEADER}: {worker_identity!r}; "
            f"expected {expected_worker!r}"
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        return f"invalid JSON response: {error.msg}"
    if not isinstance(payload, dict):
        return "response is not a JSON object"
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return "missing usage"
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return "missing usage.completion_tokens"
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "missing choices[0]"
    choice = choices[0]
    text = choice.get("text")
    if not isinstance(text, str):
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        return "empty response text"
    return completion_tokens, text


def request_completion(
    *,
    client: httpx2.Client,
    spec: WorkerSpec,
    model: str,
    prompt: str,
    max_tokens: int,
    phase: Phase,
    required_identity: str,
) -> RequestRecord:
    started_at = time.perf_counter()
    try:
        response = client.post(
            f"{spec.endpoint_url.rstrip('/')}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
    except httpx2.HTTPError as error:
        return _failed_record(spec, phase, started_at, f"{type(error).__name__}: {error}")
    finished_at = time.perf_counter()
    server_identity = response.headers.get(ENGINE_HEADER)
    worker_identity = response.headers.get(WORKER_HEADER)
    verified = verify_completion(
        status=response.status_code,
        body=response.content,
        server_identity=server_identity,
        worker_identity=worker_identity,
        required_identity=required_identity,
        expected_worker=spec.expected_worker,
    )
    if isinstance(verified, str):
        return RequestRecord(
            spec.endpoint_index, spec.endpoint_url, spec.expected_worker,
            spec.client_index, phase, started_at, finished_at, None, "", verified,
            server_identity, worker_identity, response.status_code,
        )
    completion_tokens, text = verified
    return RequestRecord(
        spec.endpoint_index, spec.endpoint_url, spec.expected_worker,
        spec.client_index, phase, started_at, finished_at, completion_tokens, text,
        None, server_identity, worker_identity, response.status_code,
    )


def _failed_record(
    spec: WorkerSpec,
    phase: Phase,
    started_at: float,
    error: str,
) -> RequestRecord:
    return RequestRecord(
        spec.endpoint_index, spec.endpoint_url, spec.expected_worker, spec.client_index,
        phase, started_at, time.perf_counter(), None, "", error,
    )


def run_phase(
    *,
    client: httpx2.Client,
    endpoints: Sequence[str],
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency_per_gpu: int,
    duration_s: float,
    phase: Phase,
    required_identity: str,
    request_fn: RequestFunction = request_completion,
) -> tuple[list[RequestRecord], float, float]:
    specs = worker_specs(endpoints, concurrency_per_gpu)
    start_gate = threading.Event()
    deadline: list[float] = []

    def run_worker(spec: WorkerSpec) -> list[RequestRecord]:
        start_gate.wait()
        records: list[RequestRecord] = []
        while time.perf_counter() < deadline[0]:
            record = request_fn(
                client=client, spec=spec, model=model, prompt=prompt,
                max_tokens=max_tokens, phase=phase, required_identity=required_identity,
            )
            records.append(record)
            if record.error is not None:
                break
        return records

    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = [executor.submit(run_worker, spec) for spec in specs]
        started_at = time.perf_counter()
        ended_at = started_at + duration_s
        deadline.append(ended_at)
        start_gate.set()
        records = [record for future in futures for record in future.result()]
    return records, started_at, ended_at


def summarize_window(
    *,
    records: Sequence[RequestRecord],
    expected_workers: Sequence[str],
    started_at: float,
    ended_at: float,
    target_tps: float,
) -> BenchmarkSummary:
    duration_s = ended_at - started_at
    attempts = [
        record for record in records
        if record.phase == "measure" and started_at <= record.started_at < ended_at
    ]
    valid = [
        record for record in attempts
        if record.finished_at <= ended_at and record.error is None
        and record.completion_tokens is not None and bool(record.text.strip())
    ]
    valid_tokens = sum(record.completion_tokens or 0 for record in valid)
    covered = sorted({record.expected_worker for record in valid})
    expected = list(expected_workers)
    errors = sum(record.error is not None for record in attempts)
    aggregate_tps = valid_tokens / duration_s if duration_s > 0 else 0.0
    required_tokens = math.ceil(target_tps * duration_s) if duration_s > 0 else 0
    passed = (
        duration_s > 0 and aggregate_tps >= target_tps
        and valid_tokens >= required_tokens and errors == 0 and covered == expected
    )
    return {
        "aggregate_generated_tps": aggregate_tps,
        "valid_generated_tokens": valid_tokens,
        "required_generated_tokens": required_tokens,
        "errors": errors,
        "gpu_workers_covered": covered,
        "gpu_workers_expected": expected,
        "duration_s": duration_s,
        "target_tps": target_tps,
        "passed": passed,
    }


def write_ledger(path: Path, records: Sequence[RequestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(asdict(record), sort_keys=True)}\n" for record in records),
    )
