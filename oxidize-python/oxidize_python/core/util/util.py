"""Shared utilities mirroring oxidize-golang/core/util/util.go."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WebWorkerRequest:
    type: str
    payload: bytes | dict[str, Any] | None = None


@dataclass
class WebWorkerResponse:
    type: str
    ok: bool
    payload: bytes | dict[str, Any] | None = None
    error: str = ""


@dataclass
class GenerateRequest:
    prompt: str = ""
    max_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    stream: bool = False
    stop_token: int = 0
    seed: int = 0


@dataclass
class GenerateResponse:
    text: str = ""
    tokens: list[int] | None = None
    finish_reason: str = ""


@dataclass
class TokenEvent:
    index: int = 0
    token: int = 0
    text: str = ""


def encode_web_worker_request(req: WebWorkerRequest) -> bytes:
    payload = req.payload
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    elif payload is None:
        payload = b""
    return json.dumps({"type": req.type, "payload": payload.decode("latin-1") if payload else None}).encode()


def decode_web_worker_request(data: bytes) -> WebWorkerRequest:
    obj = json.loads(data)
    return WebWorkerRequest(type=obj["type"], payload=obj.get("payload"))


def encode_web_worker_response(resp: WebWorkerResponse) -> bytes:
    return json.dumps(
        {
            "type": resp.type,
            "ok": resp.ok,
            "payload": resp.payload,
            "error": resp.error or None,
        }
    ).encode()


def decode_web_worker_response(data: bytes) -> WebWorkerResponse:
    obj = json.loads(data)
    return WebWorkerResponse(
        type=obj["type"],
        ok=obj["ok"],
        payload=obj.get("payload"),
        error=obj.get("error") or "",
    )


@dataclass
class BenchmarkCase:
    name: str
    prompt: str
    max_tokens: int
    tags: list[str]


@dataclass
class PerplexityDatasetCase:
    name: str
    texts: list[str]


@dataclass
class Result:
    name: str
    tokens_per_sec: float
    latency_ms: float
    memory_mb: int


@dataclass
class Summary:
    count: int
    mean: float
    median: float
    p95: float
    min: float
    max: float


def summarise(results: list[Result]) -> Summary:
    if not results:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tps = sorted(r.tokens_per_sec for r in results)
    total = sum(tps)
    n = len(tps)
    median = tps[n // 2]
    p95_idx = max(0, min(n - 1, int(__import__("math").ceil(0.95 * n)) - 1))
    return Summary(
        count=n,
        mean=total / n,
        median=median,
        p95=tps[p95_idx],
        min=tps[0],
        max=tps[-1],
    )


def format_error_message(err: BaseException | None) -> str:
    if err is None:
        return ""
    return str(err)


@dataclass
class PipelineStep:
    name: str
    apply: Callable[[Any], tuple[Any, BaseException | None]]
    enabled: bool = True


class Pipeline:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = []

    def add(self, step: PipelineStep) -> None:
        if step.enabled:
            self.steps.append(step)

    def run(self, inp: Any) -> tuple[Any, BaseException | None]:
        cur = inp
        for step in self.steps:
            out, err = step.apply(cur)
            if err is not None:
                return None, PipelineError(step.name, err)
            cur = out
        return cur, None


class PipelineError(Exception):
    def __init__(self, step: str, cause: BaseException) -> None:
        self.step = step
        self.cause = cause
        super().__init__(f"pipeline[{step}]: {cause}")
