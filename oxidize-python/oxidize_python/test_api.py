from __future__ import annotations

from oxidize_python.internal.api.responses import (
    build_text_completion,
    model_not_found,
    text_response_to_dict,
    validate_candidate_count,
)
from oxidize_python.internal.api.schema import ChatCompletionRequest, CompletionRequest


def test_chat_request_parses_stream_and_temperature() -> None:
    raw = {
        "model": "demo",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0.5,
        "max_tokens": 64,
    }
    req = ChatCompletionRequest.from_json(raw)
    assert req.stream is True
    assert req.temperature == 0.5
    assert req.max_tokens_or(128) == 64
    assert req.first_user_message() == "hello"


def test_completion_request_prompt() -> None:
    req = CompletionRequest.from_json({"model": "m", "prompt": "ping", "max_tokens": 10})
    assert req.prompt == "ping"
    assert req.max_tokens_or(128) == 10


def test_validate_candidate_count_rejects_zero_n() -> None:
    err = validate_candidate_count(0, None)
    assert err is not None
    assert err.status_code == 400


def test_model_not_found_status() -> None:
    err = model_not_found("ghost")
    assert err.status_code == 404


def test_text_response_roundtrip() -> None:
    resp = build_text_completion("demo", "ok")
    data = text_response_to_dict(resp)
    assert data["choices"][0]["text"] == "ok"


def test_server_metrics_are_prometheus_text(tmp_path) -> None:
    from oxidize_python.internal.server import _App

    app = _App(str(tmp_path))
    app.requests_total = 3
    text = app.metrics_text()
    assert "# TYPE oxidize_requests_total counter" in text
    assert "oxidize_requests_total 3" in text
    assert "oxidize_requests_inflight 0" in text


def test_mesh_chat_disabled_shape(tmp_path) -> None:
    from oxidize_python.internal.server import _App

    app = _App(str(tmp_path))
    payload, status = app.mesh_chat_completion(
        {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert status == 503
    assert payload["error"]["message"] == "mesh runtime is not configured"
