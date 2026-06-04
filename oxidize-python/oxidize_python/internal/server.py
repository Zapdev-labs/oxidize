"""OpenAI-compatible HTTP server mirroring oxidize-golang/internal/server."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from oxidize_python.core.model.loader import LoaderConfig
from oxidize_python.internal.api.responses import (
    build_chat_chunk,
    build_chat_completion,
    build_embeddings_response,
    build_models_response,
    build_text_chunk,
    build_text_completion,
    chat_response_to_dict,
    error_response_to_dict,
    malformed_json,
    model_not_found,
    text_response_to_dict,
    validate_candidate_count,
)
from oxidize_python.internal.api.schema import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingsRequest,
)
from oxidize_python.internal.generate import PlaceholderSpec, placeholder_text
from oxidize_python.internal.generate.cache import default_model_cache
from oxidize_python.internal.generate.stream import CompletionParams, stream_completion
from oxidize_python.internal.serviceinfo.models import default_model_id, discover_models

MAX_JSON_BODY_BYTES = 1 << 20


class _App:
    def __init__(self, models_dir: str) -> None:
        self.models = discover_models(models_dir)
        self.model_ids = {m.id for m in self.models}
        self._paths = {m.id: m.path for m in self.models}
        self._cache = default_model_cache
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_inflight = 0

    def default_model(self) -> str:
        return default_model_id(self.models)

    def model_path(self, model_id: str) -> str:
        return self._paths.get(model_id, "")

    def ensure_model(self, model_id: str) -> bool:
        if not self.model_ids:
            return True
        return model_id in self.model_ids

    def completion_text(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float | None = None,
    ) -> str:
        path = self.model_path(model_id)
        if not path and self.models:
            path = self.models[0].path
        if not path:
            return ""
        try:
            from oxidize_python.internal.generate.runtime import completion_text

            return completion_text(path, prompt, max_tokens, temperature=temperature)
        except Exception:
            return ""

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def models_list(self) -> dict[str, Any]:
        if not self.models:
            ids = [self.default_model()]
        else:
            ids = [m.id for m in self.models]
        resp = build_models_response(*ids)
        return {
            "object": resp.object,
            "data": [asdict(m) for m in resp.data],
        }

    def metrics_text(self) -> str:
        with self._lock:
            requests_total = self.requests_total
            requests_inflight = self.requests_inflight
        return (
            "# HELP oxidize_requests_total Total HTTP requests handled.\n"
            "# TYPE oxidize_requests_total counter\n"
            f"oxidize_requests_total {requests_total}\n"
            "# HELP oxidize_requests_inflight HTTP requests currently in flight.\n"
            "# TYPE oxidize_requests_inflight gauge\n"
            f"oxidize_requests_inflight {requests_inflight}\n"
        )

    def chat_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        req = ChatCompletionRequest.from_json(body)
        if err := validate_candidate_count(req.n, req.best_of):
            return error_response_to_dict(err), err.status_code
        model = req.model or self.default_model()
        if not self.ensure_model(model):
            err = model_not_found(model)
            return error_response_to_dict(err), err.status_code
        prompt = req.first_user_message()
        max_tokens = req.max_tokens_or(128)
        temp = req.temperature
        text = self.completion_text(model, prompt, max_tokens, temperature=temp)
        if not text:
            text = placeholder_text(PlaceholderSpec()) or f"oxidize-python: {prompt[:80]}"
        if req.stream:
            path = self.model_path(model)
            if path:
                return {
                    "sse_tokens": True,
                    "model": model,
                    "path": path,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                }, 200
            return {"sse": True, "chunks": [text, ""]}, 200
        resp = build_chat_completion(model, text)
        return chat_response_to_dict(resp), 200

    def text_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        req = CompletionRequest.from_json(body)
        if err := validate_candidate_count(req.n, req.best_of):
            return error_response_to_dict(err), err.status_code
        model = req.model or self.default_model()
        if not self.ensure_model(model):
            err = model_not_found(model)
            return error_response_to_dict(err), err.status_code
        max_tokens = req.max_tokens_or(128)
        text = self.completion_text(model, req.prompt, max_tokens, temperature=req.temperature)
        if not text:
            text = placeholder_text(PlaceholderSpec()) or req.prompt
        if req.stream:
            path = self.model_path(model)
            if path:
                return {
                    "sse_tokens": True,
                    "model": model,
                    "path": path,
                    "prompt": req.prompt,
                    "max_tokens": max_tokens,
                    "temperature": req.temperature,
                }, 200
            return {"sse": True, "chunks": [text, ""]}, 200
        resp = build_text_completion(model, text)
        return text_response_to_dict(resp), 200

    def embeddings(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        req = EmbeddingsRequest.from_json(body)
        model = req.model or self.default_model()
        if not self.ensure_model(model):
            err = model_not_found(model)
            return error_response_to_dict(err), err.status_code
        resp = build_embeddings_response(model)
        return {
            "object": resp.object,
            "model": resp.model,
            "data": [asdict(d) for d in resp.data],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }, 200

    def mesh_chat_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        ChatCompletionRequest.from_json(body)
        return {
            "error": {
                "message": "mesh runtime is not configured",
                "type": "service_unavailable",
            }
        }, 503


def listen(host: str = "127.0.0.1", port: int = 8080, models_dir: str = "") -> None:
    app = _App(models_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_JSON_BODY_BYTES:
                self._json(
                    {
                        "error": {
                            "message": "request body too large",
                            "type": "invalid_request_error",
                        }
                    },
                    413,
                )
                return None
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                err = malformed_json()
                self._json(error_response_to_dict(err), err.status_code)
                return None

        def _json(self, obj: dict[str, Any], status: int = 200) -> None:
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _text(self, payload: str, content_type: str, status: int = 200) -> None:
            raw = payload.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _sse_chat_tokens(
            self,
            model: str,
            path: str,
            prompt: str,
            max_tokens: int,
            temperature: float | None,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            temp = 0.0 if temperature is None else float(temperature)
            params = CompletionParams(
                max_tokens=max_tokens,
                temperature=temp,
                loader=LoaderConfig(),
            )
            pieces: list[str] = []

            def on_piece(piece: str) -> None:
                pieces.append(piece)
                chunk = build_chat_chunk(model, piece, False)
                data = json.dumps(chat_response_to_dict(chunk))
                self.wfile.write(f"data: {data}\n\n".encode())

            text = stream_completion(path, prompt, params, on_piece)
            if not text and not pieces:
                fallback = placeholder_text(PlaceholderSpec()) or prompt[:80]
                chunk = build_chat_chunk(model, fallback, True)
                self.wfile.write(f"data: {json.dumps(chat_response_to_dict(chunk))}\n\n".encode())
            else:
                chunk = build_chat_chunk(model, "", True)
                self.wfile.write(f"data: {json.dumps(chat_response_to_dict(chunk))}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def _sse_text_tokens(
            self,
            model: str,
            path: str,
            prompt: str,
            max_tokens: int,
            temperature: float | None,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            temp = 0.0 if temperature is None else float(temperature)
            params = CompletionParams(
                max_tokens=max_tokens, temperature=temp, loader=LoaderConfig()
            )

            def on_piece(piece: str) -> None:
                chunk = build_text_chunk(model, piece, False)
                data = json.dumps(text_response_to_dict(chunk))
                self.wfile.write(f"data: {data}\n\n".encode())

            text = stream_completion(path, prompt, params, on_piece)
            if not text:
                fallback = placeholder_text(PlaceholderSpec()) or prompt
                chunk = build_text_chunk(model, fallback, True)
                self.wfile.write(f"data: {json.dumps(text_response_to_dict(chunk))}\n\n".encode())
            else:
                chunk = build_text_chunk(model, "", True)
                self.wfile.write(f"data: {json.dumps(text_response_to_dict(chunk))}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def _sse_chat(self, model: str, chunks: list[str]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for i, piece in enumerate(chunks):
                finished = i == len(chunks) - 1
                chunk = build_chat_chunk(model, piece, finished)
                data = json.dumps(chat_response_to_dict(chunk))
                self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def _sse_text(self, model: str, chunks: list[str]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for i, piece in enumerate(chunks):
                finished = i == len(chunks) - 1
                chunk = build_text_chunk(model, piece, finished)
                data = json.dumps(text_response_to_dict(chunk))
                self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def do_GET(self) -> None:
            if self.path in ("/healthz", "/livez", "/readyz"):
                self._json(app.health())
            elif self.path == "/metrics":
                self._text(app.metrics_text(), "text/plain; version=0.0.4")
            elif self.path == "/openapi.json":
                self._json(
                    {
                        "openapi": "3.0.0",
                        "info": {"title": "oxidize-python", "version": "0.1.0"},
                    }
                )
            elif self.path == "/v1/models":
                self._json(app.models_list())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            with app._lock:
                app.requests_total += 1
                app.requests_inflight += 1
            try:
                body = self._read_json()
                if body is None:
                    return
                if self.path == "/v1/chat/completions":
                    result, status = app.chat_completion(body)
                    if result.get("sse_tokens"):
                        self._sse_chat_tokens(
                            result["model"],
                            result["path"],
                            result["prompt"],
                            result["max_tokens"],
                            result.get("temperature"),
                        )
                        return
                    if result.get("sse"):
                        model = str(body.get("model") or app.default_model())
                        self._sse_chat(model, result["chunks"])
                        return
                    self._json(result, status)
                elif self.path == "/v1/completions":
                    result, status = app.text_completion(body)
                    if result.get("sse_tokens"):
                        self._sse_text_tokens(
                            result["model"],
                            result["path"],
                            result["prompt"],
                            result["max_tokens"],
                            result.get("temperature"),
                        )
                        return
                    if result.get("sse"):
                        model = str(body.get("model") or app.default_model())
                        self._sse_text(model, result["chunks"])
                        return
                    self._json(result, status)
                elif self.path == "/v1/embeddings":
                    result, status = app.embeddings(body)
                    self._json(result, status)
                elif self.path == "/v1/mesh/chat/completions":
                    result, status = app.mesh_chat_completion(body)
                    self._json(result, status)
                else:
                    self.send_error(404)
            finally:
                with app._lock:
                    app.requests_inflight -= 1

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"oxidize-python server listening on http://{host}:{port}")
    httpd.serve_forever()
