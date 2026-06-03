"""Minimal OpenAI-compatible HTTP server mirroring oxidize-server routes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from oxidize_python.internal.generate import PlaceholderSpec, placeholder_text


class _App:
    def __init__(self, models_dir: str) -> None:
        self.models: list[dict[str, str]] = []
        if models_dir:
            for p in sorted(Path(models_dir).glob("*.gguf")):
                self.models.append({"id": p.name, "path": str(p)})
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_inflight = 0

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def models_list(self) -> dict[str, Any]:
        data = [
            {
                "id": m["id"],
                "object": "model",
                "created": 0,
                "owned_by": "oxidize",
            }
            for m in self.models
        ]
        if not data:
            data = [{"id": "oxidize-default", "object": "model", "created": 0, "owned_by": "oxidize"}]
        return {"object": "list", "data": data}

    def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model", "oxidize-default")
        messages = body.get("messages") or []
        content = ""
        if messages:
            last = messages[-1]
            c = last.get("content", "")
            content = c if isinstance(c, str) else str(c)
        text = placeholder_text(PlaceholderSpec()) or f"oxidize-python placeholder: {content[:80]}"
        return {
            "id": "chatcmpl-oxidize",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def listen(host: str = "127.0.0.1", port: int = 8080, models_dir: str = "") -> None:
    app = _App(models_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode() or "{}")

        def _json(self, obj: dict[str, Any], status: int = 200) -> None:
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path in ("/healthz", "/livez", "/readyz"):
                self._json(app.health())
            elif self.path == "/metrics":
                self._json({"requests_total": app.requests_total, "requests_inflight": app.requests_inflight})
            elif self.path == "/openapi.json":
                self._json({"openapi": "3.0.0", "info": {"title": "oxidize-python", "version": "0.1.0"}})
            elif self.path == "/v1/models":
                self._json(app.models_list())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/v1/chat/completions":
                self._json(app.chat_completion(self._read_json()))
            elif self.path == "/v1/completions":
                body = self._read_json()
                prompt = body.get("prompt", "")
                text = placeholder_text(PlaceholderSpec()) or str(prompt)
                self._json(
                    {
                        "id": "cmpl-oxidize",
                        "object": "text_completion",
                        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                    }
                )
            elif self.path == "/v1/embeddings":
                self._json(
                    {
                        "object": "list",
                        "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 8}],
                    }
                )
            else:
                self.send_error(404)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"oxidize-python server listening on http://{host}:{port}")
    httpd.serve_forever()
