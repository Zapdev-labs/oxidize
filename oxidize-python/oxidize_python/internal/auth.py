"""API key authentication mirroring oxidize-golang/internal/auth."""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler


def wrap_handler(
    handler: type[BaseHTTPRequestHandler], expected_key: str | None = None
) -> type[BaseHTTPRequestHandler]:
    key = (
        expected_key if expected_key is not None else os.environ.get("OXIDIZE_API_KEY", "")
    ).strip()

    class AuthHandler(handler):
        def _authorized(self) -> bool:
            if not self.path.startswith("/v1/") or not key:
                return True
            return _has_api_key(self, key)

        def _reject(self) -> None:
            payload = json.dumps(
                {"error": {"message": "Invalid API key", "type": "invalid_api_key"}}
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if not self._authorized():
                self._reject()
                return
            super().do_GET()

        def do_POST(self) -> None:
            if not self._authorized():
                self._reject()
                return
            super().do_POST()

    return AuthHandler


def _has_api_key(handler: BaseHTTPRequestHandler, expected: str) -> bool:
    if _constant_time_equal(handler.headers.get("x-api-key", ""), expected):
        return True
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return _constant_time_equal(auth[7:], expected)
    query = handler.path.split("?", 1)
    if len(query) == 2:
        for part in query[1].split("&"):
            if part.startswith("api_key="):
                return _constant_time_equal(part.split("=", 1)[1], expected)
    return False


def _constant_time_equal(actual: str, expected: str) -> bool:
    if len(actual) != len(expected):
        return False
    return hmac.compare_digest(actual.encode(), expected.encode())
