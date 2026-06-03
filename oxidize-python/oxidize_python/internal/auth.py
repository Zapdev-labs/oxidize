import hmac
import json
import os
from http.server import BaseHTTPRequestHandler


def middleware(handler: type[BaseHTTPRequestHandler], expected_key: str | None = None) -> type[BaseHTTPRequestHandler]:
    key = (expected_key if expected_key is not None else os.environ.get("OXIDIZE_API_KEY", "")).strip()

    class Wrapped(handler):
        def do_GET(self) -> None:
            self._gate()

        def do_POST(self) -> None:
            self._gate()

        def _gate(self) -> None:
            if not self.path.startswith("/v1/") or not key or _has_api_key(self, key):
                return super().do_GET() if self.command == "GET" else super().do_POST()
            self._write_json({"error": {"message": "Invalid API key", "type": "invalid_api_key"}}, 401)

        def _write_json(self, body: dict, status: int) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Wrapped


def _has_api_key(handler: BaseHTTPRequestHandler, expected: str) -> bool:
    if _constant_time_equal(handler.headers.get("x-api-key", ""), expected):
        return True
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return _constant_time_equal(auth[7:], expected)
    return False


def _constant_time_equal(actual: str, expected: str) -> bool:
    if len(actual) != len(expected):
        return False
    return hmac.compare_digest(actual.encode(), expected.encode())
