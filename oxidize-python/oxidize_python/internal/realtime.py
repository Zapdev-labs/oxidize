"""Minimal WebSocket helpers for /v1/realtime (mirrors Go internal/server/realtime.go)."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
from http.server import BaseHTTPRequestHandler
from typing import Any

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def handle_realtime(handler: BaseHTTPRequestHandler) -> None:
    key = handler.headers.get("Sec-WebSocket-Key", "")
    if not key or handler.headers.get("Upgrade", "").lower() != "websocket":
        handler.send_error(400, "websocket upgrade required")
        return
    accept = base64.b64encode(
        hashlib.sha1((key + WEBSOCKET_GUID).encode()).digest()
    ).decode()
    handler.connection.sendall(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode()
    )
    _write_json(
        handler.connection,
        {"type": "session.created", "session": {"modalities": ["text"]}},
    )
    while True:
        payload, opcode = _read_frame(handler.connection)
        if payload is None:
            return
        if opcode == 0x8:
            return
        if opcode != 0x1:
            continue
        _handle_event(handler.connection, payload)


def _handle_event(conn: socket.socket, payload: bytes) -> None:
    try:
        event = json.loads(payload.decode())
    except json.JSONDecodeError:
        _write_json(conn, {"type": "error", "error": {"message": "malformed realtime event"}})
        return
    kind = event.get("type")
    if kind == "session.update":
        _write_json(conn, {"type": "session.updated", "session": event.get("session")})
    elif kind == "conversation.item.create":
        _write_json(conn, {"type": "conversation.item.created", "item": event.get("item")})
    elif kind == "response.create":
        _write_json(
            conn,
            {"type": "response.created", "response": {"status": "in_progress"}},
        )
        _write_json(conn, {"type": "error", "error": {"message": "no model loaded"}})
    elif kind == "response.cancel":
        _write_json(conn, {"type": "response.done", "response": {"status": "cancelled"}})
    else:
        _write_json(conn, {"type": "error", "error": {"message": "unsupported realtime event"}})


def _read_frame(conn: socket.socket) -> tuple[bytes | None, int]:
    header = _read_exact(conn, 2)
    if header is None:
        return None, 0
    opcode = header[0] & 0x0F
    masked = header[1] & 0x80
    length = header[1] & 0x7F
    if length == 126:
        ext = _read_exact(conn, 2)
        if ext is None:
            return None, 0
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = _read_exact(conn, 8)
        if ext is None:
            return None, 0
        length = struct.unpack(">Q", ext)[0]
    mask = b""
    if masked:
        mask = _read_exact(conn, 4) or b""
    payload = _read_exact(conn, length)
    if payload is None:
        return None, 0
    if masked and mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return payload, opcode


def _read_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _write_json(conn: socket.socket, value: dict[str, Any]) -> None:
    _write_text(conn, json.dumps(value).encode())


def _write_text(conn: socket.socket, payload: bytes) -> None:
    header = bytearray([0x81])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n <= 65535:
        header.extend([126, (n >> 8) & 0xFF, n & 0xFF])
    else:
        header.extend(
            [127, 0, 0, 0, 0, (n >> 24) & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF]
        )
    conn.sendall(bytes(header) + payload)
