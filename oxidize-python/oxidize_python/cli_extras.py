"""Pipeline, mesh, and profile helpers mirroring oxidize-golang/internal/cli."""

from __future__ import annotations

import socket
from typing import IO

from oxidize_python.cli_flags import RunOptions
from oxidize_python.core.mesh.mesh import MeshChatEngine, MeshNode


def maybe_run_pipeline(opts: RunOptions, model_path: str, stdout: IO[str]) -> bool:
    if not opts.pipe_head and not opts.pipe_tail:
        return False
    listen = opts.pipe_listen.strip() or "127.0.0.1:9901"
    peer = opts.pipe_peer.strip()
    if opts.pipe_head:
        stdout.write(f"pipeline head: listen={listen} peer={peer or '(none)'}\n")
        if peer:
            host, _, port_s = peer.rpartition(":")
            with socket.create_connection((host, int(port_s)), timeout=5) as conn:
                conn.sendall((opts.prompt + "\n").encode())
                stdout.write(conn.recv(4096).decode(errors="replace"))
        return True
    if opts.pipe_tail:
        host, _, port_s = listen.partition(":")
        port = int(port_s or "9901")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host or "127.0.0.1", port))
            srv.listen(1)
            stdout.write(f"pipeline tail: listening on {listen}\n")
            conn, _ = srv.accept()
            with conn:
                data = conn.recv(65536).decode(errors="replace").strip()
                from oxidize_python.internal.generate.runtime import run_from_gguf

                cfg = opts.run_config(model_path)
                cfg.prompt = data or cfg.prompt
                run_from_gguf(cfg, stdout)
                conn.sendall(b"\n")
        return True
    return False


def maybe_run_mesh_chat(
    opts: RunOptions,
    model_path: str,
    stdout: IO[str],
    stderr: IO[str],
) -> bool:
    if not opts.mesh:
        return False
    port = opts.mesh_port or 0
    local = MeshNode(
        id="local",
        addr=f"127.0.0.1:{port}",
        role="worker",
        healthy=True,
    )
    engine = MeshChatEngine(local)
    engine.router.update(local)
    stdout.write(
        f"oxidize mesh chat (gossip engine). peers={len(engine.router.peers())}. "
        "type exit to quit.\n"
    )
    cfg = opts.run_config(model_path)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=stdout)
            return True
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            return True
        for peer in engine.router.peers():
            if peer.id != local.id:
                engine.router.update(peer)
        cfg.prompt = line
        from oxidize_python.internal.generate.runtime import run_from_gguf

        try:
            run_from_gguf(cfg, stdout)
        except Exception as err:
            print(f"generation failed: {err}", file=stderr)
        print(file=stdout)
    return True
