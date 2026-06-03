"""CLI mirroring oxidize-cli / oxidize-golang/internal/cli."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from oxidize_python.internal.generate import cli_transcript


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        return _run_legacy(args)
    cmd = args[0]
    rest = args[1:]
    if cmd == "run":
        return _run_command(rest)
    if cmd in ("list", "ls"):
        return _list_command(rest)
    if cmd == "serve":
        return _serve_command(rest)
    if cmd in ("-h", "--help", "help"):
        _print_help()
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    _print_help(file=sys.stderr)
    return 1


def _run_legacy(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize", add_help=False)
    p.add_argument("--prompt", default="")
    ns, _ = p.parse_known_args(args)
    if ns.prompt.strip():
        sys.stdout.write(cli_transcript(ns.prompt))
    return 0


def _run_command(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize run", add_help=False)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=128)
    ns, positional = p.parse_known_args(args)
    if not positional:
        print("oxidize run requires a model name or local .gguf path", file=sys.stderr)
        return 1
    prompt = ns.prompt.strip()
    if not prompt:
        return 0
    model_path = _resolve_model_path(positional[0])
    if model_path.lower().endswith(".gguf"):
        try:
            from oxidize_python.internal.generate.runtime import RunConfig, run_from_gguf

            run_from_gguf(
                RunConfig(model_path=model_path, prompt=prompt, max_new_tokens=ns.max_tokens),
                sys.stdout,
            )
            return 0
        except Exception as e:
            print(f"generation failed: {e}", file=sys.stderr)
            return 1
    sys.stdout.write(cli_transcript(prompt))
    return 0


def _resolve_model_path(name_or_path: str) -> str:
    if name_or_path.lower().endswith(".gguf"):
        return name_or_path
    models_dir = _resolve_models_dir("")
    if not models_dir:
        return name_or_path
    candidate = Path(models_dir) / f"{name_or_path}.gguf"
    return str(candidate) if candidate.is_file() else name_or_path


def _list_command(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize list", add_help=False)
    p.add_argument("--models-dir", default="")
    ns, _ = p.parse_known_args(args)
    models_dir = _resolve_models_dir(ns.models_dir)
    sys.stdout.write(f"{'NAME':<48} {'SIZE':>9} PATH\n")
    if not models_dir:
        return 0
    root = Path(models_dir)
    if not root.is_dir():
        return 0
    for path in sorted(root.glob("*.gguf")):
        size = path.stat().st_size / (1024**3)
        sys.stdout.write(f"{path.name:<48} {size:>8.2f}G {path}\n")
    return 0


def _serve_command(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--models-dir", default="")
    ns = p.parse_args(args)
    try:
        from oxidize_python.internal.server import listen
    except ImportError as e:
        print(f"server module not available: {e}", file=sys.stderr)
        return 1
    with contextlib.suppress(KeyboardInterrupt):
        listen(host=ns.host, port=ns.port, models_dir=_resolve_models_dir(ns.models_dir))
    return 0


def _resolve_models_dir(raw: str) -> str:
    if raw.strip():
        return raw
    candidate = Path.cwd() / "models"
    return str(candidate) if candidate.is_dir() else ""


def _print_help(file: object = sys.stdout) -> None:
    print(
        """Usage: oxidize <command> [args]

Commands:
  run <model> [prompt]     Run a model locally
  serve [options]          Start the OpenAI-compatible server
  list                     List local GGUF models in ./models""",
        file=file,
    )
