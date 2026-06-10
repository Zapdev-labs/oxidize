"""CLI mirroring oxidize-cli / oxidize-golang/internal/cli."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from oxidize_python.cli_flags import COMMON_RUN_HELP, add_run_flags, options_from_namespace
from oxidize_python.internal.generate import cli_transcript


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        return _run_legacy(args)
    cmd = args[0]
    rest = args[1:]
    handlers = {
        "run": _run_command,
        "chat": _chat_command,
        "bench": _bench_command,
        "inspect": _inspect_command,
        "list": _list_command,
        "ls": _list_command,
        "serve": _serve_command,
        "gpu-cluster": _gpu_cluster_command,
        "help": _help_command,
        "-h": _help_command,
        "--help": _help_command,
    }
    handler = handlers.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd}", file=sys.stderr)
        _print_help(file=sys.stderr)
        return 1
    return handler(rest)


def _parse_run(ns: argparse.Namespace, positional: list[str]) -> tuple[str, str]:
    opts, positional = options_from_namespace(ns, positional)
    if not positional:
        raise SystemExit("requires a model name or local .gguf path")
    return positional[0], opts.prompt


def _resolve_model_path(name_or_path: str, hf_file: str = "") -> str:
    if name_or_path.lower().endswith(".gguf"):
        return name_or_path
    if Path(name_or_path).exists():
        return name_or_path
    if "/" in name_or_path:
        from oxidize_python.hf.hub import ResolveOptions, resolve_gguf

        return resolve_gguf(ResolveOptions(repo=name_or_path, filename=hf_file))
    models_dir = _resolve_models_dir("")
    if models_dir:
        candidate = Path(models_dir) / f"{name_or_path}.gguf"
        if candidate.is_file():
            return str(candidate)
    return name_or_path


def _run_legacy(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize", add_help=False)
    add_run_flags(p)
    p.add_argument("--model", default="")
    ns, _ = p.parse_known_args(args)
    try:
        opts, _ = options_from_namespace(ns, [])
    except SystemExit as err:
        print(err, file=sys.stderr)
        return 1
    prompt = opts.prompt
    if not prompt:
        return 0
    model = str(getattr(ns, "model", "") or "").strip()
    if model:
        try:
            path = _resolve_model_path(model, opts.hf_file)
            if path.lower().endswith(".gguf") and Path(path).is_file():
                return _run_gguf(opts.run_config(path))
        except Exception as err:
            print(f"generation failed: {err}", file=sys.stderr)
            return 1
    sys.stdout.write(cli_transcript(prompt))
    return 0


def _run_gguf(cfg, *, profile: bool = False) -> int:
    from oxidize_python.internal.generate.runtime import run_from_gguf

    try:
        run_from_gguf(cfg, sys.stdout)
        if profile:
            print("# profile: generation complete (see stats line above)", file=sys.stderr)
        return 0
    except Exception as err:
        print(f"generation failed: {err}", file=sys.stderr)
        return 1


def _run_command(args: list[str]) -> int:
    if args and args[0] in ("-h", "--help"):
        _print_run_help()
        return 0
    p = argparse.ArgumentParser(prog="oxidize run", add_help=False)
    add_run_flags(p)
    ns, positional = p.parse_known_args(args)
    try:
        opts, positional = options_from_namespace(ns, positional)
        model_arg, prompt = _parse_run(ns, positional)
        opts.prompt = prompt
    except SystemExit as err:
        print(f"oxidize run {err}", file=sys.stderr)
        return 1
    if not opts.prompt:
        return 0
    try:
        path = _resolve_model_path(model_arg, opts.hf_file)
    except Exception as err:
        print(f"resolve model: {err}", file=sys.stderr)
        return 1
    from oxidize_python.cli_extras import maybe_run_mesh_chat, maybe_run_pipeline

    if maybe_run_pipeline(opts, path, sys.stdout):
        return 0
    if maybe_run_mesh_chat(opts, path, sys.stdout, sys.stderr):
        return 0
    if path.lower().endswith(".gguf") and Path(path).is_file():
        return _run_gguf(opts.run_config(path), profile=opts.profile)
    sys.stdout.write(cli_transcript(opts.prompt))
    return 0


def _chat_command(args: list[str]) -> int:
    if args and args[0] in ("-h", "--help"):
        _print_chat_help()
        return 0
    p = argparse.ArgumentParser(prog="oxidize chat", add_help=False)
    add_run_flags(p)
    ns, positional = p.parse_known_args(args)
    try:
        opts, positional = options_from_namespace(ns, positional)
    except SystemExit as err:
        print(f"oxidize chat {err}", file=sys.stderr)
        return 1
    if not positional:
        print("oxidize chat requires a model", file=sys.stderr)
        return 1
    try:
        path = _resolve_model_path(positional[0], opts.hf_file)
    except Exception as err:
        print(f"resolve model: {err}", file=sys.stderr)
        return 1
    from oxidize_python.cli_extras import maybe_run_mesh_chat

    if maybe_run_mesh_chat(opts, path, sys.stdout, sys.stderr):
        return 0
    cfg = opts.run_config(path)
    print("oxidize chat mode. type 'exit' or 'quit' to leave.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            return 0
        cfg.prompt = line
        code = _run_gguf(cfg, profile=opts.profile)
        if code != 0:
            return code
        print()


def _bench_command(args: list[str]) -> int:
    if args and args[0] in ("-h", "--help"):
        print("Usage: oxidize bench <model> [--iterations N] [--max-tokens N] [--prompt TEXT]")
        return 0
    p = argparse.ArgumentParser(prog="oxidize bench", add_help=False)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt", default="benchmark")
    p.add_argument("--file", default="", dest="hf_file")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--ctx-size", type=int, default=0)
    ns, positional = p.parse_known_args(args)
    if not positional:
        print("oxidize bench requires a model", file=sys.stderr)
        return 1
    try:
        path = _resolve_model_path(positional[0], ns.hf_file)
    except Exception as err:
        print(f"resolve model: {err}", file=sys.stderr)
        return 1
    import time

    from oxidize_python.core.ffi import RustModel, _ensure_loaded

    print(
        f"=== Oxidize bench ===\nmodel: {path}\n"
        f"iterations: {ns.iterations} max_tokens: {ns.max_tokens}\n"
    )

    # Fast path: use Rust FFI model (same kernels as the Rust binary)
    if _ensure_loaded():
        try:
            model = RustModel(path)
            prompt_tokens = [1]  # BOS — good enough for throughput bench
            total_tokens = 0
            total_seconds = 0.0
            for round_i in range(1, ns.iterations + 1):
                model.reset_session()
                start = time.monotonic()
                model.forward(prompt_tokens)
                tok = model.sample_argmax()
                # Prefill already produced the first generated token.
                generated = 1
                for _ in range(max(ns.max_tokens - 1, 0)):
                    model.forward([tok])
                    tok = model.sample_argmax()
                    generated += 1
                elapsed = time.monotonic() - start
                speed = generated / elapsed if elapsed > 0 else 0.0
                total_tokens += generated
                total_seconds += elapsed
                print(
                    f"round {round_i}: tokens={generated} "
                    f"elapsed={elapsed:.3f}s speed={speed:.2f} tok/s"
                )
            avg = total_tokens / total_seconds if total_seconds > 0 else 0.0
            print(f"\naverage: {avg:.2f} tok/s over {total_tokens} tokens")
            return 0
        except Exception as e:
            print(f"Rust FFI failed ({e}), falling back to pure-Python model...", file=sys.stderr)

    # Fallback: pure-Python model
    from oxidize_python.core.model.generation import GenerationStream, default_generation_config
    from oxidize_python.core.model.inference import InferenceModel
    from oxidize_python.core.model.loader import LoaderConfig, load_gguf_model_from_path
    from oxidize_python.core.model.model import Session

    loader = LoaderConfig(threads=ns.threads, ctx_size=ns.ctx_size)
    loaded = load_gguf_model_from_path(path, loader)
    if not isinstance(loaded, InferenceModel) or loaded.stack is None or not loaded.stack.loaded():
        print(f"bench: model {path!r} has no loadable weights", file=sys.stderr)
        return 1
    prompt_tokens = [1]
    total_tokens = 0
    total_seconds = 0.0
    for round_i in range(1, ns.iterations + 1):
        session = Session()
        gen_cfg = default_generation_config()
        gen_cfg.max_new_tokens = ns.max_tokens
        gen_cfg.sampling.temperature = 0.0
        stream = GenerationStream(loaded, session, gen_cfg)
        stream.seed(prompt_tokens)
        start = time.monotonic()
        generated = 0
        for _ in range(ns.max_tokens):
            _, done, err = stream.next()
            if err:
                print(f"bench failed: {err}", file=sys.stderr)
                return 1
            generated += 1
            if done:
                break
        elapsed = time.monotonic() - start
        tokens = session.consumed_tokens() or generated
        speed = tokens / elapsed if elapsed > 0 else 0.0
        total_tokens += tokens
        total_seconds += elapsed
        print(f"round {round_i}: tokens={tokens} elapsed={elapsed:.3f}s speed={speed:.2f} tok/s")
    avg = total_tokens / total_seconds if total_seconds > 0 else 0.0
    print(f"\naverage: {avg:.2f} tok/s over {total_tokens} tokens")
    return 0


def _inspect_command(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print("Usage: oxidize inspect <model.gguf>")
        return 0 if args else 1
    try:
        path = _resolve_model_path(args[0])
    except Exception as err:
        print(f"resolve model: {err}", file=sys.stderr)
        return 1
    from oxidize_python.core.quantization.types import from_ggml_type
    from oxidize_python.internal.gguf.parse import load_file
    from oxidize_python.internal.gguf.tensor_size import tensor_byte_size

    try:
        file = load_file(path)
    except Exception as err:
        print(f"inspect: {err}", file=sys.stderr)
        return 1
    print(f"Metadata in {path}:")
    for key in sorted(file.metadata):
        val = file.metadata[key]
        text = val.string if val.string else repr(val)
        print(f"  {key} = {text}")
    print(f"\nTensors in {path}:")
    for tensor in file.tensor_infos:
        count = 1
        for d in tensor.dimensions:
            count *= int(d)
        qtype = from_ggml_type(tensor.ggml_type)
        try:
            qsize = tensor_byte_size(tensor.ggml_type, count)
        except Exception:
            qsize = 0
        print(
            f"  {tensor.name} dims={tensor.dimensions} type={qtype} "
            f"offset={tensor.relative_offset} qsize={qsize}"
        )
    size = Path(path).stat().st_size
    print(f"\nfile_size={size} bytes")
    return 0


def _list_command(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="oxidize list", add_help=False)
    p.add_argument("--models-dir", default="")
    ns, _ = p.parse_known_args(args)
    models_dir = _resolve_models_dir(ns.models_dir)
    from oxidize_python.internal.serviceinfo.models import discover_models

    sys.stdout.write(f"{'NAME':<48} {'SIZE':>9} PATH\n")
    if not models_dir:
        return 0
    for model in discover_models(models_dir):
        size = Path(model.path).stat().st_size / (1024**3)
        name = model.id or model.path
        sys.stdout.write(f"{name:<48} {size:>8.2f}G {model.path}\n")
    return 0


def _serve_command(args: list[str]) -> int:
    if args and args[0] in ("-h", "--help"):
        _print_serve_help()
        return 0
    p = argparse.ArgumentParser(prog="oxidize serve", add_help=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--models-dir", default="")
    ns = p.parse_args(args)
    from oxidize_python.internal.server import listen

    with contextlib.suppress(KeyboardInterrupt):
        listen(host=ns.host, port=ns.port, models_dir=_resolve_models_dir(ns.models_dir))
    return 0


def _resolve_models_dir(raw: str) -> str:
    if raw.strip():
        return raw
    candidate = Path.cwd() / "models"
    return str(candidate) if candidate.is_dir() else ""


def _gpu_cluster_command(args: list[str]) -> int:
    from oxidize_python import gpucluster as gc

    sub = args[0] if args else "help"
    if sub == "profiles":
        for p in gc.all_profiles():
            print(
                f"{p.family.slug:<14} product={p.product:<26} arch={p.generation:<9} "
                f"mem={p.memory_mib}MiB tdp={p.tdp_watts}W nvlink={p.nvlink} "
                f"mig={p.mig_capable} timeslice={p.time_slice_replicas} net={p.network_class}"
            )
        return 0
    if sub == "detect":
        gpus = gc.detect_gpus()
        if not gpus:
            print("no NVIDIA GPUs detected (nvidia-smi unavailable or no devices)")
            return 0
        for g in gpus:
            fam = g.family.slug if g.family else "unknown"
            print(
                f"GPU {g.index}: {g.name} ({g.memory_total_mib}MiB) "
                f"family={fam} mig={g.mig_enabled}"
            )
        print("--- summary ---")
        for fam, n in gc.summarize(gpus):
            print(f"{fam.slug}: {n}")
        return 0
    if sub == "generate":
        if args[1:] and args[1] in ("-h", "--help"):
            print(
                "Usage: oxidize gpu-cluster generate [options]\n\n"
                "Options:\n"
                "  --family b200|a100|rtx-pro-6000   GPU family (default: all)\n"
                "  --nodes N                         Number of nodes (default: per-family preset)\n"
                "  --gpus-per-node N                 GPUs per node (default: per-family preset)"
            )
            return 0
        p = argparse.ArgumentParser(prog="oxidize gpu-cluster generate", add_help=False)
        p.add_argument("--family", default="")
        p.add_argument("--nodes", type=int, default=0)
        p.add_argument("--gpus-per-node", type=int, default=0)
        ns = p.parse_args(args[1:])

        if ns.nodes < 0 or ns.gpus_per_node < 0:
            print("error: --nodes and --gpus-per-node must be positive integers", file=sys.stderr)
            return 2

        if ns.family:
            fam = gc.GpuFamily.from_slug(ns.family)
            if fam is None:
                print("error: --family expects b200|a100|rtx-pro-6000", file=sys.stderr)
                return 2
            defaults_nodes = {gc.GpuFamily.B200: 8, gc.GpuFamily.A100: 16, gc.GpuFamily.RTX_PRO_6000: 4}
            defaults_gpn = {gc.GpuFamily.B200: 8, gc.GpuFamily.A100: 8, gc.GpuFamily.RTX_PRO_6000: 2}
            count = ns.nodes or defaults_nodes[fam]
            gpn = ns.gpus_per_node or defaults_gpn[fam]
            specs = [gc.NodePoolSpec(fam, count, gpn)]
        else:
            specs = [
                gc.NodePoolSpec(gc.GpuFamily.B200, 8, 8),
                gc.NodePoolSpec(gc.GpuFamily.A100, 16, 8),
                gc.NodePoolSpec(gc.GpuFamily.RTX_PRO_6000, 4, 2),
            ]
        families = [s.family for s in specs]

        print(gc.node_pools_yaml(specs), end="")
        print("---")
        print(gc.device_plugin_config_yaml(families), end="")
        for f in families:
            mig = gc.mig_config_yaml(f)
            if mig is not None:
                print("---")
                print(mig, end="")
        print("---")
        print(gc.prometheus_rules_yaml(), end="")
        for f in families:
            print("---")
            print(gc.helm_values_yaml(f), end="")
        return 0

    print(
        "usage: oxidize gpu-cluster <generate|detect|profiles>\n\n"
        "generate [--family b200|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]\n"
        "detect   probe local NVIDIA GPUs via nvidia-smi\n"
        "profiles list known GPU tier profiles"
    )
    return 0


def _help_command(_args: list[str]) -> int:
    _print_help()
    return 0


def _print_help(file: object | None = None) -> None:
    out = sys.stdout if file is None else file
    print(
        """Usage: oxidize <command> [args]

Commands:
  run <model> [prompt]     Run a model locally
  chat <model>             Interactive chat REPL
  bench <model>            Decode throughput benchmark
  inspect <model.gguf>     Print GGUF metadata and tensors
  serve [options]          Start the OpenAI-compatible server
  gpu-cluster <sub>        GPU cluster helpers (generate, detect, profiles)
  list                     List local GGUF models in ./models""",
        file=out,
    )


def _print_run_help() -> None:
    print(
        f"""Usage: oxidize run <model> [prompt] [options]

Models can be local .gguf files or Hugging Face GGUF repos.

{COMMON_RUN_HELP}"""
    )


def _print_chat_help() -> None:
    print(
        f"""Usage: oxidize chat <model> [options]

Interactive REPL for local GGUF models.

{COMMON_RUN_HELP}"""
    )


def _print_serve_help() -> None:
    print(
        """Usage: oxidize serve [options]

Starts the OpenAI-compatible API server.

Common options: --host, --port, --models-dir"""
    )


if __name__ == "__main__":
    raise SystemExit(main())
