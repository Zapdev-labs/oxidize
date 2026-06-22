# oxidize-cpp / modal / benchmark.py
#
# Usage:
#   modal run oxidize-cpp/modal/benchmark.py --model <gguf-on-volume>
#
# <gguf-on-volume> is a path RELATIVE to the root of the `oxidize-models` Modal
# volume (or an absolute path under the mount point /models). Upload your GGUF
# first, e.g.:
#   modal volume put oxidize-models ./Qwen2.5-0.5B-Q4_K_M.gguf Qwen2.5-0.5B-Q4_K_M.gguf
#   modal run oxidize-cpp/modal/benchmark.py --model Qwen2.5-0.5B-Q4_K_M.gguf
#
# What it does
# ------------
# For each GPU (H100, A100) it runs the SAME model + prompt + max-tokens through:
#   (a) the C++ engine:  oxidize-cpp --model <p> --prompt <q> --max-tokens N --json --cuda
#   (b) the Rust engine: oxidize run <p> --prompt <q> --max-tokens N --backend cuda --no-api
# parses tokens/sec from each, and returns a dict. The local entrypoint
# aggregates all four (engine x gpu) results, prints a comparison table, and
# writes a results JSON next to this file.
#
# The Rust CLI mirrors how the repo's `oxidize-cli` binary (binary name `oxidize`)
# is invoked: `oxidize run <model> ...` rewrites to a one-shot `--prompt` run and
# exits after printing `generation stats: tokens=N speed=X.XX tok/s`.

import json
import re
import subprocess
import time

import modal

from image import (
    CPP_BIN,
    MODELS_MOUNT,
    RUST_BIN,
    image,
    models_volume,
)

APP_NAME = "oxidize-cpp-bench"
app = modal.App(APP_NAME)

DEFAULT_PROMPT = "Explain the theory of relativity in one paragraph."
DEFAULT_MAX_TOKENS = 128

# Rust one-shot stats line: "generation stats: tokens=128 speed=42.31 tok/s"
_RUST_STATS_RE = re.compile(
    r"generation stats:\s*tokens=(\d+)\s+speed=([0-9]+(?:\.[0-9]+)?)\s*tok/s",
    re.IGNORECASE,
)
# Generic fallback "<float> tok/s" anywhere in output.
_GENERIC_TOKS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*tok/s", re.IGNORECASE)


def _resolve_model_path(model: str) -> str:
    """Accept either a path relative to the volume root or an absolute /models path."""
    if model.startswith("/"):
        return model
    return f"{MODELS_MOUNT}/{model.lstrip('/')}"


def _run(cmd: list[str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, wall


def _bench_cpp(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run the C++ engine with --json --cuda and parse its JSON report."""
    cmd = [
        CPP_BIN,
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--json",
        "--cuda",
    ]
    rc, out, err, wall = _run(cmd)
    result = {
        "engine": "cpp",
        "ok": rc == 0,
        "returncode": rc,
        "wall_seconds": round(wall, 4),
        "tokens": None,
        "tokens_per_sec": None,
        "raw": (out or "")[-4000:],
        "stderr": (err or "")[-2000:],
    }
    if rc != 0:
        return result

    # Preferred: the binary emits a JSON object (possibly on the last line).
    parsed = None
    for line in reversed([l for l in out.splitlines() if l.strip()]):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None

    if isinstance(parsed, dict):
        tps = (
            parsed.get("tokens_per_sec")
            or parsed.get("tokens_per_second")
            or parsed.get("tok_per_sec")
            or parsed.get("speed")
        )
        toks = parsed.get("tokens") or parsed.get("generated_tokens")
        if tps is not None:
            result["tokens_per_sec"] = float(tps)
        if toks is not None:
            result["tokens"] = int(toks)

    # Fallback: scrape "<x> tok/s" from text output.
    if result["tokens_per_sec"] is None:
        m = _GENERIC_TOKS_RE.search(out)
        if m:
            result["tokens_per_sec"] = float(m.group(1))

    return result


def _bench_rust(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run the Rust 'oxidize run' one-shot path and parse the stats line."""
    cmd = [
        RUST_BIN, "run", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--backend", "cuda",
        "--no-api",
    ]
    rc, out, err, wall = _run(cmd)
    combined = f"{out}\n{err}"
    result = {
        "engine": "rust",
        "ok": rc == 0,
        "returncode": rc,
        "wall_seconds": round(wall, 4),
        "tokens": None,
        "tokens_per_sec": None,
        "raw": combined[-4000:],
        "stderr": (err or "")[-2000:],
    }
    if rc != 0:
        return result

    m = _RUST_STATS_RE.search(combined)
    if m:
        result["tokens"] = int(m.group(1))
        result["tokens_per_sec"] = float(m.group(2))
    else:
        g = _GENERIC_TOKS_RE.search(combined)
        if g:
            result["tokens_per_sec"] = float(g.group(1))
    return result


def _gpu_info() -> dict:
    try:
        rc, out, _, _ = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"]
        )
        if rc == 0 and out.strip():
            name, mem, drv = (x.strip() for x in out.strip().splitlines()[0].split(","))
            return {"name": name, "memory_total": mem, "driver": drv}
    except Exception:  # pragma: no cover - best-effort metadata
        pass
    return {"name": "unknown", "memory_total": "?", "driver": "?"}


def _run_both(gpu_label: str, model: str, prompt: str, max_tokens: int) -> dict:
    models_volume.reload()
    model_path = _resolve_model_path(model)
    return {
        "gpu": gpu_label,
        "gpu_info": _gpu_info(),
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "cpp": _bench_cpp(model_path, prompt, max_tokens),
        "rust": _bench_rust(model_path, prompt, max_tokens),
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={MODELS_MOUNT: models_volume},
    timeout=60 * 30,
)
def bench_h100(model: str, prompt: str = DEFAULT_PROMPT,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _run_both("H100", model, prompt, max_tokens)


@app.function(
    image=image,
    gpu="A100",
    volumes={MODELS_MOUNT: models_volume},
    timeout=60 * 30,
)
def bench_a100(model: str, prompt: str = DEFAULT_PROMPT,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _run_both("A100", model, prompt, max_tokens)


def _fmt_tps(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("oxidize C++ vs Rust  -  CUDA throughput (tok/s)")
    print("=" * 72)
    header = f"{'GPU':<8}{'Engine':<8}{'tok/s':>12}{'tokens':>10}{'wall(s)':>12}{'ok':>5}"
    print(header)
    print("-" * 72)
    for r in results:
        for eng in ("cpp", "rust"):
            e = r[eng]
            print(
                f"{r['gpu']:<8}{eng:<8}"
                f"{_fmt_tps(e['tokens_per_sec']):>12}"
                f"{str(e['tokens']) if e['tokens'] is not None else 'n/a':>10}"
                f"{e['wall_seconds']:>12.2f}"
                f"{('Y' if e['ok'] else 'N'):>5}"
            )
    print("-" * 72)
    # Speedup summary (cpp relative to rust) per GPU.
    print(f"{'GPU':<8}{'cpp/rust speedup':>24}")
    for r in results:
        c = r["cpp"]["tokens_per_sec"]
        ru = r["rust"]["tokens_per_sec"]
        if isinstance(c, (int, float)) and isinstance(ru, (int, float)) and ru > 0:
            print(f"{r['gpu']:<8}{(c / ru):>23.2f}x")
        else:
            print(f"{r['gpu']:<8}{'n/a':>24}")
    print("=" * 72 + "\n")


@app.local_entrypoint()
def main(model: str, prompt: str = DEFAULT_PROMPT,
         max_tokens: int = DEFAULT_MAX_TOKENS, out: str = "bench_results.json"):
    """Aggregate H100 + A100 results for both engines.

    --model      GGUF path relative to the `oxidize-models` volume (or /models/...).
    --prompt     shared prompt (default benchmark prompt).
    --max-tokens generation length for both engines.
    --out        results JSON path written locally.
    """
    print(f"[oxidize-bench] model={model} max_tokens={max_tokens}")
    print(f"[oxidize-bench] prompt={prompt!r}")

    # Run both GPUs concurrently.
    h100_call = bench_h100.spawn(model, prompt, max_tokens)
    a100_call = bench_a100.spawn(model, prompt, max_tokens)

    results = []
    for label, call in (("H100", h100_call), ("A100", a100_call)):
        try:
            results.append(call.get())
        except Exception as exc:  # surface per-GPU failures without aborting the run
            print(f"[oxidize-bench] {label} run failed: {exc}")
            results.append({
                "gpu": label,
                "gpu_info": {},
                "model": _resolve_model_path(model),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "cpp": {"engine": "cpp", "ok": False, "tokens": None,
                        "tokens_per_sec": None, "wall_seconds": 0.0,
                        "error": str(exc)},
                "rust": {"engine": "rust", "ok": False, "tokens": None,
                         "tokens_per_sec": None, "wall_seconds": 0.0,
                         "error": str(exc)},
            })

    _print_table(results)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[oxidize-bench] wrote {out}")
    return payload
