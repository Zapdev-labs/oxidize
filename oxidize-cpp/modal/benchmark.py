# oxidize-cpp / modal / benchmark.py
#
# Credit-safe H100/A100 test of the C++/CUDA engine.
#
# Usage:
#   modal run oxidize-cpp/modal/benchmark.py
#   modal run oxidize-cpp/modal/benchmark.py --max-tokens 128
#
# What it does, in order, per GPU (H100 then A100):
#   1. CORRECTNESS GATE (cheap): run the C++ engine on a fixed F16 model with
#      fixed pre-tokenized ids on BOTH --cpu and --cuda. The CUDA greedy token
#      sequence MUST equal the CPU sequence AND a known-good golden sequence
#      (verified locally vs llama.cpp). If it does not, we ABORT before spending
#      any real GPU time on benchmarking a broken kernel.
#   2. BENCHMARK: only if the gate passes, measure decode tok/s for C++ --cuda
#      vs C++ --cpu (and a best-effort Rust baseline), using the SAME fixed ids
#      so there is no tokenizer dependence.
#
# The C++ CLI has no built-in tokenizer yet, so we drive it with --tokens
# (pre-tokenized ids) rather than --prompt. This makes the run fully
# deterministic and apples-to-apples across engines/devices.

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

DEFAULT_MAX_TOKENS = 64

# Benchmark model: small dense F16 (no quantization) so the correctness gate is
# an EXACT token match (quantized models legitimately diverge on near-ties).
GATE_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GATE_FILE = "qwen2.5-0.5b-instruct-fp16.gguf"
# "The capital of France is" under the Qwen tokenizer (no BOS).
GATE_PROMPT_IDS = "785,6722,315,9625,374"
# Greedy continuation verified locally to match llama.cpp token-for-token.
GATE_GOLDEN = "12095 13 1084 374 279 7772 3283 304 4505 323 279 2086"
GATE_GOLDEN_LEN = 12

_RUST_STATS_RE = re.compile(
    r"generation stats:\s*tokens=(\d+)\s+speed=([0-9]+(?:\.[0-9]+)?)\s*tok/s",
    re.IGNORECASE,
)
_GEN_TOKENS_RE = re.compile(r"gen_tokens:\s*([0-9 ]+)")


def _run(cmd: list[str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start


def _gpu_info() -> dict:
    rc, out, _, _ = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"]
    )
    if rc == 0 and out.strip():
        parts = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        if len(parts) == 3:
            return {"name": parts[0], "memory_total": parts[1], "driver": parts[2]}
    return {"name": "unknown", "memory_total": "?", "driver": "?"}


def _cpp_gen_tokens(model_path: str, ids: str, max_tokens: int, cuda: bool) -> tuple[str, str]:
    """Return (gen_tokens_string, raw_stderr) from a non-JSON C++ run."""
    cmd = [CPP_BIN, "--model", model_path, "--tokens", ids,
           "--max-tokens", str(max_tokens)]
    if cuda:
        cmd.append("--cuda")
    rc, out, err, _ = _run(cmd)
    if rc != 0:
        return "", err[-2000:]
    m = _GEN_TOKENS_RE.search(out)
    return (m.group(1).strip() if m else ""), err[-1000:]


def _cpp_bench(model_path: str, ids: str, max_tokens: int, cuda: bool) -> dict:
    """Run with --json and parse decode/prefill tok/s."""
    cmd = [CPP_BIN, "--model", model_path, "--tokens", ids,
           "--max-tokens", str(max_tokens), "--json"]
    if cuda:
        cmd.append("--cuda")
    rc, out, err, wall = _run(cmd)
    res = {"engine": "cpp", "device": "cuda" if cuda else "cpu", "ok": rc == 0,
           "wall_seconds": round(wall, 4), "decode_tps": None, "prefill_tps": None,
           "stderr": err[-1500:]}
    if rc != 0:
        return res
    for line in reversed([l.strip() for l in out.splitlines() if l.strip()]):
        if line.startswith("{") and line.endswith("}"):
            try:
                p = json.loads(line)
                res["decode_tps"] = p.get("decode_tps")
                res["prefill_tps"] = p.get("prefill_tps")
                res["device"] = p.get("device", res["device"])
                break
            except json.JSONDecodeError:
                continue
    return res


def _rust_bench(model_path: str, max_tokens: int) -> dict:
    """Best-effort Rust CUDA baseline. Never fatal."""
    import os
    res = {"engine": "rust", "ok": False, "decode_tps": None, "note": ""}
    if not os.path.exists(RUST_BIN):
        res["note"] = "rust binary not built (non-fatal)"
        return res
    cmd = [RUST_BIN, "run", model_path, "--prompt", "The capital of France is",
           "--max-tokens", str(max_tokens), "--no-api"]
    rc, out, err, wall = _run(cmd)
    combined = f"{out}\n{err}"
    res["ok"] = rc == 0
    res["wall_seconds"] = round(wall, 4)
    m = _RUST_STATS_RE.search(combined)
    if m:
        res["tokens"] = int(m.group(1))
        res["decode_tps"] = float(m.group(2))
    else:
        res["note"] = "could not parse rust tok/s"
    res["stderr"] = err[-1500:]
    return res


def _gpu_run(gpu_label: str, max_tokens: int) -> dict:
    models_volume.reload()
    model_path = f"{MODELS_MOUNT}/{GATE_FILE}"
    info = _gpu_info()
    out = {"gpu": gpu_label, "gpu_info": info, "model": GATE_FILE,
           "gate": {}, "cpp_cuda": None, "cpp_cpu": None, "rust": None}

    # ---- 1. CORRECTNESS GATE (cheap: a few tokens) ----
    cpu_toks, cpu_err = _cpp_gen_tokens(model_path, GATE_PROMPT_IDS, GATE_GOLDEN_LEN, cuda=False)
    cuda_toks, cuda_err = _cpp_gen_tokens(model_path, GATE_PROMPT_IDS, GATE_GOLDEN_LEN, cuda=True)
    gate = {
        "golden": GATE_GOLDEN,
        "cpu_tokens": cpu_toks,
        "cuda_tokens": cuda_toks,
        "cpu_matches_golden": cpu_toks == GATE_GOLDEN,
        "cuda_matches_cpu": (cuda_toks == cpu_toks and cuda_toks != ""),
        "cuda_matches_golden": cuda_toks == GATE_GOLDEN,
        "cpu_stderr": cpu_err,
        "cuda_stderr": cuda_err,
    }
    gate["passed"] = gate["cuda_matches_cpu"] and gate["cuda_matches_golden"]
    out["gate"] = gate
    if not gate["passed"]:
        out["aborted"] = "correctness gate failed; skipped benchmark to save GPU time"
        return out

    # ---- 2. BENCHMARK (gate passed) ----
    out["cpp_cuda"] = _cpp_bench(model_path, GATE_PROMPT_IDS, max_tokens, cuda=True)
    out["cpp_cpu"] = _cpp_bench(model_path, GATE_PROMPT_IDS, max_tokens, cuda=False)
    out["rust"] = _rust_bench(model_path, max_tokens)
    return out


# ---- model provisioning (CPU; no GPU cost) --------------------------------
@app.function(image=image, volumes={MODELS_MOUNT: models_volume}, timeout=60 * 20)
def ensure_model() -> str:
    import os
    from huggingface_hub import hf_hub_download
    dst = f"{MODELS_MOUNT}/{GATE_FILE}"
    if os.path.exists(dst):
        return f"present: {dst}"
    p = hf_hub_download(repo_id=GATE_REPO, filename=GATE_FILE, local_dir="/tmp/dl")
    os.makedirs(MODELS_MOUNT, exist_ok=True)
    import shutil
    shutil.copy(p, dst)
    models_volume.commit()
    return f"downloaded: {dst} ({os.path.getsize(dst)} bytes)"


@app.function(image=image, gpu="H100", volumes={MODELS_MOUNT: models_volume}, timeout=60 * 20)
def bench_h100(max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _gpu_run("H100", max_tokens)


@app.function(image=image, gpu="A100", volumes={MODELS_MOUNT: models_volume}, timeout=60 * 20)
def bench_a100(max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _gpu_run("A100", max_tokens)


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def _print_report(results: list[dict]) -> None:
    print("\n" + "=" * 74)
    print("oxidize-cpp  -  C++/CUDA correctness + throughput on H100/A100")
    print("=" * 74)
    for r in results:
        g = r.get("gate", {})
        gi = r.get("gpu_info", {})
        print(f"\n[{r['gpu']}]  {gi.get('name','?')}  ({gi.get('memory_total','?')})")
        print(f"  correctness gate: {'PASS' if g.get('passed') else 'FAIL'}"
              f"  (cuda==cpu: {g.get('cuda_matches_cpu')}, "
              f"cuda==golden: {g.get('cuda_matches_golden')})")
        if not g.get("passed"):
            print(f"    golden: {g.get('golden')}")
            print(f"    cpu:    {g.get('cpu_tokens')}")
            print(f"    cuda:   {g.get('cuda_tokens')}")
            if g.get("cuda_stderr"):
                print(f"    cuda stderr: {g['cuda_stderr'][-400:]}")
            continue
        cc, cp, ru = r.get("cpp_cuda") or {}, r.get("cpp_cpu") or {}, r.get("rust") or {}
        print(f"  C++ CUDA  decode: {_fmt(cc.get('decode_tps')):>8} tok/s   "
              f"prefill: {_fmt(cc.get('prefill_tps')):>8} tok/s")
        print(f"  C++ CPU   decode: {_fmt(cp.get('decode_tps')):>8} tok/s")
        if isinstance(cc.get("decode_tps"), (int, float)) and \
           isinstance(cp.get("decode_tps"), (int, float)) and cp["decode_tps"] > 0:
            print(f"  -> C++ CUDA/CPU speedup: {cc['decode_tps']/cp['decode_tps']:.2f}x")
        if ru.get("ok") and isinstance(ru.get("decode_tps"), (int, float)):
            print(f"  Rust      decode: {_fmt(ru.get('decode_tps')):>8} tok/s  (baseline)")
        else:
            print(f"  Rust baseline: {ru.get('note') or 'unavailable'}")
    # H100 vs A100 for C++ CUDA.
    by = {r["gpu"]: r for r in results if (r.get("gate") or {}).get("passed")}
    if "H100" in by and "A100" in by:
        h = (by["H100"].get("cpp_cuda") or {}).get("decode_tps")
        a = (by["A100"].get("cpp_cuda") or {}).get("decode_tps")
        if isinstance(h, (int, float)) and isinstance(a, (int, float)) and a > 0:
            print(f"\n  H100 vs A100 (C++ CUDA decode): {h/a:.2f}x  ({_fmt(h)} vs {_fmt(a)} tok/s)")
    print("=" * 74 + "\n")


@app.local_entrypoint()
def main(max_tokens: int = DEFAULT_MAX_TOKENS, out: str = "bench_results.json"):
    print(f"[oxidize-bench] ensuring model {GATE_FILE} on volume ...")
    print("  " + ensure_model.remote())

    print("[oxidize-bench] launching H100 + A100 (gate first, then benchmark) ...")
    h = bench_h100.spawn(max_tokens)
    a = bench_a100.spawn(max_tokens)
    results = []
    for label, call in (("H100", h), ("A100", a)):
        try:
            results.append(call.get())
        except Exception as exc:
            print(f"[oxidize-bench] {label} failed: {exc}")
            results.append({"gpu": label, "gate": {"passed": False}, "error": str(exc)})

    _print_report(results)
    with open(out, "w") as f:
        json.dump({"max_tokens": max_tokens, "results": results}, f, indent=2)
    print(f"[oxidize-bench] wrote {out}")
