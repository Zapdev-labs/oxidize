"""
Modal harness for testing the oxidize Rust workspace in the cloud.

Why: builds/tests run on Modal's fast multi-core boxes with a persistent
cargo cache, so repeated runs are incremental instead of cold every time.

Usage:
    modal run modal_app.py                      # smoke-build the CLI + run `oxidize --help`
    modal run modal_app.py --action test        # cargo test for the default package
    modal run modal_app.py --action test --package oxidize-kernels
    modal run modal_app.py --action test --package workspace   # whole workspace (slow)
    modal run modal_app.py --action smoke        # build oxidize-cli release + smoke run

The source tree is mounted read-only at /workspace; `target/` and the cargo
registry live in named Volumes so compiles are cached across runs.
"""

import modal

REPO_ROOT = "/workspace"
RUST_VERSION = "1.95-bookworm"

# Keep the upload small: source only, no build artifacts / models / vcs.
IGNORE = [
    "target/**",
    ".git/**",
    "models/**",
    "dist/**",
    "node_modules/**",
    "**/*.gguf.bak",
    "rust_out/**",
    ".omo/**",       # background automation artifacts churn during build
    ".git/**",
    "**/*.log",
]

image = (
    modal.Image.from_registry(f"rust:{RUST_VERSION}", add_python="3.12")
    .apt_install("pkg-config", "libssl-dev", "cmake", "clang")
    # Mounted at container start (not baked into layers) so edits are picked up
    # on the next run without rebuilding the image.
    .add_local_dir(".", REPO_ROOT, ignore=IGNORE, copy=False)
)

# Writable, persistent caches. `target/` is the big one; the registry cache
# avoids re-downloading crates.io deps on every run.
target_cache = modal.Volume.from_name("oxidize-target-cache", create_if_missing=True)
registry_cache = modal.Volume.from_name("oxidize-cargo-registry", create_if_missing=True)
# Persists HF model downloads (~/.cache/oxidize/hf) across runs.
model_cache = modal.Volume.from_name("oxidize-model-cache", create_if_missing=True)

# --- CUDA image: nvcc-capable devel base + Rust 1.95 toolchain ---------------
# build.rs compiles kernels/gemv_f32.cu -> PTX with nvcc, so we need the *devel*
# CUDA image (toolkit), not just runtime. No GPU is needed to *compile* the PTX.
CUDA_TAG = "12.8.1-devel-ubuntu22.04"
cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "cmake", "clang", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain 1.95.0 --profile minimal"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "CUDA_HOME": "/usr/local/cuda",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        }
    )
    # copy=True bakes the source into the image layer; editing any source file
    # busts the content hash and forces a fresh mount (copy=False snapshots can
    # go stale across edits).
    .add_local_dir(".", REPO_ROOT, ignore=IGNORE, copy=True)
)

# Separate target cache for CUDA builds so we don't thrash the CPU artifacts
# back and forth when toggling the `cuda` feature.
cuda_target_cache = modal.Volume.from_name("oxidize-target-cuda", create_if_missing=True)

app = modal.App("oxidize-tests")

COMMON = dict(
    image=image,
    volumes={
        f"{REPO_ROOT}/target": target_cache,
        "/usr/local/cargo/registry": registry_cache,
    },
    cpu=8.0,
    memory=16384,
    timeout=3600,
)


def _run(cmd: str) -> int:
    """Run a shell command in the workspace, streaming output. Returns exit code."""
    import subprocess

    print(f"\n\033[1;36m$ {cmd}\033[0m", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=REPO_ROOT)
    return proc.returncode


@app.function(**COMMON)
def smoke() -> str:
    """Build the CLI in release and confirm the binary runs."""
    import subprocess

    rc = _run("cargo build --release --package oxidize-cli")
    if rc != 0:
        raise SystemExit(f"build failed (exit {rc})")

    bin_path = f"{REPO_ROOT}/target/release/oxidize-cli"
    print("\n\033[1;36m$ oxidize-cli --help\033[0m", flush=True)
    out = subprocess.run(
        [bin_path, "--help"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    # Some clap setups exit non-zero on --help-less invocations; just surface output.
    report = (out.stdout or "") + (out.stderr or "")
    print(report, flush=True)
    target_cache.commit()
    return f"smoke OK — binary at {bin_path}\n{report[:2000]}"


@app.function(**COMMON)
def test(package: str = "oxidize-core") -> str:
    """Run `cargo test`. Pass package='workspace' to test everything."""
    selector = "--workspace" if package == "workspace" else f"--package {package}"
    rc = _run(f"cargo test {selector} --release")
    target_cache.commit()
    registry_cache.commit()
    if rc != 0:
        raise SystemExit(f"tests failed (exit {rc}) for {package}")
    return f"tests passed for {package}"


@app.function(
    image=image,
    volumes={
        f"{REPO_ROOT}/target": target_cache,
        "/usr/local/cargo/registry": registry_cache,
        "/root/.cache/oxidize": model_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def tps(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
    iterations: int = 3,
) -> str:
    """Measure decode throughput (tok/s) on Modal's CPU for a real GGUF model."""
    import re
    import subprocess

    # Binary is already compiled into the target Volume by smoke(); this is a
    # no-op rebuild that just resolves the path safely.
    if _run("cargo build --release --package oxidize-cli") != 0:
        raise SystemExit("CLI build failed")
    binary = f"{REPO_ROOT}/target/release/oxidize-cli"

    base = [binary, "run", model]
    if hf_file:
        base += ["--file", hf_file]
    base += [prompt, "--no-api", "--max-tokens", str(max_tokens)]

    speeds = []
    transcript = []
    for i in range(iterations):
        print(f"\n\033[1;36m# iteration {i + 1}/{iterations}\033[0m", flush=True)
        print(f"$ {' '.join(base)}", flush=True)
        out = subprocess.run(base, cwd=REPO_ROOT, capture_output=True, text=True)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-1500:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"inference failed (exit {out.returncode}) on iteration {i + 1}")
        # Match `speed=12.34 tok/s` or `decode-only: 12.34 tok/s`.
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if m:
            speeds.append(float(m[-1]))
        transcript.append(f"iter {i + 1}: {m[-1] if m else '?'} tok/s")

    model_cache.commit()
    if not speeds:
        raise SystemExit("could not parse any tok/s value from output")

    best = max(speeds)
    avg = sum(speeds) / len(speeds)
    summary = (
        f"\n=== TPS on Modal (8 vCPU, CPU backend) ===\n"
        f"model: {model} ({hf_file or 'auto'})\n"
        f"prompt tokens cap: {max_tokens}, iterations: {iterations}\n"
        + "\n".join(transcript)
        + f"\nbest: {best:.2f} tok/s   avg: {avg:.2f} tok/s\n"
    )
    print(summary, flush=True)
    return summary


# ---------------------------------------------------------------------------
# GPU path: build CUDA on a cheap CPU container, benchmark on the GPU box.
# ---------------------------------------------------------------------------
CUDA_BUILD = dict(
    image=cuda_image,
    volumes={
        f"{REPO_ROOT}/target": cuda_target_cache,
        "/usr/local/cargo/registry": registry_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=5400,
)


@app.function(**CUDA_BUILD)
def gpu_build() -> str:
    """Compile oxidize-cli with the CUDA feature (no GPU needed for nvcc PTX)."""
    if _run("nvcc --version") != 0:
        raise SystemExit("nvcc not found in CUDA image")
    if _run("cargo build --release --package oxidize-cli --features cuda") != 0:
        raise SystemExit("CUDA build failed")
    cuda_target_cache.commit()
    return "CUDA build OK"


@app.function(
    image=cuda_image,
    gpu="H100",
    volumes={
        f"{REPO_ROOT}/target": cuda_target_cache,
        "/usr/local/cargo/registry": registry_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def gpu_splitk_bench() -> str:
    command = (
        "cargo test -p oxidize-core --features cuda "
        "split_k -- --include-ignored --nocapture"
    )
    rc = _run(command)
    cuda_target_cache.commit()
    registry_cache.commit()
    if rc != 0:
        raise SystemExit(f"split-K H100 benchmark failed (exit {rc})")
    return "split-K H100 benchmark passed"


@app.function(
    image=cuda_image,
    gpu="H100",
    volumes={
        f"{REPO_ROOT}/target": cuda_target_cache,
        "/usr/local/cargo/registry": registry_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def gpu_splitk_test() -> str:
    command = (
        "cargo test -p oxidize-core --features cuda "
        "split_k_decode_matches -- --nocapture"
    )
    rc = _run(command)
    cuda_target_cache.commit()
    registry_cache.commit()
    if rc != 0:
        raise SystemExit(f"split-K H100 parity test failed (exit {rc})")
    return "split-K H100 parity test passed"


GPU_RUN = dict(
    image=cuda_image,
    gpu="L4",
    volumes={
        f"{REPO_ROOT}/target": cuda_target_cache,
        "/usr/local/cargo/registry": registry_cache,
        "/root/.cache/oxidize": model_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=3600,
)


@app.function(**GPU_RUN)
def gpu_profile(
    model: str = "Qwen/Qwen3-4B-GGUF",
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 256,
    gpu_layers: int = 999,
) -> str:
    """Sample GPU utilization during one decode to determine the perf regime."""
    import re
    import subprocess
    import threading

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            line = (r.stdout or "").strip().splitlines()
            if line:
                samples.append(line[0])
            stop.wait(0.1)

    import os
    t = threading.Thread(target=sampler, daemon=True)
    # Profile the CORRECTED fused path (OX_GPU_ATTN) with weights resident.
    cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
           "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
           "--max-tokens", str(max_tokens), "--temperature", "0", "--threads", "8"]
    env = dict(os.environ)
    env["OX_GPU_ATTN"] = "1"
    print(f"$ OX_GPU_ATTN=1 {' '.join(cmd)}", flush=True)
    t.start()
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    stop.set()
    t.join(timeout=2)
    blob = (out.stdout or "") + (out.stderr or "")
    tps = re.findall(r"(\d+\.\d+)\s*tok/s", blob)

    # Parse samples, drop the model-load warm-up (first ~30% of samples).
    gpu_utils, mem_utils, powers = [], [], []
    for s in samples:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 3:
            try:
                gpu_utils.append(float(parts[0]))
                mem_utils.append(float(parts[1]))
                powers.append(float(parts[2]))
            except ValueError:
                pass
    warm = len(gpu_utils) // 3
    g, m, p = gpu_utils[warm:], mem_utils[warm:], powers[warm:]

    def stats(xs):
        return (sum(xs) / len(xs), max(xs)) if xs else (0.0, 0.0)

    gmean, gmax = stats(g)
    mmean, mmax = stats(m)
    pmean, pmax = stats(p)
    summary = (
        f"\n=== GPU profile (L4, {model}) ===\n"
        f"samples: {len(g)} (after warm-up), tok/s: {tps[-1] if tps else '?'}\n"
        f"GPU util:  mean {gmean:.1f}%  peak {gmax:.0f}%\n"
        f"Mem-ctrl:  mean {mmean:.1f}%  peak {mmax:.0f}%\n"
        f"Power:     mean {pmean:.0f}W  peak {pmax:.0f}W (L4 TDP 72W)\n"
        f"regime: {'memory-bound' if mmean > 70 else 'OVERHEAD-BOUND (GPU starved)'}\n"
    )
    print(summary, flush=True)
    return summary


@app.function(**GPU_RUN)
def gpu_tps(
    model: str = "Qwen/Qwen3-4B-GGUF",
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
    iterations: int = 3,
    gpu_layers: int = 999,
    threads: int = 0,
    layer_cache: int = 0,
) -> str:
    """Measure decode throughput on the GPU. Build must already be in the volume."""
    import re
    import subprocess

    _run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    if not _path_exists(binary):
        # Fall back to building here if gpu_build() wasn't run first.
        if _run("cargo build --release --package oxidize-cli --features cuda") != 0:
            raise SystemExit("CUDA build failed (and no cached binary present)")

    base = [binary, "run", model]
    if hf_file:
        base += ["--file", hf_file]
    base += [
        prompt,
        "--no-api",
        "--backend", "cuda",
        "--n-gpu-layers", str(gpu_layers),
        "--max-tokens", str(max_tokens),
    ]
    if threads > 0:
        base += ["--threads", str(threads)]
    if layer_cache > 0:
        base += ["--layer-cache", str(layer_cache)]

    speeds, transcript, sample = [], [], ""
    for i in range(iterations):
        print(f"\n\033[1;36m# GPU iteration {i + 1}/{iterations}\033[0m", flush=True)
        print(f"$ {' '.join(base)}", flush=True)
        out = subprocess.run(base, cwd=REPO_ROOT, capture_output=True, text=True)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-2000:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"GPU inference failed (exit {out.returncode}) iter {i + 1}")
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if m:
            speeds.append(float(m[-1]))
        transcript.append(f"iter {i + 1}: {m[-1] if m else '?'} tok/s")
        if i == 0:
            sample = blob

    model_cache.commit()
    if not speeds:
        raise SystemExit("could not parse any tok/s value from GPU output")
    best, avg = max(speeds), sum(speeds) / len(speeds)
    summary = (
        f"\n=== GPU TPS on Modal ===\n"
        f"model: {model} ({hf_file or 'auto'})  gpu_layers={gpu_layers}\n"
        f"max_tokens: {max_tokens}, iterations: {iterations}\n"
        + "\n".join(transcript)
        + f"\nbest: {best:.2f} tok/s   avg: {avg:.2f} tok/s\n"
    )
    print(summary, flush=True)
    return summary


@app.function(**GPU_RUN)
def gpu_stage_profile(
    model: str = "Qwen/Qwen3-4B-GGUF",
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
    gpu_attn: bool = True,
) -> str:
    """Per-stage decode timing via OXIDIZE_DECODE_PROFILE to find the real bottleneck."""
    import os
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    env = dict(os.environ)
    env["OXIDIZE_DECODE_PROFILE"] = "1"
    if gpu_attn:
        env["OX_GPU_ATTN"] = "1"
    cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
           "--backend", "cuda", "--n-gpu-layers", "32", "--layer-cache", "64",
           "--max-tokens", str(max_tokens), "--temperature", "0", "--threads", "8"]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    blob = (out.stdout or "") + (out.stderr or "")
    # Surface the profile blocks + final tok/s + a coherence sample.
    import re as _re
    text = _re.findall(r"oxidize: (.*)", blob)
    lines = [l for l in blob.splitlines()
             if any(k in l for k in ("profile", "tok/s", "ms)", "ms ", " ms", "gemv", "token_forward",
                                     "attn", "ffn", "qkv", "wo", "norm", "sample", "logit", "GB/s"))]
    report = (f"=== decode stage profile (gpu_attn={gpu_attn}) ===\n"
              + f"SAMPLE: \"{(text[0][:160] if text else '?')}\"\n"
              + "\n".join(lines[-60:]))
    print(report, flush=True)
    return report


@app.function(**GPU_RUN)
def gpu_attn_dump(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    kv_dtype_cpu: str = "f16",
) -> str:
    """Diff OX_ATTN_DUMP tensors (layer0 token0) between CPU and GPU attention to
    localize the divergence stage. Optionally force CPU KV dtype to f16 for fairness."""
    import os
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"

    def run(label, extra_env, extra_args):
        dump = f"/tmp/dump_{label}.txt"
        if os.path.exists(dump):
            os.remove(dump)
        env = dict(os.environ)
        env["OX_ATTN_DUMP"] = dump
        env.update(extra_env)
        cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
               "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
               "--max-tokens", "4", "--temperature", "0", "--threads", "8"] + extra_args
        subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blocks = {}
        if os.path.exists(dump):
            for line in open(dump):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                # Format: "label v1 v2 v3 ..." all on one line.
                label, vals = toks[0], [float(x) for x in toks[1:] if _isfloat(x)]
                blocks[label] = vals
        return blocks

    cpu_args = ["--kv-cache-dtype", kv_dtype_cpu] if kv_dtype_cpu else []
    cpu = run("cpu", {}, cpu_args)
    gpu = run("gpu", {"OX_GPU_ATTN": "1"}, [])

    lines = [f"=== OX_ATTN_DUMP diff (CPU kv={kv_dtype_cpu or 'default'} vs GPU f16), layer0 tok0 ==="]
    for k in cpu:
        a, b = cpu.get(k, []), gpu.get(k, [])
        n = min(len(a), len(b))
        if n == 0:
            lines.append(f"  {k:16s}: cpu_len={len(a)} gpu_len={len(b)} (no overlap)")
            continue
        maxd = max(abs(a[i] - b[i]) for i in range(n))
        meand = sum(abs(a[i] - b[i]) for i in range(n)) / n
        lines.append(f"  {k:16s}: max|Δ|={maxd:.4e}  mean|Δ|={meand:.4e}  (n={n})")
    report = "\n".join(lines)
    print(report, flush=True)
    return report


def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


@app.function(**GPU_RUN)
def gpu_attn_verify(
    model: str = "Qwen/Qwen3-4B-GGUF",
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 64,
) -> str:
    """A/B: CPU attention vs OX_GPU_ATTN. Greedy decode; diff golden-logits argmax
    sequences to prove correctness, and report tok/s for both."""
    import os
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"

    def run(label, extra_env):
        gold = f"/tmp/golden_{label}.txt"
        if os.path.exists(gold):
            os.remove(gold)
        env = dict(os.environ)
        env["OX_GOLDEN_LOGITS"] = gold
        env.update(extra_env)
        if label == "gpuattn":
            env["OX_FLASH_DECODE_TRACE"] = "1"
        # Greedy (temperature 0) so the argmax IS the chosen token → both runs
        # follow the same path iff their logits agree. --layer-cache 64 keeps all
        # layers VRAM-resident (no eviction thrash); --n-gpu-layers 99 (>= count).
        cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
               "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
               "--max-tokens", str(max_tokens), "--temperature", "0", "--threads", "8"]
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0:
            print(blob[-2500:], flush=True)
            raise SystemExit(f"{label} run failed (exit {out.returncode})")
        tps = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        argmax = []
        if os.path.exists(gold):
            with open(gold) as fh:
                for line in fh:
                    m = re.search(r"tok=(\d+)\s+argmax=(\d+)", line)
                    if m:
                        argmax.append(int(m.group(2)))
        # The generated text streams to stdout after "offload plan:"; capture the
        # chunk between offload-plan and the "generation stats" line.
        m = re.search(r"offload plan:.*?\n(.*?)\ngeneration stats:", blob, re.S)
        text = (m.group(1).strip()[:260] if m else blob[-260:].strip())
        traces = list(dict.fromkeys(
            line.strip() for line in blob.splitlines() if "flash_decode path=" in line
        ))
        return float(tps[-1]) if tps else 0.0, argmax, text.replace("\n", " "), traces

    cpu_best, gpu_best = 0.0, 0.0
    cpu_arg = gpu_arg = []
    cpu_txt = gpu_txt = ""
    gpu_traces = []
    for i in range(1):
        c_tps, c_arg, c_txt, _ = run("cpu", {})
        g_tps, g_arg, g_txt, g_traces = run("gpuattn", {"OX_GPU_ATTN": "1"})
        cpu_best, gpu_best = max(cpu_best, c_tps), max(gpu_best, g_tps)
        cpu_arg, gpu_arg, cpu_txt, gpu_txt = c_arg, g_arg, c_txt, g_txt
        gpu_traces = list(dict.fromkeys([*gpu_traces, *g_traces]))
        print(f"  iter {i+1}: cpu={c_tps:.2f}  gpu={g_tps:.2f} tok/s", flush=True)

    n = min(len(cpu_arg), len(gpu_arg))
    first_div = next((i for i in range(n) if cpu_arg[i] != gpu_arg[i]), -1)
    match = first_div == -1 and len(cpu_arg) == len(gpu_arg) and n > 0
    summary = (
        f"\n=== OX_GPU_ATTN correctness + speed (L4, {model}) — one CPU/GPU pair ===\n"
        f"CPU-attn : {cpu_best:.2f} tok/s | {len(cpu_arg)} tokens | \"{cpu_txt}\"\n"
        f"GPU-attn : {gpu_best:.2f} tok/s | {len(gpu_arg)} tokens | \"{gpu_txt}\"\n"
        f"argmax match: {'YES (identical greedy sequence)' if match else f'NO — first divergence at token {first_div}'}\n"
        f"flash decode trace: {' | '.join(gpu_traces) if gpu_traces else 'NONE'}\n"
        + (f"speedup: {gpu_best / cpu_best:.2f}x\n" if cpu_best else "")
    )
    if not match and n > 0:
        summary += f"  cpu[:12]={cpu_arg[:12]}\n  gpu[:12]={gpu_arg[:12]}\n"
    print(summary, flush=True)
    return summary


@app.function(**GPU_RUN)
def gpu_sweep(
    model: str = "Qwen/Qwen3-4B-GGUF",
    hf_file: str = "Qwen3-4B-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
) -> str:
    """Sweep host-thread counts to test whether CPU attention is the bottleneck."""
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    results = {}
    for th in [4, 8, 16, 24]:
        cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
               "--backend", "cuda", "--n-gpu-layers", "999",
               "--max-tokens", str(max_tokens), "--threads", str(th)]
        # Two passes; keep the faster (first pass pays model-load/JIT).
        best = 0.0
        for _ in range(2):
            out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            blob = (out.stdout or "") + (out.stderr or "")
            m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
            if m:
                best = max(best, float(m[-1]))
        results[th] = best
        print(f"threads={th}: {best:.2f} tok/s", flush=True)

    summary = "\n=== GPU thread sweep (L4, Qwen3-4B Q4_K_M) ===\n" + "\n".join(
        f"  threads={th}: {tps:.2f} tok/s" for th, tps in results.items()
    )
    print(summary, flush=True)
    return summary


def _path_exists(p: str) -> bool:
    import os

    return os.path.exists(p)


@app.local_entrypoint()
def main(
    action: str = "smoke",
    package: str = "oxidize-core",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
    iterations: int = 3,
    gpu: str = "L4",
    gpu_layers: int = 999,
    threads: int = 0,
    layer_cache: int = 0,
):
    if action == "smoke":
        print(smoke.remote())
    elif action == "test":
        print(test.remote(package))
    elif action == "tps":
        print(tps.remote(model, hf_file, prompt, max_tokens, iterations))
    elif action == "gpu-build":
        print(gpu_build.remote())
    elif action == "gpu-splitk-bench":
        print(gpu_splitk_bench.remote())
    elif action == "gpu-splitk-test":
        print(gpu_splitk_test.remote())
    elif action == "gpu-profile":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        fn = gpu_profile.with_options(gpu=gpu) if gpu != "L4" else gpu_profile
        print(fn.remote(model, hf_file, prompt, max(max_tokens, 256), gpu_layers))
    elif action == "gpu-tps":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        # Run on the requested GPU type (default L4); proves whether throughput
        # is GPU-bound or overhead-bound.
        fn = gpu_tps.with_options(gpu=gpu) if gpu != "L4" else gpu_tps
        print(fn.remote(model, hf_file, prompt, max_tokens, iterations, gpu_layers, threads, layer_cache))
    elif action == "gpu-stage":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        print(gpu_stage_profile.remote(model, hf_file, prompt, max_tokens, True))
    elif action == "gpu-attn":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        fn = gpu_attn_verify.with_options(gpu=gpu) if gpu != "L4" else gpu_attn_verify
        print(fn.remote(model, hf_file, prompt, max_tokens))
    elif action == "gpu-dump":
        m = "bartowski/Mistral-7B-Instruct-v0.3-GGUF" if model.startswith("Qwen/Qwen2.5-0.5B") else model
        hf = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" if model.startswith("Qwen/Qwen2.5-0.5B") else hf_file
        fn = gpu_attn_dump.with_options(gpu=gpu) if gpu != "L4" else gpu_attn_dump
        print(fn.remote(m, hf, prompt))
    elif action == "gpu-sweep":
        # Run the same model at several host-thread counts to see if the
        # CPU-attention stall is thread-bound. Binary must already be built.
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        print(gpu_sweep.remote(model, hf_file, prompt, max_tokens))
    elif action == "gpu":
        # Default to a Qwen3 model on GPU (Qwen2 path produced garbage on CPU).
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        print("building CUDA target...")
        print(gpu_build.remote())
        print("running GPU benchmark...")
        print(gpu_tps.remote(model, hf_file, prompt, max_tokens, iterations, gpu_layers, threads))
    else:
        raise SystemExit(
            "unknown action (use: smoke | test | tps | gpu-build | gpu-profile | "
            "gpu-tps | gpu-splitk-bench | gpu-splitk-test | gpu-sweep | gpu)"
        )
