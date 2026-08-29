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
    ".cursor/**",
    ".claude/**",    # agent scheduler/lock files churn during build (modified-during-build guard)
    "deploy/**",
    "uv.lock",
    "bun.lock",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
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

# Fresh-mount CPU image: copy=True bakes source into the layer so edits ALWAYS
# bust the content hash. Use for benches that depend on just-edited source (the
# copy=False `image` above can serve a stale snapshot across rapid edits).
cpu_fresh_image = (
    modal.Image.from_registry(f"rust:{RUST_VERSION}", add_python="3.12")
    .apt_install("pkg-config", "libssl-dev", "cmake", "clang")
    .add_local_dir(".", REPO_ROOT, ignore=IGNORE, copy=True)
)

# Writable, persistent caches. `target/` is the big one; the registry cache
# avoids re-downloading crates.io deps on every run.
target_cache = modal.Volume.from_name("oxidize-target-cache", create_if_missing=True)
registry_cache = modal.Volume.from_name("oxidize-cargo-registry", create_if_missing=True)
# Persists HF model downloads (~/.cache/oxidize/hf) across runs.
model_cache = modal.Volume.from_name("oxidize-model-cache", create_if_missing=True)

# build.rs compiles kernels/gemv_f32.cu -> PTX with nvcc, so we need the *devel*
# CUDA image (toolkit), not just runtime. No GPU is needed to *compile* the PTX.
CUDA_TAG = "12.8.1-devel-ubuntu22.04"
cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "curl",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "cmake",
        "clang",
        "git",
        "nsight-systems-2025.1.3",
    )
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


@app.function(
    image=cpu_fresh_image,
    volumes={
        f"{REPO_ROOT}/target": target_cache,
        "/usr/local/cargo/registry": registry_cache,
        "/root/.cache/oxidize": model_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def batched_decode_tps(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    steps: int = 64,
    batches: str = "1,8,16",
) -> str:
    """Prove TRUE continuous-batching decode: aggregate tok/s should scale with
    batch size because decode is memory-bound and N sequences share one set of
    batched weight reads (`InferenceModel::forward_batch`). Runs on Modal's 8 vCPU
    CPU box and reports tok/s for each batch size, plus a flag-off control."""
    import glob
    import os
    import re
    import subprocess

    # Build the CLI (includes the batched_decode_bench bin) into the cache.
    if _run("cargo build --release --package oxidize-cli --bin batched_decode_bench") != 0:
        raise SystemExit("batched_decode_bench build failed")
    bench = f"{REPO_ROOT}/target/release/batched_decode_bench"
    cli = f"{REPO_ROOT}/target/release/oxidize-cli"
    if _run("cargo build --release --package oxidize-cli --bin oxidize-cli") != 0:
        raise SystemExit("oxidize-cli build failed")

    # `cache_safe_name` in model_resolution.rs replaces every non-alphanumeric
    # char with '-'. The GGUF lands at ~/.cache/oxidize/hf/<safe>/main/<file>.
    safe = re.sub(r"[^A-Za-z0-9]", "-", model)
    gguf = f"/root/.cache/oxidize/hf/{safe}/main/{hf_file}"
    if not os.path.exists(gguf):
        # Trigger oxidize's HF resolver to populate the cache (1 token is enough).
        warm = [cli, "run", model]
        if hf_file:
            warm += ["--file", hf_file]
        warm += ["warm", "--no-api", "--max-tokens", "1"]
        print(f"$ {' '.join(warm)}", flush=True)
        print("(streaming model download/warm output below)", flush=True)
        subprocess.run(warm, cwd=REPO_ROOT)
        if not os.path.exists(gguf):
            hits = glob.glob(f"/root/.cache/oxidize/hf/{safe}/main/*.gguf")
            if hits:
                gguf = hits[0]
            else:
                raise SystemExit(f"could not locate GGUF at {gguf} (warm exit {w.returncode})")
    print(f"resolved GGUF: {gguf}", flush=True)

    sizes = [int(b) for b in batches.split(",") if b.strip()]

    def run_one(batch: int, flag_on: bool) -> float:
        env = dict(os.environ)
        if flag_on:
            env["OX_BATCHED_DECODE"] = "1"
        else:
            env.pop("OX_BATCHED_DECODE", None)
        cmd = [bench, "--model", gguf, "--batch", str(batch), "--steps", str(steps)]
        print(f"\n\033[1;36m# batch={batch} flag={'on' if flag_on else 'off'}\033[0m", flush=True)
        print(f"$ {' '.join(cmd)}", flush=True)
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-1500:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"batched_decode_bench failed (exit {out.returncode}) at batch={batch}")
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if not m:
            raise SystemExit(f"could not parse tok/s at batch={batch}")
        return float(m[-1])

    # Control: flag OFF at batch=1 (proves default path is unaffected).
    control = run_one(1, flag_on=False)

    results = {b: run_one(b, flag_on=True) for b in sizes}
    base = results.get(1, results[sizes[0]])

    lines = [f"control (flag off) batch=1: {control:.2f} tok/s"]
    for b in sizes:
        ratio = results[b] / base if base > 0 else float("nan")
        lines.append(f"batch={b:<3d} {results[b]:.2f} tok/s   ({ratio:.2f}x vs batch=1)")

    model_cache.commit()
    summary = (
        f"\n=== Batched decode TPS on Modal (8 vCPU, CPU backend) ===\n"
        f"model: {model} ({hf_file or 'auto'})  steps/seq: {steps}\n"
        + "\n".join(lines)
        + "\n\nExpectation: aggregate tok/s grows with batch size (decode is\n"
        "memory-bound, so N sequences amortize one set of weight reads) until\n"
        "compute-bound; ~2-4x by batch=16 on 8 vCPU is the success signal.\n"
    )
    print(summary, flush=True)
    return summary


# GPU path: build CUDA on a cheap CPU container, benchmark on the GPU box.
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
        "flash_decode_cuda_tests -- --include-ignored --nocapture"
    )
    rc = _run(command)
    cuda_target_cache.commit()
    registry_cache.commit()
    if rc != 0:
        raise SystemExit(f"split-K GPU benchmark failed (exit {rc})")
    return "split-K GPU benchmark passed"


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
        raise SystemExit(f"split-K GPU parity test failed (exit {rc})")
    return "split-K GPU parity test passed"


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
def gpu_gemv_mw_bench() -> str:
    command = (
        "cargo test --release -p oxidize-core --features cuda "
        "q4k_gemv_block_size_sweep_is_exact_and_faster "
        "-- --ignored --nocapture"
    )
    rc = _run(command)
    cuda_target_cache.commit()
    registry_cache.commit()
    if rc != 0:
        raise SystemExit(f"multi-row Q4_K benchmark failed (exit {rc})")
    return "multi-row Q4_K benchmark passed"


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


def _requantize_q4ks(gguf: str) -> str:
    """Produce a uniform strict-Q4_K (Q4_K_S) copy of `gguf`.

    The on-device batched decode path requires EVERY projection + the lm_head to
    be strict Q4_K (144-byte blocks). Stock Q4_K_M mixes in Q6_K tensors (attn_v,
    ffn_down, output) so it is ineligible and falls back to CPU. Q4_K_S in
    oxidize-quantize applies Q4_K uniformly (no Q6_K), making the model eligible.
    Caches the result next to the source so repeat runs skip the conversion."""
    import os
    import subprocess

    out = gguf + ".q4ks.gguf"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"requantized Q4_K_S GGUF (cached): {out}", flush=True)
        return out
    if _run("cargo build --release --package oxidize-quantize") != 0:
        raise SystemExit("oxidize-quantize build failed")
    qbin = f"{REPO_ROOT}/target/release/oxidize-quantize"
    cmd = [qbin, "--input", gguf, "--output", out, "--target", "Q4_K_S"]
    print(f"$ {' '.join(cmd)}", flush=True)
    print("(streaming requantize output below)", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"requantize to Q4_K_S failed (exit {r.returncode})")
    model_cache.commit()
    print(f"requantized Q4_K_S GGUF: {out}", flush=True)
    return out


@app.function(**GPU_RUN)
def gpu_batched_verify(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-f32.gguf",
) -> str:
    """Numerical-correctness gate for the on-device batched decode forward:
    `batched_gpu_matches_single_seq` asserts the B-row `forward_batch_gpu` output
    matches B independent single-seq runs by argmax (+ top-1 logit tolerance)."""
    import glob
    import os
    import re
    import subprocess

    # Resolve / warm a real Q4_K GGUF the test can mmap (the test reads
    # OX_BATCHED_TEST_GGUF, defaulting to /root/models/<file>).
    if _run("cargo build --release --package oxidize-cli --bin oxidize-cli --features cuda") != 0:
        raise SystemExit("oxidize-cli CUDA build failed")
    cli = f"{REPO_ROOT}/target/release/oxidize-cli"
    safe = re.sub(r"[^A-Za-z0-9]", "-", model)
    gguf = f"/root/.cache/oxidize/hf/{safe}/main/{hf_file}"
    if not os.path.exists(gguf):
        warm = [cli, "run", model, "--file", hf_file, "warm", "--no-api", "--max-tokens", "1"]
        print(f"$ {' '.join(warm)}", flush=True)
        print("(streaming model download/warm output below)", flush=True)
        subprocess.run(warm, cwd=REPO_ROOT)
        if not os.path.exists(gguf):
            hits = glob.glob(f"/root/.cache/oxidize/hf/{safe}/main/*.gguf")
            gguf = hits[0] if hits else gguf
    print(f"resolved GGUF: {gguf}", flush=True)
    gguf = _requantize_q4ks(gguf)

    env = dict(os.environ)
    env["OX_BATCHED_TEST_GGUF"] = gguf
    env["OX_GPU_BATCHED_DEBUG"] = "1"
    command = (
        "cargo test -p oxidize-core --features cuda "
        "batched_gpu_matches_single_seq -- --include-ignored --nocapture"
    )
    print(f"$ {command}", flush=True)
    rc = subprocess.run(command, shell=True, cwd=REPO_ROOT, env=env).returncode
    cuda_target_cache.commit()
    registry_cache.commit()
    model_cache.commit()
    if rc != 0:
        raise SystemExit(f"batched GPU parity test failed (exit {rc})")
    return "batched GPU parity test passed"


@app.function(**GPU_RUN)
def gpu_batched_tps(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-f32.gguf",
    prompt: str = "1,2,3,4",
    max_tokens: int = 64,
) -> str:
    """Throughput of the on-device batched decode path at B=1,4,8. Drives
    `batched_decode_bench` with OX_GPU_BATCHED=1 (which routes decode rows through
    `forward_batch_gpu`, falling back to CPU when ineligible) and prints aggregate
    tok/s scaling vs B=1, plus a flag-OFF control proving the default path is
    unchanged."""
    import glob
    import os
    import re
    import subprocess

    if _run("cargo build --release --package oxidize-cli --bin batched_decode_bench --features cuda") != 0:
        raise SystemExit("batched_decode_bench CUDA build failed")
    if _run("cargo build --release --package oxidize-cli --bin oxidize-cli --features cuda") != 0:
        raise SystemExit("oxidize-cli CUDA build failed")
    bench = f"{REPO_ROOT}/target/release/batched_decode_bench"
    cli = f"{REPO_ROOT}/target/release/oxidize-cli"

    safe = re.sub(r"[^A-Za-z0-9]", "-", model)
    gguf = f"/root/.cache/oxidize/hf/{safe}/main/{hf_file}"
    if not os.path.exists(gguf):
        warm = [cli, "run", model, "--file", hf_file, "warm", "--no-api", "--max-tokens", "1"]
        print(f"$ {' '.join(warm)}", flush=True)
        print("(streaming model download/warm output below)", flush=True)
        subprocess.run(warm, cwd=REPO_ROOT)
        if not os.path.exists(gguf):
            hits = glob.glob(f"/root/.cache/oxidize/hf/{safe}/main/*.gguf")
            if hits:
                gguf = hits[0]
            else:
                raise SystemExit(f"could not locate GGUF at {gguf}")
    print(f"resolved GGUF: {gguf}", flush=True)
    gguf = _requantize_q4ks(gguf)

    steps = max(max_tokens, 64)

    def run_one(batch: int, gpu_batched: bool) -> float:
        env = dict(os.environ)
        if gpu_batched:
            env["OX_GPU_BATCHED"] = "1"
        else:
            env.pop("OX_GPU_BATCHED", None)
        cmd = [bench, "--model", gguf, "--batch", str(batch), "--steps", str(steps),
               "--prompt", prompt]
        label = "gpu_batched" if gpu_batched else "default"
        print(f"\n\033[1;36m# batch={batch} {label}\033[0m\n$ {' '.join(cmd)}", flush=True)
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-1500:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"batched_decode_bench failed (exit {out.returncode}) at batch={batch}")
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if not m:
            raise SystemExit(f"could not parse tok/s at batch={batch}")
        return float(m[-1])

    # Control: OX_GPU_BATCHED unset at B=1 (default device forward unchanged).
    control = run_one(1, gpu_batched=False)

    sizes = [1, 4, 8]
    results = {b: run_one(b, gpu_batched=True) for b in sizes}
    base = results[1]

    lines = [f"control (OX_GPU_BATCHED unset) batch=1: {control:.2f} tok/s"]
    for b in sizes:
        ratio = results[b] / base if base > 0 else float("nan")
        lines.append(f"batch={b:<2d} {results[b]:.2f} tok/s   ({ratio:.2f}x vs batch=1)")

    cuda_target_cache.commit()
    registry_cache.commit()
    model_cache.commit()
    summary = (
        f"\n=== Batched GPU decode TPS on H100 ===\n"
        f"model: {model} ({hf_file})  steps/seq: {steps}\n"
        + "\n".join(lines)
        + "\n\nExpectation: aggregate tok/s grows with B because decode is\n"
        "HBM-bandwidth-bound and B sequences amortize one weight pass per\n"
        "projection (`forward_batch_gpu`). The control proves OX_GPU_BATCHED\n"
        "unset leaves the single-token device path untouched.\n"
    )
    print(summary, flush=True)
    return summary


@app.function(**GPU_RUN)
def gpu_best(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 160,
) -> str:
    """Sweep GPU decode configs to find the best tok/s (vs llama.cpp 156 on A100)."""
    import os
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    _run("nvidia-smi --query-gpu=name --format=csv,noheader")
    configs = [
        ("default (attn+mw+lmhead_q8k)", {"OX_GPU_ATTN": "1"}, 8),
        ("no lmhead_q8k", {"OX_GPU_ATTN": "1", "OX_GPU_LMHEAD_Q8K": "0"}, 8),
        ("fused_qkv", {"OX_GPU_ATTN": "1", "OX_GPU_FUSED_QKV": "1"}, 8),
        ("gpu_native (cpu-attn)", {}, 8),
    ]
    results = []
    for label, extra, threads in configs:
        env = dict(os.environ)
        env.update(extra)
        cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
               "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
               "--max-tokens", str(max_tokens), "--temperature", "0", "--threads", str(threads)]
        best = 0.0
        note = ""
        for _ in range(2):
            out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
            blob = (out.stdout or "") + (out.stderr or "")
            if out.returncode != 0:
                note = "FAILED: " + (re.search(r"InferenceFailed\(\"(.*?)\"", blob).group(1)
                                     if re.search(r"InferenceFailed", blob) else blob[-100:].replace("\n", " "))
                break
            m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
            if m:
                best = max(best, float(m[-1]))
        results.append((label, best, note))
        print(f"  {label:26s}: {best:6.2f} tok/s  {note[:60]}", flush=True)

    results.sort(key=lambda r: -r[1])
    bl, bt, _ = results[0]
    summary = (
        f"\n=== oxidize BEST GPU decode (A100, Mistral-7B Q4_K_M) ===\n"
        + "\n".join(f"  {l:26s}: {t:6.2f} tok/s {n[:40]}" for l, t, n in results)
        + f"\nBEST: {bl} = {bt:.2f} tok/s   (llama.cpp same A100+file: 156)\n"
        + (f"gap: {156 / bt:.1f}x\n" if bt > 0 else "all configs failed\n")
    )
    print(summary, flush=True)
    return summary


@app.function(**GPU_RUN)
def gpu_rewind_determinism(
    repo: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
) -> str:
    """Phase-2 device KV-rollback validation. Forwards a probe token, rewind_to,
    re-forwards it, asserts identical logits. With OX_GPU_ATTN=1 on CUDA this
    exercises gpu_kv_rewind across every layer (the speculative-rejection fix);
    a missed device rollback would attend over a stale row and diverge."""
    import os
    import subprocess

    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, hf_file, cache_dir="/root/.cache/oxidize/hf-determinism")
    os.environ["OXIDIZE_TEST_GGUF_MODELS"] = path
    os.environ["OX_GPU_ATTN"] = "1"
    cmd = (
        "cargo test -p oxidize-core --features cuda "
        "rewind_to_round_trip_is_deterministic_on_real_model -- --ignored --nocapture"
    )
    rc = _run(cmd)
    cuda_target_cache.commit()
    registry_cache.commit()
    model_cache.commit()
    if rc != 0:
        raise SystemExit(f"rewind determinism (device KV rollback) test failed (exit {rc})")
    return "rewind determinism (device KV rollback) test passed"


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
                # Ignore malformed sampler lines and continue.
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
def gpu_next_profile(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 64,
) -> str:
    """Profile FFN fusion with CUDA events and a real H100 CLI A/B."""
    import os
    import re
    import statistics
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    build = _run("cargo build --release -p oxidize-cli --features cuda")
    cuda_target_cache.commit()
    registry_cache.commit()
    if build != 0:
        raise SystemExit(f"current-tree CUDA CLI build failed (exit {build})")

    benchmark = subprocess.run(
        [
            "cargo",
            "test",
            "--release",
            "-p",
            "oxidize-core",
            "--features",
            "cuda",
            "q4k_gate_up_silu_cuda_event_benchmark",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    benchmark_blob = (benchmark.stdout or "") + (benchmark.stderr or "")
    if benchmark.returncode != 0:
        print(benchmark_blob[-3000:], flush=True)
        raise SystemExit("direct CUDA-event FFN benchmark failed")
    direct_metrics: dict[str, float] = {}
    for marker in ("eager_ffn_ms", "fused_ffn_ms"):
        match = re.search(rf"^{marker}=([0-9.]+)$", benchmark_blob, re.MULTILINE)
        if match is None:
            raise SystemExit(f"direct CUDA-event benchmark omitted {marker}")
        direct_metrics[marker] = float(match.group(1))
    if min(direct_metrics.values()) <= 0.0:
        raise SystemExit("direct CUDA-event benchmark reported a non-positive median")

    local_model = "/root/.cache/oxidize/hf/bartowski-Mistral-7B-Instruct-v0-3-GGUF/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    common_args = [
        "--no-api",
        "--no-auto",
        "--backend",
        "cuda",
        "--n-gpu-layers",
        "99",
        "--layer-cache",
        "64",
        "--ctx-size",
        "4096",
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--threads",
        "8",
        prompt,
    ]
    if not os.path.isfile(local_model):
        resolve = subprocess.run(
            [binary, "run", model, "--file", hf_file, *common_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if resolve.returncode != 0 or "generation failed:" in (
            (resolve.stdout or "") + (resolve.stderr or "")
        ):
            print(((resolve.stdout or "") + (resolve.stderr or ""))[-3000:], flush=True)
            raise SystemExit("one-time model resolution failed")
    if not os.path.isfile(local_model):
        raise SystemExit(f"resolved model is missing at {local_model}")

    command = [
        binary,
        "run",
        local_model,
        *common_args,
    ]
    variants = (
        (
            "off",
            {
                "OX_GPU_FFN_FUSE": "0",
                "OX_GPU_LMHEAD_MULTIROW": "0",
                "OX_GPU_GEMV_BLOCK": "256",
            },
        ),
        (
            "on",
            {
                "OX_GPU_FFN_FUSE": "1",
                "OX_GPU_LMHEAD_MULTIROW": "0",
                "OX_GPU_GEMV_BLOCK": "1024",
            },
        ),
    )
    measurements: dict[str, list[float]] = {}
    traces: dict[str, tuple[tuple[int, ...], str]] = {}

    def parse_token_wall(blob: str) -> float:
        entries = re.findall(
            r"^\s*(\S+)\s+(\d+)x(\d+)\s+calls=\d+\s+total=\s*([0-9.]+)ms",
            blob,
            flags=re.MULTILINE,
        )
        scalar = {
            label: float(ms)
            for label, rows, cols, ms in entries
            if rows == "0" and cols == "0"
        }
        token_wall = scalar.get("token_forward", 0.0)
        if token_wall <= 0.0:
            raise SystemExit("OXIDIZE_DECODE_PROFILE omitted token_forward total")
        return token_wall

    def parse_generated_output(blob: str) -> str:
        match = re.search(r"offload plan:.*?\n(.*?)\ngeneration stats:", blob, re.S)
        return " ".join(match.group(1).split()) if match is not None else ""

    for label, optimization_env in variants:
        token_wall_samples: list[float] = []
        reference_trace: tuple[int, ...] | None = None
        reference_output: str | None = None
        for iteration in range(13):
            stem = f"/tmp/oxidize-next-{label}-{iteration}"
            golden = f"{stem}.golden"
            env = dict(os.environ)
            env.update(
                {
                    "OX_GPU_ATTN": "1",
                    "OXIDIZE_DECODE_PROFILE": "1",
                    "OX_GOLDEN_LOGITS": golden,
                    **optimization_env,
                }
            )
            out = subprocess.run(
                command, cwd=REPO_ROOT, capture_output=True, text=True, env=env
            )
            blob = (out.stdout or "") + (out.stderr or "")
            if out.returncode != 0 or "generation failed:" in blob:
                print(blob[-3000:], flush=True)
                raise SystemExit(f"profile run {label}/{iteration} failed")
            token_wall_ms = parse_token_wall(blob)
            if iteration >= 3:
                token_wall_samples.append(token_wall_ms)

            with open(golden, encoding="utf-8") as handle:
                trace = tuple(
                    int(match.group(1))
                    for match in re.finditer(r"argmax=(\d+)", handle.read())
                )
            output = parse_generated_output(blob)
            if not trace or not output:
                print(f"profile output tail:\n{blob[-3000:]}", flush=True)
                raise SystemExit("profile run omitted greedy logit trace or output")
            if reference_trace is not None and trace != reference_trace:
                raise SystemExit(f"non-deterministic greedy trace for FFN fuse {label}")
            if reference_output is not None and output != reference_output:
                raise SystemExit(f"non-deterministic greedy output for FFN fuse {label}")
            reference_trace, reference_output = trace, output
            if os.path.exists(golden):
                os.remove(golden)
        if reference_trace is None or reference_output is None:
            raise SystemExit(f"no completed measurements for FFN fuse {label}")
        traces[label] = (reference_trace, reference_output)
        measurements[label] = token_wall_samples

    if traces["off"] != traces["on"]:
        raise SystemExit("exact CUDA optimizations changed greedy output or logit trace")
    token_wall_off_ms = statistics.median(measurements["off"])
    token_wall_on_ms = statistics.median(measurements["on"])
    if token_wall_on_ms <= 0.0:
        raise SystemExit("fused CLI token-wall median is non-positive")
    lines = [
        "=== H100 direct-event FFN + CLI profile: Mistral-7B Q4_K_M ===",
        f"eager_ffn_ms={direct_metrics['eager_ffn_ms']:.6f}",
        f"fused_ffn_ms={direct_metrics['fused_ffn_ms']:.6f}",
        f"token_wall_off_ms={token_wall_off_ms:.3f}",
        f"token_wall_on_ms={token_wall_on_ms:.3f}",
        f"token_wall_speedup={token_wall_off_ms / token_wall_on_ms:.3f}",
    ]
    lines.append("greedy_output_parity=PASS")
    lines.append("greedy_logit_trace_parity=PASS")
    summary = "\n".join(lines) + "\n"
    print(summary, flush=True)
    model_cache.commit()
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
    gpu_attn: bool | None = None,
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
    else:
        base += ["--layer-cache", "64"]

    speeds, transcript = [], []
    env = None
    if gpu_attn is not None:
        import os
        env = dict(os.environ)
        if gpu_attn:
            env["OX_GPU_ATTN"] = "1"
        else:
            env["OX_GPU_ATTN"] = "0"
    for i in range(iterations):
        print(f"\n\033[1;36m# GPU iteration {i + 1}/{iterations}\033[0m", flush=True)
        print(f"$ {' '.join(base)}", flush=True)
        out = subprocess.run(base, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-2000:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"GPU inference failed (exit {out.returncode}) iter {i + 1}")
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if m:
            speeds.append(float(m[-1]))
        transcript.append(f"iter {i + 1}: {m[-1] if m else '?'} tok/s")

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
           "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
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
               "--no-auto", "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
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
    import os
    import re
    import subprocess

    if prompt == "@splitk-long":
        prompt = ("The quick brown fox checks exact CUDA attention semantics. " * 60).strip()

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    if not os.path.exists(binary):
        build_rc = _run("cargo build --release -p oxidize-cli --features cuda")
        cuda_target_cache.commit()
        registry_cache.commit()
        if build_rc != 0:
            raise SystemExit(f"current-tree CLI CUDA build failed (exit {build_rc})")

    def run(label, extra_env):
        gold = f"/tmp/golden_{label}.txt"
        if os.path.exists(gold):
            os.remove(gold)
        env = dict(os.environ)
        env["OX_GOLDEN_LOGITS"] = gold
        env.update(extra_env)
        env["OX_FLASH_DECODE_TRACE"] = "1"
        # Greedy (temperature 0) so the argmax IS the chosen token → both runs
        # follow the same path iff their logits agree. --layer-cache 64 keeps all
        # layers VRAM-resident (no eviction thrash); --n-gpu-layers 99 (>= count).
        cmd = [binary, "run", model, "--file", hf_file, "--no-api",
               "--no-auto", "--backend", "cuda", "--n-gpu-layers", "99",
               "--layer-cache", "64", "--ctx-size", "4096", "--max-tokens", str(max_tokens),
               "--temperature", "0", "--threads", "8", prompt]
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0:
            print(blob[-2500:], flush=True)
            raise SystemExit(f"{label} run failed (exit {out.returncode})")
        if "generation failed:" in blob:
            print(blob[-2500:], flush=True)
            raise SystemExit(f"{label} reported a generation failure")
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

    legacy_best, split_best = 0.0, 0.0
    legacy_arg = split_arg = []
    legacy_txt = split_txt = ""
    gpu_traces = []
    for i in range(1):
        l_tps, l_arg, l_txt, l_traces = run(
            "legacy", {"OX_GPU_ATTN": "1", "OX_FLASH_DECODE_FORCE_LEGACY": "1"}
        )
        s_tps, s_arg, s_txt, s_traces = run("splitk", {"OX_GPU_ATTN": "1"})
        legacy_best, split_best = max(legacy_best, l_tps), max(split_best, s_tps)
        legacy_arg, split_arg, legacy_txt, split_txt = l_arg, s_arg, l_txt, s_txt
        gpu_traces = list(dict.fromkeys([*l_traces, *s_traces]))
        print(f"  iter {i+1}: legacy={l_tps:.2f} split_k={s_tps:.2f} tok/s", flush=True)

    n = min(len(legacy_arg), len(split_arg))
    first_div = next((i for i in range(n) if legacy_arg[i] != split_arg[i]), -1)
    argmax_match = first_div == -1 and len(legacy_arg) == len(split_arg) and n > 0
    output_match = bool(legacy_txt) and legacy_txt == split_txt
    match = argmax_match or output_match
    summary = (
        f"\n=== production flash decode A/B (H100, {model}) ===\n"
        f"legacy  : {legacy_best:.2f} tok/s | {len(legacy_arg)} logits | \"{legacy_txt}\"\n"
        f"split-K : {split_best:.2f} tok/s | {len(split_arg)} logits | \"{split_txt}\"\n"
        f"greedy parity: {'YES (identical output)' if match else f'NO — first divergence at token {first_div}'}\n"
        f"flash decode trace: {' | '.join(gpu_traces) if gpu_traces else 'NONE'}\n"
        + (f"speedup: {split_best / legacy_best:.2f}x\n" if legacy_best else "")
    )
    if not match and n > 0:
        summary += f"  legacy[:12]={legacy_arg[:12]}\n  split_k[:12]={split_arg[:12]}\n"
    print(summary, flush=True)
    if not match or not gpu_traces:
        raise SystemExit("CLI attention verification did not prove parity and dispatch")
    return summary


@app.function(**GPU_RUN)
def gpu_dflash_ab(
    target_model: str = "Qwen/Qwen3-4B-GGUF",
    target_file: str = "Qwen3-4B-Q4_K_M.gguf",
    draft_model: str = "Zapdev/Qwen3-4B-DFlash-GGUF",
    draft_file: str = "Qwen3-4B-DFlash-q8_0.gguf",
    prompt: str = "Explain speculative decoding in two concise paragraphs.",
    max_tokens: int = 96,
) -> str:
    import os
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"

    def run(label: str, extra_args: list[str]) -> tuple[float, str]:
        env = dict(os.environ)
        env["OX_DFLASH_OUTPUT_PARITY"] = "1"
        cmd = [
            binary,
            "run",
            target_model,
            "--file",
            target_file,
            "--no-api",
            "--backend",
            "cuda",
            "--n-gpu-layers",
            "99",
            "--layer-cache",
            "64",
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "0",
            "--threads",
            "8",
            *extra_args,
            prompt,
        ]
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        blob = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0:
            print(blob[-2500:], flush=True)
            raise SystemExit(f"{label} DFlash A/B run failed")
        speeds = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        return float(speeds[-1]) if speeds else 0.0, blob

    target_tps, target_blob = run("target", [])
    dflash_tps, dflash_blob = run(
        "dflash",
        ["--draft-model", draft_model, "--draft-file", draft_file],
    )
    target_text = _generated_text_sample(target_blob)
    dflash_text = _generated_text_sample(dflash_blob)
    parity = target_text == dflash_text and bool(target_text)
    parity_line = "dflash_output_parity=PASS" if parity else "dflash_output_parity=FAIL"
    summary = (
        "\n=== DFlash CLI A/B ===\n"
        f"target : {target_tps:.2f} tok/s | \"{target_text}\"\n"
        f"dflash : {dflash_tps:.2f} tok/s | \"{dflash_text}\"\n"
        f"{parity_line}\n"
        f"draft_file={draft_file}\n"
    )
    print(summary, flush=True)
    if not parity:
        raise SystemExit("dflash_output_parity=FAIL")
    return summary


def _generated_text_sample(blob: str) -> str:
    import re

    match = re.search(r"offload plan:.*?\n(.*?)\ngeneration stats:", blob, re.S)
    if match:
        return match.group(1).strip()[:260].replace("\n", " ")
    return blob[-260:].strip().replace("\n", " ")


@app.function(**GPU_RUN)
def gpu_flag_sweep(
    model: str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_file: str = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    prompt: str = "Explain what a tokenizer does in two sentences.",
    max_tokens: int = 128,
) -> str:
    """A/B the GPU-native decode path against the FORCED-CPU baseline
    (OX_GPU_ATTN=0). The baseline is pure-CPU attention+projections — for a
    QK-norm model (Qwen3) that is the ONLY path the engine could use before the
    GPU QK-norm change, so `gpu_attn` here measures the CPU->GPU routing win.
    Greedy decode with OX_GOLDEN_LOGITS so each config's argmax sequence is
    compared to the CPU baseline — a perf win that changes the tokens is a BUG.
    Reports tok/s, speedup vs baseline, and PARITY/DIVERGED for every config."""
    import os
    import re
    import subprocess

    binary = f"{REPO_ROOT}/target/release/oxidize-cli"
    _run("nvidia-smi --query-gpu=name --format=csv,noheader")

    def run(label, extra_env):
        gold = f"/tmp/golden_{label}.txt"
        if os.path.exists(gold):
            os.remove(gold)
        env = dict(os.environ)
        env["OX_GOLDEN_LOGITS"] = gold
        env.update(extra_env)
        cmd = [binary, "run", model, "--file", hf_file, prompt, "--no-api",
               "--backend", "cuda", "--n-gpu-layers", "99", "--layer-cache", "64",
               "--max-tokens", str(max_tokens), "--temperature", "0", "--threads", "8"]
        best = 0.0
        argmax = []
        # Two passes; keep the faster (first pays model-load/PTX JIT).
        for _ in range(2):
            out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
            blob = (out.stdout or "") + (out.stderr or "")
            if out.returncode != 0:
                print(blob[-1500:], flush=True)
                return 0.0, [], "FAILED"
            m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
            if m:
                best = max(best, float(m[-1]))
        if os.path.exists(gold):
            with open(gold) as fh:
                for line in fh:
                    mm = re.search(r"tok=(\d+)\s+argmax=(\d+)", line)
                    if mm:
                        argmax.append(int(mm.group(2)))
        return best, argmax, "ok"

    configs = [
        ("cpu_baseline", {"OX_GPU_ATTN": "0"}),
        ("gpu_attn", {"OX_GPU_ATTN": "1"}),
        # Multi-warp-per-row Q4_K f32-in GEMV (QKV+Wo). Same f32 activation as
        # gpu_attn, so it must share gpu_attn's divergence point exactly while
        # running faster (more in-flight HBM requests per row).
        ("gpu_attn+gemv_mw", {"OX_GPU_ATTN": "1", "OX_GPU_GEMV_MW": "1"}),
        ("gpu_attn+ffn_fuse", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "1"}),
        ("gpu_attn+fused_qkv", {"OX_GPU_ATTN": "1", "OX_GPU_FUSED_QKV": "1"}),
        ("gpu_attn+layer_q8k", {"OX_GPU_ATTN": "1", "OX_GPU_LAYER_Q8K": "1"}),
        ("gpu_full_stack", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "1",
                            "OX_GPU_FUSED_MMQ": "1", "OX_GPU_LMHEAD_MULTIROW": "1"}),
        ("gpu_full_stack+gemv_mw", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "1",
                                    "OX_GPU_FUSED_MMQ": "1", "OX_GPU_LMHEAD_MULTIROW": "1",
                                    "OX_GPU_GEMV_MW": "1"}),
        ("gpu_full+cuda_graph", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "1",
                                 "OX_GPU_FUSED_MMQ": "1", "OX_GPU_LMHEAD_MULTIROW": "1",
                                 "OX_GPU_CUDA_GRAPH": "1"}),
        ("gpu_full+mw+graph", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "1",
                               "OX_GPU_FUSED_MMQ": "1", "OX_GPU_LMHEAD_MULTIROW": "1",
                               "OX_GPU_GEMV_MW": "1", "OX_GPU_CUDA_GRAPH": "1"}),
        ("eager_no_opts", {"OX_GPU_ATTN": "1", "OX_GPU_FFN_FUSE": "0",
                           "OX_GPU_FUSED_MMQ": "0", "OX_GPU_CUDA_GRAPH": "0",
                           "OX_GPU_LMHEAD_MULTIROW": "0", "OX_GPU_GEMV_MW": "0"}),
    ]
    results = []
    base_tps, base_arg, _ = run(*configs[0])
    results.append((configs[0][0], base_tps, "REF (CPU)"))
    for label, env in configs[1:]:
        tps, arg, status = run(label, env)
        if status == "FAILED":
            results.append((label, 0.0, "FAILED"))
            continue
        n = min(len(base_arg), len(arg))
        first_div = next((i for i in range(n) if base_arg[i] != arg[i]), -1)
        parity = first_div == -1 and len(arg) == len(base_arg) and n > 0
        spd = f"{tps / base_tps:.3f}x" if base_tps else "?"
        results.append((label, tps,
                        f"{spd}  {'PARITY' if parity else f'DIVERGED@{first_div}'}"))

    lines = [f"  {l:20s}: {t:6.2f} tok/s  {note}" for l, t, note in results]
    best_gpu = max((t for l, t, n in results if l != "cpu_baseline"), default=0.0)
    summary = (
        f"\n=== GPU decode A/B ({model}) ===\n"
        f"baseline = OX_GPU_ATTN=0 (forced CPU); gpu_* rows route to the GPU\n"
        + "\n".join(lines)
        + f"\nBEST gpu row: {best_gpu:.2f} tok/s  (llama.cpp Mistral-7B Q4_K_M A100 ref: ~156 tok/s)\n"
        + (f"gap vs llama.cpp: {156 / best_gpu:.1f}x\n" if best_gpu > 0 else "")
        + "\n(PARITY = identical greedy argmax sequence to CPU baseline = correct)\n"
    )
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


@app.function(**GPU_RUN)
def llama_cpp_bench(
    gguf: str = "/root/.cache/oxidize/hf/bartowski-Mistral-7B-Instruct-v0-3-GGUF/main/Mistral-7B-Instruct-v0.3-f32.gguf.q4ks.gguf",
) -> str:
    """Same-box llama.cpp reference on H100 against the SAME Q4_K_S GGUF oxidize
    benches. Builds llama.cpp (CUDA, sm_90) once into the model_cache volume, then
    reports single-stream tg (llama-bench) and batched aggregate t/s at npl=1,4,8
    (llama-batched-bench) for an apples-to-apples comparison."""
    import os
    import subprocess

    root = "/root/.cache/oxidize/llamacpp"
    bench = f"{root}/build/bin/llama-bench"
    bbench = f"{root}/build/bin/llama-batched-bench"
    if not (os.path.exists(bench) and os.path.exists(bbench)):
        if not os.path.exists(f"{root}/CMakeLists.txt"):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggml-org/llama.cpp", root], check=True)
        subprocess.run(
            ["cmake", "-B", f"{root}/build", "-S", root,
             "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=90",
             "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(
            ["cmake", "--build", f"{root}/build", "-j", "--config", "Release",
             "--target", "llama-bench", "llama-batched-bench"], check=True)
        model_cache.commit()
    if not os.path.exists(gguf):
        raise SystemExit(f"GGUF not found: {gguf} (run gpu-batched-tps first to populate it)")

    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    print("\n\033[1;36m# llama.cpp single-stream (tg128)\033[0m", flush=True)
    s = subprocess.run([bench, "-m", gguf, "-ngl", "99", "-p", "0", "-n", "128", "-r", "3"],
                       capture_output=True, text=True)
    print((s.stdout or "") + (s.stderr or "")[-500:], flush=True)
    print("\n\033[1;36m# llama.cpp batched-bench (npl 1,4,8)\033[0m", flush=True)
    b = subprocess.run(
        [bbench, "-m", gguf, "-ngl", "99", "-c", "8192",
         "-npp", "16", "-ntg", "128", "-npl", "1,4,8"],
        capture_output=True, text=True)
    print((b.stdout or "") + (b.stderr or "")[-500:], flush=True)
    return "llama.cpp bench done"


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
    elif action == "batched-tps":
        # max_tokens doubles as decode steps/seq here; default batches 1,8,16.
        print(batched_decode_tps.remote(model, hf_file, max(max_tokens, 64), "1,8,16"))
    elif action == "gpu-build":
        print(gpu_build.remote())
    elif action == "gpu-splitk-bench":
        print(gpu_splitk_bench.with_options(gpu=gpu).remote())
    elif action == "gpu-splitk-test":
        print(gpu_splitk_test.with_options(gpu=gpu).remote())
    elif action == "gpu-gemv-mw-bench":
        print(gpu_gemv_mw_bench.with_options(gpu=gpu).remote())
    elif action == "gpu-profile":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        fn = gpu_profile.with_options(gpu=gpu) if gpu != "L4" else gpu_profile
        print(fn.remote(model, hf_file, prompt, max(max_tokens, 256), gpu_layers))
    elif action == "gpu-next-profile":
        m = "bartowski/Mistral-7B-Instruct-v0.3-GGUF" if model.startswith("Qwen/Qwen2.5-0.5B") else model
        hf = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" if model.startswith("Qwen/Qwen2.5-0.5B") else hf_file
        print(
            gpu_next_profile.with_options(gpu="H100").remote(
                m, hf, prompt, max(max_tokens, 64)
            )
        )
    elif action == "gpu-best":
        m = "bartowski/Mistral-7B-Instruct-v0.3-GGUF" if model.startswith("Qwen/Qwen2.5-0.5B") else model
        hf = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" if model.startswith("Qwen/Qwen2.5-0.5B") else hf_file
        fn = gpu_best.with_options(gpu=gpu) if gpu != "L4" else gpu_best
        print(fn.remote(m, hf, prompt, max(max_tokens, 160)))
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
        fn = gpu_attn_verify.with_options(gpu=gpu) if gpu != "L4" else gpu_attn_verify
        print(fn.remote(model, hf_file, prompt, max_tokens))
    elif action == "gpu-batched-verify":
        fn = gpu_batched_verify.with_options(gpu=gpu) if gpu != "L4" else gpu_batched_verify
        print(fn.remote(model, hf_file))
    elif action == "llama-cpp-bench":
        print(llama_cpp_bench.with_options(gpu="H100").remote())
    elif action == "gpu-batched-tps":
        # Pin the eligible no-bias model + a NUMERIC prompt (the bench parses
        # --prompt as comma-separated u32 token ids, not free text).
        fn = gpu_batched_tps.with_options(gpu="H100")
        print(fn.remote(
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "Mistral-7B-Instruct-v0.3-f32.gguf",
            "1,2,3,4",
            max(max_tokens, 64),
        ))
    elif action == "gpu-dump":
        m = "bartowski/Mistral-7B-Instruct-v0.3-GGUF" if model.startswith("Qwen/Qwen2.5-0.5B") else model
        hf = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" if model.startswith("Qwen/Qwen2.5-0.5B") else hf_file
        fn = gpu_attn_dump.with_options(gpu=gpu) if gpu != "L4" else gpu_attn_dump
        print(fn.remote(m, hf, prompt))
    elif action == "gpu-flag-sweep":
        if model == "Qwen/Qwen2.5-0.5B-Instruct-GGUF":
            model, hf_file = "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"
        fn = gpu_flag_sweep.with_options(gpu=gpu) if gpu != "L4" else gpu_flag_sweep
        print(fn.remote(model, hf_file, prompt, max_tokens))
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
            "unknown action (use: smoke | test | tps | batched-tps | gpu-build | "
            "gpu-profile | gpu-next-profile | gpu-tps | gpu-splitk-bench | "
            "gpu-splitk-test | gpu-sweep | gpu)"
        )
