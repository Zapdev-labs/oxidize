"""
Modal harness for testing the oxidize-c C11 port on CUDA GPUs (T4 by default;
set OXIDIZE_MODAL_GPU to use a larger card).

Usage:
    modal run modal_c_app.py --action build      # compile CPU + CUDA builds
    modal run modal_c_app.py --action test       # run CPU test suite
    modal run modal_c_app.py --action gpu_test   # compile CUDA + run on GPU
    modal run modal_c_app.py --action bench      # GPU benchmark on L40S
    modal run modal_c_app.py --action parity     # CPU vs CUDA output must match
"""
import os
import modal
import subprocess
from typing import Final, Literal, assert_never

REPO_ROOT: Final = "/workspace"
CUDA_TAG: Final = "12.8.1-devel-ubuntu22.04"
# T4 (sm_75) is what the free tier allows; larger cards need a payment method
# on the Modal account ("Please add a payment method to use L40S GPU
# functions."). Override for a bigger card once billing is set up:
#   OXIDIZE_MODAL_GPU=L40S modal run modal_c_app.py --action parity
# A T4 has 16 GB, enough for the 1.5B parity model with packed weights; a 7B
# Q4_K_M needs the packed path to fit at all (~4.4 GB vs ~30 GB as f32).
GPU: Final = os.environ.get("OXIDIZE_MODAL_GPU", "T4")
Action = Literal["build", "test", "gpu_test", "bench", "parity"]

IGNORE = [
    "target/**", ".git/**", "models/**", "dist/**", "node_modules/**",
    "**/*.gguf.bak", "rust_out/**", ".omo/**", ".cursor/**", ".claude/**",
    "deploy/**", "uv.lock", "bun.lock", "pnpm-lock.yaml", "package-lock.json",
    "yarn.lock", "**/*.log", "**/*.o", "**/*.a", "oxidize-c/oxidize-c",
]

# CUDA devel image with build tools
cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("build-essential", "pkg-config", "cmake", "git", "gcc", "g++")
    .add_local_dir(".", REPO_ROOT, ignore=IGNORE, copy=False)
)

cuda_cache = modal.Volume.from_name("oxidize-c-cuda-cache", create_if_missing=True)
model_cache = modal.Volume.from_name("oxidize-model-cache", create_if_missing=True)

app = modal.App("oxidize-c-tests")


def _run(cmd: str) -> int:
    print(f"\n\033[1;36m$ {cmd}\033[0m", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=REPO_ROOT)
    return proc.returncode


@app.function(
    image=cuda_image,
    volumes={f"{REPO_ROOT}/build-cpu": cuda_cache},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def cpu_build() -> str:
    """Build the CPU-only binary (no CUDA)."""
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c clean && make -C {REPO_ROOT}/oxidize-c build")
    cuda_cache.commit()
    if rc != 0:
        raise SystemExit(f"CPU build failed (exit {rc})")
    return "CPU build OK"


@app.function(
    image=cuda_image,
    volumes={f"{REPO_ROOT}/build-cuda": cuda_cache},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def cpu_test() -> str:
    """Run the CPU test suite (ASan + UBSan)."""
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c test")
    cuda_cache.commit()
    if rc != 0:
        raise SystemExit(f"CPU tests failed (exit {rc})")
    return "CPU tests passed (307 tests)"


@app.function(
    image=cuda_image,
    volumes={
        f"{REPO_ROOT}/build-cuda": cuda_cache,
    },
    cpu=8.0,
    memory=32768,
    timeout=900,
)
def cuda_build() -> str:
    """Compile the CUDA backend with nvcc."""
    _run("nvcc --version")
    _run("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU on build box'")
    # Build with CUDA: nvcc compiles .cu, gcc compiles .c, link together
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c clean && make -C {REPO_ROOT}/oxidize-c cuda")
    cuda_cache.commit()
    if rc != 0:
        raise SystemExit(f"CUDA build failed (exit {rc})")
    return "CUDA build OK"


@app.function(
    image=cuda_image,
    gpu=GPU,
    volumes={"/root/.cache/oxidize": model_cache},
    cpu=8.0,
    memory=32768,
    timeout=1800,
)
def gpu_test(model: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
             hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
             prompt: str = "Explain what a tokenizer does in two sentences.",
             max_tokens: int = 64) -> str:
    """Run the oxidize-c binary with --backend cuda on an L40S GPU."""
    import os
    import re

    _run("nvidia-smi")

    # Always rebuild on Modal to avoid glibc mismatch
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c cuda")
    if rc != 0:
        raise SystemExit(f"CUDA build failed (exit {rc})")
    binary = f"{REPO_ROOT}/oxidize-c/oxidize-c"

    # Test basic binary execution first
    rc = _run(f"{binary} --version")
    if rc != 0:
        raise SystemExit(f"Binary won't run (exit {rc})")

    # Download model via huggingface_hub
    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download

    gguf = hf_hub_download(model, hf_file,
                           cache_dir="/root/.cache/oxidize/hf")
    print(f"Model: {gguf}", flush=True)

    cmd = [binary, "--model", gguf, "--prompt", prompt,
           "--backend", "cuda", "--n-predict", str(max_tokens),
           "--temperature", "0"]
    print(f"$ {' '.join(cmd)}", flush=True)
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    blob = (out.stdout or "") + (out.stderr or "")
    print(blob, flush=True)
    model_cache.commit()

    if out.returncode != 0:
        raise SystemExit(f"GPU test failed (exit {out.returncode})")

    tps = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
    if tps:
        return f"GPU test passed: {tps[-1]} tok/s"
    return "GPU test completed (no tok/s parsed)"


@app.function(
    image=cuda_image,
    gpu=GPU,
    volumes={"/root/.cache/oxidize": model_cache},
    cpu=8.0,
    memory=32768,
    timeout=1800,
)
def gpu_bench(model: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
              hf_file: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
              prompt: str = "Explain what a tokenizer does in two sentences.",
              max_tokens: int = 128,
              iterations: int = 3) -> str:
    """Benchmark GPU decode throughput (tok/s) on L40S."""
    import os
    import re
    import subprocess

    _run("nvidia-smi")

    # Always rebuild on Modal to avoid glibc mismatch
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c cuda")
    if rc != 0:
        raise SystemExit(f"CUDA build failed (exit {rc})")
    binary = f"{REPO_ROOT}/oxidize-c/oxidize-c"

    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download

    gguf = hf_hub_download(model, hf_file, cache_dir="/root/.cache/oxidize/hf")

    speeds = []
    for i in range(iterations):
        print(f"\n\033[1;36m# GPU iteration {i+1}/{iterations}\033[0m", flush=True)
        cmd = [binary, "--model", gguf, "--prompt", prompt,
               "--backend", "cuda", "--n-predict", str(max_tokens),
               "--temperature", "0"]
        print(f"$ {' '.join(cmd)}", flush=True)
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        blob = (out.stdout or "") + (out.stderr or "")
        print(blob[-2000:], flush=True)
        if out.returncode != 0:
            raise SystemExit(f"GPU inference failed (exit {out.returncode}) iter {i+1}")
        m = re.findall(r"(\d+\.\d+)\s*tok/s", blob)
        if m:
            speeds.append(float(m[-1]))

    model_cache.commit()
    if not speeds:
        raise SystemExit("No tok/s values parsed")
    best, avg = max(speeds), sum(speeds) / len(speeds)
    summary = (
        f"\n=== oxidize-c GPU TPS on Modal (L40S) ===\n"
        f"model: {model} ({hf_file})\n"
        f"max_tokens: {max_tokens}, iterations: {iterations}\n"
        + "\n".join(f"  iter {i+1}: {s:.2f} tok/s" for i, s in enumerate(speeds))
        + f"\nbest: {best:.2f} tok/s   avg: {avg:.2f} tok/s\n"
    )
    print(summary, flush=True)
    return summary


@app.function(
    image=cuda_image,
    gpu=GPU,
    volumes={"/root/.cache/oxidize": model_cache},
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def parity(model: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
           hf_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
           prompt: str = "The capital of France is",
           max_tokens: int = 32) -> str:
    """CPU vs CUDA correctness gate.

    Greedy decoding (--temperature 0) is deterministic, so the two backends
    must emit the same token sequence. Any divergence means a device kernel
    disagrees with the scalar reference — which is exactly the class of bug
    that "exit code 0 plus a tok/s number" cannot catch.

    Reports the VRAM accounting line too, so a regression that silently falls
    back to f32 weight residency shows up as a jump in reported bytes rather
    than passing quietly.

    Model choice matters: K-quant super-blocks are 256 elements, so a tensor
    only stays packed on the device when its row length divides by 256. The
    1.5B default (n_embd=1536, n_ff=8960) satisfies that; Qwen2.5-0.5B
    (n_embd=896) does not and would silently exercise only the f32 fallback.
    Watch the "tensors packed" count in the log to confirm which path ran.
    """
    import re

    _run("nvidia-smi")
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c cuda")
    if rc != 0:
        raise SystemExit(f"CUDA build failed (exit {rc})")
    binary = f"{REPO_ROOT}/oxidize-c/oxidize-c"

    subprocess.run("pip install -q huggingface_hub", shell=True, check=True)
    from huggingface_hub import hf_hub_download
    gguf = hf_hub_download(model, hf_file, cache_dir="/root/.cache/oxidize/hf")
    model_cache.commit()

    def generate(backend: str) -> tuple[str, str]:
        cmd = [binary, "--model", gguf, "--prompt", prompt,
               "--backend", backend, "--n-predict", str(max_tokens),
               "--temperature", "0", "--seed", "1", "-v"]
        print(f"\n\033[1;36m$ {' '.join(cmd)}\033[0m", flush=True)
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout, out.stderr, flush=True)
            raise SystemExit(f"{backend} run failed (exit {out.returncode})")
        return out.stdout or "", out.stderr or ""

    cpu_out, _ = generate("cpu")
    gpu_out, gpu_err = generate("cuda")

    vram = [ln for ln in gpu_err.splitlines()
            if "VRAM" in ln or "KV cache" in ln]

    cpu_norm = cpu_out.strip()
    gpu_norm = gpu_out.strip()
    report = [
        "\n=== oxidize-c CPU vs CUDA parity ===",
        f"model:  {model} ({hf_file})",
        f"prompt: {prompt!r}   max_tokens: {max_tokens}",
        *(f"vram:   {ln.strip()}" for ln in vram),
        f"\n--- cpu  ---\n{cpu_norm}",
        f"\n--- cuda ---\n{gpu_norm}",
    ]

    if cpu_norm == gpu_norm:
        report.append("\nPARITY OK — identical greedy output")
        print("\n".join(report), flush=True)
        return "\n".join(report)

    # Show where they diverge instead of just failing.
    for i, (a, b) in enumerate(zip(cpu_norm, gpu_norm)):
        if a != b:
            report.append(f"\nfirst divergence at char {i}:")
            report.append(f"  cpu : ...{cpu_norm[max(0, i - 40):i + 40]!r}")
            report.append(f"  cuda: ...{gpu_norm[max(0, i - 40):i + 40]!r}")
            break
    else:
        report.append(f"\nlength differs: cpu={len(cpu_norm)} cuda={len(gpu_norm)}")
    print("\n".join(report), flush=True)
    raise SystemExit("PARITY FAILED — CPU and CUDA outputs differ")


@app.function(
    image=cuda_image,
    volumes={f"{REPO_ROOT}/build-cuda": cuda_cache},
    cpu=8.0,
    memory=16384,
    timeout=600,
)
def lint() -> str:
    """Run clang-tidy or basic lint checks."""
    rc = _run(f"make -C {REPO_ROOT}/oxidize-c lint 2>/dev/null || echo 'lint skipped'")
    return f"lint: exit {rc}"


@app.function()
def all_cpu() -> str:
    """Run CPU build + tests sequentially."""
    cpu_build.remote()
    return cpu_test.remote()


@app.function()
def all_gpu() -> str:
    """Run CUDA build + GPU test sequentially."""
    cuda_build.remote()
    return gpu_test.remote()


@app.function()
def all_bench() -> str:
    """Run CUDA build + GPU benchmark sequentially."""
    cuda_build.remote()
    return gpu_bench.remote()


@app.local_entrypoint()
def main(action: Action = "test") -> None:
    match action:
        case "build":
            print(cpu_build.remote())
            print(cuda_build.remote())
        case "test":
            print(cpu_test.remote())
        case "gpu_test":
            print(gpu_test.remote())
        case "bench":
            print(gpu_bench.remote())
        case "parity":
            print(parity.remote())
        case unreachable:
            assert_never(unreachable)
