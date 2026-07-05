"""Benchmark oxidize-c (quantized-resident CUDA) on Gemma 4 31B IQ4_XS.

    modal run modal_gemma4_bench.py::bench_l4
    modal run modal_gemma4_bench.py::bench_a10g

Downloads the GGUF into a Volume once, compiles oxidize-c-cuda in the
container, runs a decode benchmark, prints tok/s.
"""

import modal

app = modal.App("oxidize-c-gemma4-bench")

GGUF_URL = (
    "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/"
    "gemma-4-31B-it-IQ4_XS.gguf"
)
GGUF = "/vol/gemma-4-31B-it-IQ4_XS.gguf"

vol = modal.Volume.from_name("gemma4-gguf", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl")
    .add_local_dir("oxidize-c", "/src/oxidize-c", copy=True)
)


@app.function(volumes={"/vol": vol}, timeout=7200)
def download():
    import os
    import subprocess

    if os.path.exists(GGUF) and os.path.getsize(GGUF) > 16_000_000_000:
        print("already downloaded")
        return
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", GGUF, GGUF_URL], check=True)
    vol.commit()


def _bench(max_tokens: int, ctx: int) -> None:
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    subprocess.run(["make", "cuda"], cwd="/src/oxidize-c", check=True)
    prompt = "Write a detailed essay about the history of computing, starting with Babbage."
    subprocess.run(
        [
            "/src/oxidize-c/oxidize-c-cuda",
            "--model", GGUF,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--ctx", str(ctx),
            "--stream",
        ],
        check=True,
    )


@app.function(gpu="L4", image=image, volumes={"/vol": vol}, timeout=3600, memory=16384, cpu=8)
def bench_l4(max_tokens: int = 128, ctx: int = 4096):
    _bench(max_tokens, ctx)


@app.function(gpu="A10G", image=image, volumes={"/vol": vol}, timeout=3600, memory=16384, cpu=8)
def bench_a10g(max_tokens: int = 128, ctx: int = 4096):
    _bench(max_tokens, ctx)


@app.local_entrypoint()
def main():
    download.remote()
    print("=== L4 ===")
    bench_l4.remote()
    print("=== A10G ===")
    bench_a10g.remote()
