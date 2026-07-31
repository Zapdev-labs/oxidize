"""Benchmark oxidize-c (quantized-resident CUDA) on Gemma 4 31B IQ4_XS.

    modal run modal_gemma4_bench.py::smoke       # correctness first
    modal run modal_gemma4_bench.py::bench_l40s  # then speed

The model already lives in the `gemma4-gguf` volume; `download` only refetches
if it is missing. The container compiles oxidize-c with the CUDA backend and
runs a decode benchmark.

Gemma 4 31B is 60 layers / hidden 5376 / vocab 262144, and its 16.4 GB of
IQ4_XS weights stay packed in VRAM (see oxidize-c/src/backends/cuda_mmq.cu).
Decode at batch 1 is bandwidth-bound, so the ceiling is roughly
16.4 GB / GPU-bandwidth: ~52 tok/s on an L40S, ~36 on an A10G.
"""

import modal

app = modal.App("oxidize-c-gemma4-bench")

GGUF_URL = (
    "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/"
    "gemma-4-31B-it-IQ4_XS.gguf"
)
GGUF = "/vol/gemma-4-31B-it-IQ4_XS.gguf"
MIN_BYTES = 16_000_000_000

vol = modal.Volume.from_name("gemma4-gguf", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl", "aria2")
    .add_local_dir("oxidize-c", "/src/oxidize-c", copy=True)
)

# `make cuda` links its output as ./oxidize-c — the same name as the CPU
# target, not a separate oxidize-c-cuda binary.
BIN = "/src/oxidize-c/oxidize-c"

# Gemma 4's chat template wraps turns in <|turn>...<turn|> and opens an empty
# thought channel when thinking is disabled.
PROMPT = (
    "<|turn>user\nWrite a detailed essay about the history of computing, "
    "starting with Babbage.<turn|>\n<|turn>model\n"
    "<|channel>thought\n<channel|>"
)


def _have_model() -> bool:
    import os

    return (
        os.path.exists(GGUF)
        and os.path.getsize(GGUF) > MIN_BYTES
        and not os.path.exists(GGUF + ".aria2")
    )


def _fetch() -> None:
    import subprocess

    if _have_model():
        return
    subprocess.run(
        ["aria2c", "-x16", "-s16", "-c", "--console-log-level=warn",
         "--summary-interval=15", "-d", "/vol", "-o", GGUF.split("/")[-1],
         GGUF_URL],
        check=True,
    )
    vol.commit()


def _build() -> None:
    """Compile only for the GPU we are on. The default fatbin covers six
    architectures and dominates container startup; sm_89 is L40S/L4/Ada."""
    import subprocess

    subprocess.run(
        ["make", "cuda", "-j", "8", "CUDA_ARCHS=89"],
        cwd="/src/oxidize-c", check=True,
    )


@app.function(image=image, volumes={"/vol": vol}, timeout=7200)
def download():
    _fetch()
    import os
    print("size:", os.path.getsize(GGUF))


@app.function(gpu="L40S", image=image, volumes={"/vol": vol},
              timeout=3600, memory=32768, cpu=8)
def smoke(max_tokens: int = 48, ctx: int = 2048):
    """Correctness before speed: a fast model that emits garbage is worthless.

    Read the output — it should be coherent English on the requested topic. A
    wrong dual-geometry mapping, a missed Q/K norm, or a bad IQ4_XS unpack can
    all still produce fluent-looking token streams, so this wants eyes on it,
    not just an exit code."""
    import os
    import subprocess

    _fetch()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv"], check=True)
    _build()
    subprocess.run(
        [BIN, "--model", GGUF, "--prompt", PROMPT,
         "--max-tokens", str(max_tokens), "--ctx", str(ctx), "--verbose"],
        env=dict(os.environ, OC_PROF="1"), check=True,
    )


def _bench(max_tokens: int, ctx: int) -> None:
    import subprocess

    _fetch()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv"], check=True)
    _build()
    subprocess.run(
        [BIN, "--model", GGUF, "--prompt", PROMPT,
         "--max-tokens", str(max_tokens), "--ctx", str(ctx), "--stream"],
        check=True,
    )


@app.function(gpu="L40S", image=image, volumes={"/vol": vol},
              timeout=3600, memory=32768, cpu=8)
def bench_l40s(max_tokens: int = 256, ctx: int = 4096):
    _bench(max_tokens, ctx)


@app.function(gpu="A10G", image=image, volumes={"/vol": vol},
              timeout=3600, memory=32768, cpu=8)
def bench_a10g(max_tokens: int = 256, ctx: int = 4096):
    _bench(max_tokens, ctx)


@app.local_entrypoint()
def main():
    download.remote()
    smoke.remote()
    bench_l40s.remote()
