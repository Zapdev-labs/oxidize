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
    .apt_install("build-essential", "curl", "aria2")
    .add_local_dir("oxidize-c", "/src/oxidize-c", copy=True)
)


GGUF12_URL = (
    "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/"
    "gemma-4-12b-it-IQ4_XS.gguf"
)
GGUF12 = "/vol/gemma-4-12b-it-IQ4_XS.gguf"


@app.function(image=image, volumes={"/vol": vol}, timeout=7200)
def download12():
    import os
    import subprocess

    if os.path.exists(GGUF12) and not os.path.exists(GGUF12 + ".aria2"):
        print("already downloaded")
        return
    subprocess.run(
        ["aria2c", "-x16", "-s16", "-c", "--console-log-level=warn",
         "--summary-interval=15", "-d", "/vol", "-o", GGUF12.split("/")[-1],
         GGUF12_URL],
        check=True,
    )
    vol.commit()


@app.function(gpu="A10G", image=image, volumes={"/vol": vol}, timeout=3600, memory=32768, cpu=16)
def debug31():
    import os
    import subprocess

    subprocess.run(["make", "cuda"], cwd="/src/oxidize-c", check=True)
    p = ("<|turn>user\nWrite a short essay about the history of computing."
         "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    env = dict(os.environ, OC_PROF="1")
    subprocess.run(
        ["/src/oxidize-c/oxidize-c-cuda", "--model", GGUF,
         "--prompt", p, "--max-tokens", "6", "--ctx", "2048"],
        env=env, check=True,
    )


@app.function(image=image, volumes={"/vol": vol}, timeout=7200)
def download():
    import os

    if (os.path.exists(GGUF) and os.path.getsize(GGUF) > 16_000_000_000
            and not os.path.exists(GGUF + ".aria2")):
        print("already downloaded")
        return
    import subprocess
    subprocess.run(
        ["aria2c", "-x16", "-s16", "-c", "--console-log-level=warn",
         "--summary-interval=15", "-d", "/vol", "-o", GGUF.split("/")[-1], GGUF_URL],
        check=True,
    )
    print("size:", os.path.getsize(GGUF))
    vol.commit()


def _bench(max_tokens: int, ctx: int) -> None:
    import os
    import subprocess

    if not (os.path.exists(GGUF) and os.path.getsize(GGUF) > 16_000_000_000
            and not os.path.exists(GGUF + ".aria2")):
        subprocess.run(
            ["aria2c", "-x16", "-s16", "-c", "--console-log-level=warn",
             "--summary-interval=15", "-d", "/vol", "-o", GGUF.split("/")[-1], GGUF_URL],
            check=True,
        )
        vol.commit()

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    subprocess.run(["make", "cuda"], cwd="/src/oxidize-c", check=True)
    prompt = ("<|turn>user\nWrite a detailed essay about the history of computing,"
              " starting with Babbage.<turn|>\n<|turn>model\n"
              "<|channel>thought\n<channel|>")
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
def bench_l4(max_tokens: int = 256, kv_ctx: int = 4096):
    _bench(max_tokens, kv_ctx)


@app.function(gpu="A10G", image=image, volumes={"/vol": vol}, timeout=3600, memory=16384, cpu=8)
def bench_a10g(max_tokens: int = 256, kv_ctx: int = 4096):
    _bench(max_tokens, kv_ctx)


@app.local_entrypoint()
def main():
    download.remote()
    print("=== L4 ===")
    bench_l4.remote()
    print("=== A10G ===")
    bench_a10g.remote()
