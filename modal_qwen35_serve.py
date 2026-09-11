"""
Serve DavidAU/Qwen3.8-27B-...-NEO-CODER-MAX-MTP (IQ4_XS GGUF) on Modal via llama.cpp.

The GGUF declares ``general.architecture = qwen35``: a hybrid gated-delta-net/SSM model
(65 blocks, full attention every 4th layer) with an on-board MTP head
(``qwen35.nextn_predict_layers = 1``). Three consequences drive this file:

  1. Stock llama.cpp release binaries predate the ``qwen35`` arch, so we build from the
     pinned fork below, which carries the arch plus its CUDA kernels
     (gated_delta_net.cu / ssm-conv.cu / ssm-scan.cu).
  2. ``--spec-type draft-mtp`` turns the model's own MTP head into a draft model —
     self-speculative decoding with no second checkpoint and no vocab-match constraint.
  3. The chat template uses Jinja ``is undefined``, which only the minja path handles,
     so ``--jinja`` is mandatory or every /v1/chat/completions returns HTTP 500.

Usage:
    modal volume create oxidize-gguf                       # once
    modal volume put oxidize-gguf <local.gguf> /Qwen3.8-27B-MTP-IQ4_XS.gguf
    modal secret create llama-api-key LLAMA_API_KEY=<random>   # once; serve() refuses to start without it
    modal serve modal_qwen35_serve.py                      # ephemeral, live-reloading
    modal deploy modal_qwen35_serve.py                     # persistent URL

    OXIDIZE_MODAL_GPU=L40S modal deploy modal_qwen35_serve.py
"""
from __future__ import annotations

import os

import modal

# H100 (3.35 TB/s) is the "really quick" pick; L40S (864 GB/s) is the value pick.
# Weights are 15.9 GiB, so anything >= 24 GB holds the model; VRAM only buys context.
GPU = os.environ.get("OXIDIZE_MODAL_GPU", "H100")

# Pinned llama.cpp with qwen35 + MTP speculative decoding. Verified locally at this commit:
# `common_speculative_init_result: creating MTP draft context against the target model`.
LLAMA_CPP_REPO = os.environ.get(
    "OXIDIZE_LLAMA_CPP_REPO", "https://github.com/Jackson57279/llama.cpp.git"
)
LLAMA_CPP_REF = os.environ.get(
    "OXIDIZE_LLAMA_CPP_REF", "3c479e32ab26d657bd938d7f92d6ad05b6077e8a"
)

# Ada(89) covers L4/L40S, Hopper(90) covers H100/H200, Ampere(80) covers A100.
CUDA_ARCHS = os.environ.get("OXIDIZE_CUDA_ARCHS", "80;89;90")

MODEL_DIR = "/models"
MODEL_FILE = os.environ.get(
    "OXIDIZE_GGUF_NAME", "Qwen3.8-27B-MTP-IQ4_XS.gguf"
)
MODEL_PATH = f"{MODEL_DIR}/{MODEL_FILE}"

HF_REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
HF_FILE = "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf"

PORT = 8000

# Context. The hybrid arch makes long context cheap: only ~16 of 65 layers keep a growing
# KV cache (n_head_kv=4, head_dim=256, f16) => 4 KiB/layer/token => 64 KiB/token total.
# The other ~49 layers hold a fixed-size recurrent state. So 32k ctx ~= 2 GiB,
# 128k ~= 8 GiB, the full 262144 ~= 16 GiB.
N_CTX = int(os.environ.get("OXIDIZE_N_CTX", "32768"))
N_PARALLEL = int(os.environ.get("OXIDIZE_N_PARALLEL", "1"))
N_DRAFT = int(os.environ.get("OXIDIZE_N_DRAFT", "3"))

# This is a *reasoning* model: it emits a think block before the answer. Left unbounded, a
# short max_tokens spends the whole budget on thoughts and returns content="" with the text
# stranded in reasoning_content (observed locally: 64 tokens, all of it thinking).
# -1 = unrestricted, 0 = answer immediately, N > 0 = cap the think block at N tokens.
REASONING_BUDGET = int(os.environ.get("OXIDIZE_REASONING_BUDGET", "-1"))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "ccache")
    .run_commands(
        f"git clone {LLAMA_CPP_REPO} /opt/llama.cpp",
        f"cd /opt/llama.cpp && git checkout {LLAMA_CPP_REF}",
        # -DGGML_CUDA_FA_ALL_QUANTS lets --cache-type-k/v q8_0 work with flash attention.
        "cd /opt/llama.cpp && cmake -B build"
        " -DCMAKE_BUILD_TYPE=Release"
        " -DGGML_CUDA=ON"
        # Quoted: the ';' separators are CMake list syntax and would otherwise be
        # eaten by the build shell as command separators (exit 127).
        f' -DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCHS}"'
        " -DGGML_CUDA_FA_ALL_QUANTS=ON"
        " -DLLAMA_BUILD_TESTS=OFF"
        " -DLLAMA_BUILD_EXAMPLES=OFF"
        # llama-bench and llama-server live under tools/, gated by LLAMA_BUILD_TOOLS
        # (+ LLAMA_BUILD_SERVER for the server), not by LLAMA_BUILD_EXAMPLES. Both
        # default ON for a standalone build; pin them so the targets below always exist.
        " -DLLAMA_BUILD_TOOLS=ON"
        " -DLLAMA_BUILD_SERVER=ON"
        " -DLLAMA_CURL=ON",
        "cd /opt/llama.cpp && cmake --build build --config Release -j $(nproc) --target llama-server llama-bench",
        # No gpu= here: CMAKE_CUDA_ARCHITECTURES is explicit, so nvcc cross-compiles
        # without a device attached. Attaching one would just bill GPU-time for a build.
    )
    .pip_install("huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "LLAMA_CACHE": MODEL_DIR})
)

model_vol = modal.Volume.from_name("oxidize-gguf", create_if_missing=True)

app = modal.App("oxidize-qwen35-mtp")


@app.function(
    image=image,
    volumes={MODEL_DIR: model_vol},
    timeout=2 * 60 * 60,
    cpu=8.0,
    memory=32768,
    secrets=[modal.Secret.from_name("hf-token")],
)
def fetch_model() -> str:
    """Pull the GGUF straight from HF into the Volume (faster than uploading 16 GB from home)."""
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    dest = Path(MODEL_PATH)
    if dest.is_symlink():
        # Earlier versions moved the HF snapshot symlink here instead of the bytes.
        dest.unlink()
    if dest.exists():
        print(f"already present: {dest} ({dest.stat().st_size / 2**30:.2f} GiB)", flush=True)
        return str(dest)

    src = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        token=os.environ.get("HF_TOKEN"),
        cache_dir="/tmp/hf",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # hf_hub_download returns a snapshot symlink into the cache's blobs/ dir; moving it
    # across filesystems would recreate a dangling symlink on the Volume. Copy the bytes.
    # Copy to a temporary name and rename, so an interrupted copy never leaves
    # a truncated GGUF at MODEL_PATH for the exists() check above to accept.
    tmp = dest.with_name(dest.name + ".partial")
    shutil.copyfile(os.path.realpath(src), tmp)
    os.replace(tmp, dest)
    model_vol.commit()
    print(f"staged {dest} ({dest.stat().st_size / 2**30:.2f} GiB)", flush=True)
    return str(dest)


def _server_argv() -> list[str]:
    """llama-server flags, each earning its place. serve() appends --api-key."""
    return [
        "/opt/llama.cpp/build/bin/llama-server",
        "--model", MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        # All 65 blocks on the GPU. 15.9 GiB of weights fits any card we target.
        "--n-gpu-layers", "99",
        "--ctx-size", str(N_CTX),
        "--parallel", str(N_PARALLEL),
        # Self-speculative decoding off the model's own nextn head. The whole point of
        # the "-MTP" in the repo name; typically 1.5-2x on code/structured output.
        "--spec-type", "draft-mtp",
        "--draft-max", str(N_DRAFT),
        # Flash attention for the ~16 full-attention layers.
        "--flash-attn", "on",
        # Halves KV bytes/token (64 KiB -> 32 KiB) at negligible quality cost.
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        # Required: this model's chat template uses `is undefined`, which the legacy
        # template path rejects with HTTP 500.
        "--jinja",
        # Split the think block into message.reasoning_content so message.content holds
        # only the answer. Without this, clients see raw <think> tags inline.
        "--reasoning-format", "deepseek",
        "--reasoning-budget", str(REASONING_BUDGET),
        # Weights live on a Modal Volume (network FS); mmap'ing it would fault pages in
        # lazily over the network on every first touch. Read once, up front.
        "--no-mmap",
        # Large prompt batches keep the GPU saturated during prefill.
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--cont-batching",
        "--metrics",
        # Sampler defaults the GGUF itself ships (top_k=20, top_p=0.95, temp=1.0).
        "--temp", "1.0",
        "--top-k", "20",
        "--top-p", "0.95",
        "--min-p", "0.0",
    ]


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODEL_DIR: model_vol},
    timeout=24 * 60 * 60,
    cpu=8.0,
    memory=32768,
    scaledown_window=15 * 60,
    max_containers=1,
    secrets=[modal.Secret.from_name("llama-api-key")],
)
@modal.concurrent(max_inputs=max(N_PARALLEL, 1))
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve() -> None:
    """OpenAI-compatible endpoint: POST <url>/v1/chat/completions (Bearer $LLAMA_API_KEY)."""
    import subprocess

    api_key = os.environ.get("LLAMA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LLAMA_API_KEY is unset: refusing to expose llama-server unauthenticated. "
            "Create it with `modal secret create llama-api-key LLAMA_API_KEY=<random>`."
        )
    argv = _server_argv()
    # Log the flags before the key is added, so the secret never reaches the log.
    print(" ".join(argv), "--api-key <redacted>", flush=True)
    # The Modal web endpoint is public; require a bearer token on every request.
    subprocess.Popen([*argv, "--api-key", api_key])


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODEL_DIR: model_vol},
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
)
def bench(n_prompt: int = 512, n_gen: int = 128) -> str:
    """Ground-truth tok/s on the chosen GPU. Run before tuning anything."""
    import subprocess

    proc = subprocess.run(
        [
            "/opt/llama.cpp/build/bin/llama-bench",
            "-m", MODEL_PATH,
            "-ngl", "99",
            "-fa", "1",
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-r", "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited with code {proc.returncode}:\n{out}")
    return out


@app.local_entrypoint()
def main(prompt: str = "Write a Python one-liner that reverses a string.") -> None:
    """`modal run modal_qwen35_serve.py` -> stage the model, then report its size."""
    import json
    import shlex

    path = fetch_model.remote()
    print(f"model ready on volume: {path}")
    print("next: modal serve modal_qwen35_serve.py   (or modal deploy)")
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}]})
    print(
        'then: curl "$URL/v1/chat/completions"'
        ' -H "Authorization: Bearer $LLAMA_API_KEY"'
        " -H 'Content-Type: application/json'"
        f" -d {shlex.quote(payload)}"
    )
