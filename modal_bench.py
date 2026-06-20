"""
Cross-engine single-stream decode benchmark on one Modal A100.

Same base model (Mistral-7B-Instruct-v0.3), ~4-bit, batch=1, greedy, 128 tokens:
  - oxidize    : Q4_K_M GGUF  (run via modal_app.py, gpu-tps --gpu A100)
  - llama.cpp  : Q4_K_M GGUF  (llama-cpp-python, CUDA)   <- apples-to-apples vs oxidize
  - vLLM       : AWQ 4-bit    (production reference ceiling)
  - sglang     : AWQ 4-bit    (production reference ceiling)

Usage:
  modal run modal_bench.py --engine llamacpp
  modal run modal_bench.py --engine vllm
  modal run modal_bench.py --engine sglang
"""

import modal

GPU = "A100"
PROMPT = "Explain what a tokenizer does in two sentences."
MAX_TOKENS = 128

GGUF_REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
GGUF_FILE = "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
AWQ_REPO = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"

hf_cache = modal.Volume.from_name("bench-hf-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"

app = modal.App("oxidize-xengine-bench")

# --- llama.cpp (CUDA) ---
llamacpp_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("huggingface_hub")
    .run_commands(
        "pip install llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
    )
)

# --- vLLM ---
vllm_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.6.6", "transformers==4.47.1", "huggingface_hub"
)

# NOTE: sglang removed from this file — its pinned flashinfer dep won't pip-install
# and Modal builds every image in the app eagerly, blocking the others. Add it back
# via its official lmsysorg/sglang image in a separate file if needed.


@app.function(image=llamacpp_image, gpu=GPU, volumes={HF_CACHE: hf_cache}, timeout=1800)
def bench_llamacpp():
    import time
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(GGUF_REPO, GGUF_FILE, cache_dir=HF_CACHE)
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    llm("warmup", max_tokens=8)  # warm CUDA graphs / kv
    best = 0.0
    for _ in range(3):
        t = time.time()
        out = llm(PROMPT, max_tokens=MAX_TOKENS, temperature=0.0)
        dt = time.time() - t
        n = out["usage"]["completion_tokens"]
        best = max(best, n / dt)
    txt = out["choices"][0]["text"][:120].replace("\n", " ")
    r = f"llama.cpp (Q4_K_M GGUF) A100: {best:.1f} tok/s | \"{txt}\""
    print(r, flush=True)
    return r


@app.function(image=vllm_image, gpu=GPU, volumes={HF_CACHE: hf_cache}, timeout=1800)
def bench_vllm():
    import time
    from vllm import LLM, SamplingParams

    llm = LLM(model=AWQ_REPO, quantization="awq", max_model_len=4096,
              gpu_memory_utilization=0.9, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    llm.generate(["warmup"], sp)  # warmup (CUDA graph capture)
    best = 0.0
    for _ in range(3):
        t = time.time()
        out = llm.generate([PROMPT], sp)
        dt = time.time() - t
        n = len(out[0].outputs[0].token_ids)
        best = max(best, n / dt)
    txt = out[0].outputs[0].text[:120].replace("\n", " ")
    r = f"vLLM (AWQ 4-bit) A100: {best:.1f} tok/s | \"{txt}\""
    print(r, flush=True)
    return r


@app.local_entrypoint()
def main(engine: str = "llamacpp"):
    if engine == "llamacpp":
        print(bench_llamacpp.remote())
    elif engine == "vllm":
        print(bench_vllm.remote())
    else:
        raise SystemExit("engine must be: llamacpp | vllm")
