# oxidize-c CUDA backend

GPU-resident decode for two families, dispatched on `general.architecture`:

- **gemma4** (`gemma*`) — interleaved SWA/global, GeGLU, rotoquant int4 KV option.
- **llama family** (`llama`/`mistral`/`qwen2`/`qwen3`/`yi`/`phi3`/... — dense) —
  GQA, both RoPE modes (NeoX split-half and ggml NORMAL adjacent-pair, per the
  model's rope convention), optional q/k/v/o biases, optional per-head q/k
  RMSNorm (qwen3), SwiGLU, tied or untied (`output.weight`) logits, f16 KV.
  MoE layers stay on the CPU (see "MoE" below).

Weight types (both families): F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, AL5_XS —
every one proven equal to the CPU forward (see "Correctness gate"). Anything else
is refused at init. The device dequant `dqv<T>` is shared
(`src/cuda/cuda_dequant.cuh`), so a type proven for one family is the same code
path for the other.

## Build / run

```bash
cd oxidize-c
make cuda                      # multi-arch fatbin -> ./oxidize-c-cuda
./oxidize-c-cuda --model ~/models/gemma-4-31B-it-AL5_XS.gguf \
    --prompt "hello" --max-tokens 128 --bench          # greedy by default
./oxidize-c-cuda ... --temp 0.8 --top-k 40             # sampled
./oxidize-c-cuda ... --gpus 2                          # gemma4 layer-split pipeline
./oxidize-c-cuda ... --ngl 30                          # first 30 layers on GPU
./oxidize-c-cuda --model ~/models/Qwen3-8B-Q4_K_M.gguf --prompt "hi"  # llama family
```

The arch is detected automatically: a `llama`/`mistral`/`qwen2`/`qwen3`/`yi`/
`phi3` GGUF runs the llama-family forward, a `gemma*` GGUF the gemma4 one.

`make cuda` builds SASS for sm_70/75/80/86/89/90 plus a compute_90 PTX entry:
one binary runs on T4, L4, A10G, A100, 4090 and H100. (It used to be
`-arch=sm_90a`, which cross-compiles cleanly everywhere and then dies at
RUNTIME on anything but an H100 with "no kernel image is available for
execution on the device" — a clean nvcc build proves nothing about what can
run.) One arch, faster compile: `make cuda NVCC_GENCODE="-arch=sm_80"`.

`make` / `make test` (CPU) are untouched.

## Correctness gate — `make cuda-test`

`tests/cuda_equiv.c` builds a synthetic gemma4 GGUF in memory (the CPU suites'
own builder), loads it twice, and runs the SAME token stream through
`gemma4_forward` (CPU) and the CUDA backend, comparing every logit. It covers
every accepted weight type, the tied Q4_K-weights/Q6_K-head layout a real
"Q4_K_M" download has, the int4 rotoquant KV cache, and `-ngl` at 1/2/3 of 3
layers. No local GPU here, so it runs on Modal:

```bash
modal run modal_cuda.py --action gputest            # T4 by default
modal run modal_cuda.py --action gputest --gpu L4
```

A kernel that has not been through this is not done. It is what caught the
four places this backend had silently drifted from `model_gemma4.c` (V copied
after k_norm instead of before, no scale-less V RMSNorm, no `rope_freqs` on
global layers, and `output_scale` applied to both residual adds instead of the
FFN sum) — none of which crash; they just produce confident wrong logits.

The gate covers BOTH families. The llama half builds synthetic dense GGUFs and
asserts CUDA == CPU `llama_forward` for every weight type, both RoPE modes
(`llama` NORMAL vs `qwen3` NeoX), with/without q/k/v/o biases, with/without
per-head q/k RMSNorm, tied and untied heads, and `-ngl` at 1/2/3 of 3 layers —
`argmax_mismatch == 0/12` is the hard gate on top of the logit tolerance.

The CPU reference is pinned to the exact f32 kernels (scalar/AVX2). On an
AVX-512-VNNI host `oc_matvec` routes K-quants through `oc_dot_row_q8`, an
int8-ACTIVATION approximation, whose error on this deliberately adversarial
fixture reaches 1.3e-3 (Q4_K) and 0.33 (Q6_K) relative — orders of magnitude
above the f32 paths. That is a property of the CPU fast path, not of the GPU;
`OC_ISA=avx512 ./build/oxidize-c-cuda-test` reproduces it.

## Architecture

- `src/cuda/gemma4_cuda.{h,cu}` — gemma4 kernels + host orchestration.
- `src/cuda/llama_cuda.{h,cu}` — llama-family kernels + host orchestration
  (single GPU; full-causal f16 KV; SwiGLU; both RoPE modes).
- `src/cuda/cuda_dequant.cuh` — shared device-side `dqv<T>` decoders.
- `src/main_cuda.c` — CLI, dispatches gemma4 vs llama on the arch string; uses
  the per-family chat template. `--gpus` (gemma4 layer-split), `--ngl`.
- Loading reuses the CPU loader `gemma4_load()` for every bit of GGUF
  parsing/geometry (SWA pattern, per-layer kv heads/head_dim, rope config,
  layer_output_scale, K=V-shared global layers). The CUDA module just uploads
  the still-quantized mmap bytes. Weights stay AL5_XS in VRAM (~13.45GB +
  ~0.6GB duplicated token_embd on the last GPU when `--gpus 2`).
  Side effect: the loader's host f32 KV caches (~2GB RAM) go unused here.

## Kernel inventory (one file, `gemma4_cuda.cu`)

| kernel | role |
|---|---|
| `dqv<T>` | device-side per-value dequant for F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS, ported from the scalar decoders in `quant.c` |
| `k_matvec<T>` | fused dequant matvec, warp/row, over `dqv<T>` |
| `k_matvec_al5xs` | hand-fused AL5_XS variant (in-register 3-bit unpack, w=(q-4)*scale); batched `nb<=8` columns for future speculative verify (decode uses nb=1) |
| `k_rmsnorm` | grid = n vectors (1 for residual stream, n_heads for per-head q/k norm); Gemma weights carry +1 |
| `k_rope` | NeoX split-half, freq = theta^(-2i/rope_len); partial rotary via rope_len; skipped at pos 0 |
| `k_kv_store` | f32 -> f16 into ring (SWA, cap=window=1024) / linear (global, cap=ctx) caches; same `pos % cache_cap` indexing as CPU |
| `k_attn` | one block per q head; scores in shared memory, softmax, weighted f16 V |
| `k_geglu`, `k_add`, `k_resid_out`, `k_softcap`, `k_embed<T>` | elementwise |
| `k_argmax_stage1/2` | device greedy sampling |

Per token everything is enqueued on one stream; the only sync + D2H copy is
the sampled token id (4 bytes, greedy) or the softcapped logits (sampling).
Prefill never syncs. Softcap is skipped on the greedy path (tanh is
monotonic, argmax unchanged).

## Multi-GPU choice

`--gpus N` is a **layer-split pipeline** (contiguous ranges, one 21KB
`cudaMemcpyPeerAsync` of the hidden state per boundary per token, event-
ordered). Chosen over row-split tensor parallel because it is trivially
correct without on-pod iteration (no per-matmul allreduce to get wrong).
Tradeoff: near-zero single-stream speedup — it buys memory headroom, not
tok/s. Since 13.45GB fits one 80GB H100, **`--gpus 1` is the perf
configuration**; upgrade path if >250 t/s must be exceeded is row-split
matvec + P2P allreduce over NVLink.

## Expected perf

Decode is weight-bandwidth-bound: 13.45GB / token ÷ 3.35TB/s (H100 SXM) ≈
4.0ms ≈ **~250 tok/s** ceiling. KV traffic is negligible at short contexts
(few MB/token). Main real-world tax: ~1000 kernel launches/token (~2µs each
enqueue) — async enqueue overlaps GPU work, but if measured decode sits well
under 200 t/s with nsys showing gaps, capture the token step in a **CUDA
graph** (the step is already a fixed kernel sequence per (pos-bucket); this
is the first optimization to reach for).

## Partial offload (`--ngl N`)

Layers `[0, N)` run on the GPU; the CPU picks the residual stream up with
`gemma4_forward_from(m, pos, N, ...)` and runs the rest plus the head. Only the
GPU's layers get VRAM (weights + KV), only the CPU's layers touch the CPU KV
cache, so the split is exact — `-ngl` at 1, 2 and 3 of 3 layers all pass the
equivalence gate. `--ngl` defaults to every layer.

## Debug plan

1. `modal run modal_cuda.py --action gputest` — the gate, first, always.
2. `compute-sanitizer ./oxidize-c-cuda --model ... --max-tokens 4` — memory
   errors in kernels.
3. `--bench` for tok/s; nsys profile if under target.

## Known risks / TODOs

- Launch overhead (see above) — CUDA graphs TODO if needed.
- Never benchmarked on an H100 with a real 31B model since the rewrite; the
  gate proves correctness, not tok/s.
- `dqv<T>` re-derives block scales per value (see the comment in the .cu). Fine
  while decode is bound on the weight stream; a prefill/batched kernel should
  stage blocks in shared memory instead.
- `k_attn` needs `(head_dim + count + 256)*4` bytes of shared memory; init
  raises the attribute for ctx > ~11k. Very large `--ctx` (>50k) would need a
  two-pass attention rewrite.
- Batched matvec (`nb>1`, gemma4) is plumbed but unused; the speculative-verify
  host path does not exist yet. The llama matvec is decode-only (`nb==1`).
- llama MoE (Mixtral / Qwen-MoE / DeepSeek dense-then-MoE) stays on the CPU: the
  routed SwiGLU-per-expert block is not on the GPU. `llama_cuda_init` refuses a
  model whose GPU-offloaded range `[0, ngl)` contains a MoE layer, naming the
  `--ngl` that keeps the MoE tail on the CPU (DeepSeek-style leading-dense models
  can offload their dense prefix that way). The dense path is the target.
- llama is single-GPU (`--gpus 1`). The gemma4 layer-split pipeline buys memory
  capacity, not tok/s, and the big MoE weights that need it are the CPU's anyway.
- qwen36 (DeltaNet) and deepseek (MLA) still have no GPU path.
- IQ / Q2_K / Q3_K / Q4_1 / Q5_0 / Q5_1 / BF16 weights are rejected at init by
  design: no kernel, and a wrong kernel is worse than none.
