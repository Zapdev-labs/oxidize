# MLX backend (UNVERIFIED)

**Status: NEVER compiled or run.** Written blind against the verified CUDA
backend (`src/cuda/gemma4_cuda.cu`, `llama_cuda.cu`) and
`oxidize-core/src/backends/mlx.rs`. Requires a Mac with Apple Silicon and a
working **mlx-c** install. It **may not compile**. Do not claim correctness
until a hardware validator has:

1. Fixed every assumed mlx-c signature against the installed `<mlx/c/*.h>`.
2. Linked a custom binary (this directory is **not** part of default `make` /
   `make test`).
3. Proven logit equivalence against `gemma4_forward` / `llama_forward` (CPU),
   the same bar as `tests/cuda_equiv.c` for CUDA.

---

## Inventory

| File | Role |
|------|------|
| `mlx_backend.h` | Public API mirroring `gemma4_cuda.h` / `llama_cuda.h` |
| `mlx_common.h` / `mlx_common.c` | Shared mlx-c wrappers (matvec, rmsnorm, rope, SDPA, GeGLU/SwiGLU, softcap, KV append, host dequant upload) |
| `gemma4_mlx.c` | Gemma 4 resident forward + hybrid `--ngl` via `gemma4_forward_from` |
| `llama_mlx.c` | Llama-family dense forward + hybrid via `llama_forward_from` |
| `Makefile` | Opt-in object build only (`MLX_INCLUDE` / `MLX_LIB`) |
| `NOTES.md` | This file |

---

## Design (KEY DIFFERENCE from CUDA)

CUDA keeps GGUF-quantized weights in VRAM and fuses ggml dequant into the
matvec (`dqv<T>` in `cuda_dequant.cuh`). MLX has **no** ggml k-quant / AL5_XS
decode on device (`mlx_quantize` is a different affine layout). This port
therefore:

- Host-dequantizes every weight with `oc_dequant_row()` → f32.
- Uploads f32 arrays into unified memory via `mlx_array_new_data`.
- Supports the **same** types CUDA gates: F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS
  (`mlx_check_type`). Refuses anything else loudly.

MLX-native requant (`mlx.rs` `from_gguf_tensor_quantized`) is **not**
implemented — memory optimization left stubbed.

---

## Behaviour mirrored from CUDA / CPU

### Gemma 4 (`gemma4_mlx.c`)

- Refuse `n_gpus != 1`, refuse `--ngl 0`.
- Layer loop matches `model_gemma4.c` / `gemma4_cuda.cu`:
  - **V before k_norm**: when `attn_v` absent, V = raw K projection copy.
  - **Scale-less V RMSNorm** (ones weight), no rope on V.
  - **`rope_freqs` on global layers only** (SWA: empty freqs).
  - **`resid_out = (ffn + attn_out) * output_scale`** (not two scaled adds).
  - Sandwich norms, GeGLU, tied head, final softcap when full stack.
- Hybrid: GPU layers `[0, ngl)` then `gemma4_forward_from`.
- Embedding via host `mlx_embed_row` × `emb_scale`.

### Llama (`llama_mlx.c`)

- Refuse `n_gpus != 1`, refuse `--ngl 0`.
- Refuse any **MoE** layer inside `[0, ngl)`.
- GQA, both RoPE modes (`rope_norm` → mlx `traditional`), optional q/k/v/o
  biases, optional q/k norms, SwiGLU, tied/untied head.
- Hybrid via `llama_forward_from`. No softcap / emb scale.

---

## Build commands (Mac only)

```bash
# Adjust paths to your mlx-c install:
export MLX_INCLUDE=/opt/homebrew/include   # or /usr/local/include
export MLX_LIB=/opt/homebrew/lib

cd oxidize-c/src/mlx
make print-flags
make
# → mlx_common.o gemma4_mlx.o llama_mlx.o
make clean
```

Default oxidize-c `make` / `make test` **must not** pick these up. Linking into
a binary is manual (see `Makefile` sketch).

---

## Known risks (honest)

1. **Every mlx-c symbol is assumed** (`mlx_matmul`, `mlx_fast_rms_norm`,
   `mlx_fast_rope`, `mlx_fast_scaled_dot_product_attention`,
   `mlx_default_gpu_stream_new`, `mlx_stream_free`, `mlx_array_new_data`,
   dtype enums, status-return vs handle-return). Fix in `mlx_common.c` first.
2. **RoPE + `rope_freqs` mapping is the least-certain piece** (gemma global
   layers). mlx `fast_rope` freqs semantics may not match
   `angle /= rope_freqs[i]`.
3. **SWA window slice** uses negative indices on axis 2 — confirm `mlx_slice`
   semantics; growing KV caches never ring (memory grows with seq).
4. **No rotoquant KV** — if the model requested `kv_quant`, MLX uses f32
   caches (warns at init); not bit-equal to CUDA kv_quant path.
5. **Host-dequant entire matrices** — large unified-memory footprint vs CUDA
   quantized VRAM; may OOM models that fit quantized on GPU.
6. **FAIL_EMPTY paths leak arrays** on error; fine for bring-up, not for prod.
7. **GQA broadcast / SDPA mask** — decode uses empty mask; rely on MLX to
   broadcast `n_head != n_kv`. Untested.
8. **Never verified** — no compile, no run, no equiv test in this environment.

---

## What a validator should do first

1. Diff `<mlx/c/*.h>` against every call in `mlx_common.c`; patch wrappers.
2. `make -C src/mlx` until objects build.
3. Wire a throwaway binary + tiny F32 GGUF; compare one token’s logits to CPU.
4. Only then expand to quantized weights / SWA / hybrid `--ngl`.
