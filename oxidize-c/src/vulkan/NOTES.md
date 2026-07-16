# UNVERIFIED Vulkan backend — NOTES

**Status: UNVERIFIED.** Nothing under `src/vulkan/` has been compiled or run
against a real Vulkan driver in this work. The host forwards are structurally
complete mirrors of `src/cuda/gemma4_cuda.*` / `llama_cuda.*`, written blind.
They may not compile, may mis-barrier, may mis-encode push constants, and may
produce wrong numerics. Treat every claim below as aspirational.

## What this is

A compute-shader decode path for oxidize-c:

| Piece | Role |
|-------|------|
| `vk_common.[ch]` | Instance/device/queue, DEVICE_LOCAL buffers, upload/download, SPIR-V pipelines, per-token `rec_begin → dispatch → submit_wait` |
| `gemma4_vk.[ch]` | Gemma 4 host graph (GeGLU, sandwich norms, SWA window, softcap, hybrid `-ngl`) |
| `llama_vk.[ch]` | Llama-family dense host graph (both RoPE modes, biases, q/k norms, SwiGLU, MoE refuse) |
| `shaders/*.comp` | SPIR-V sources (plus `dequant.glsl` included by matvec/embed) |
| `Makefile` | **Only** build entry — never wired into the top-level oxidize-c Makefile |

## Build (opt-in only)

```bash
# from oxidize-c/
make -C src/vulkan shaders   # needs glslangValidator or glslc
make -C src/vulkan host      # needs vulkan headers + -lvulkan
```

Host code loads `.spv` from `$OXIDIZE_VK_SHADERS` (default `src/vulkan/shaders`
relative to process cwd; the Makefile exports an absolute path when you build
from here).

There is **no** default-build hook. Linking into `oxidize-c` / CLI is future
work and deliberately out of scope here.

## API (mirrors CUDA)

- `gemma4_vk_init/free/forward/step` — same contracts as `gemma4_cuda_*`
- `llama_vk_init/free/step` — same contracts as `llama_cuda_*`
- Hybrid `-ngl`: GPU runs layers `[0, ngl)`, copies residual to `m->x`, CPU
  finishes via `*_forward_from()`

## Hard refusals

| Condition | Why |
|-----------|-----|
| `--gpus > 1` | Multi-GPU layer-split not ported |
| `--ngl 0` | Pure-CPU path; no handle |
| Bad weight quant | Only F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS (`vk_qidx`) |
| `m->kv_quant` (gemma4 rotoquant) | No FHT / int4 KV shaders |
| MoE layer in `[0, ngl)` (llama) | No expert dispatch |
| `head_dim` / `v_head_dim` > 256 | `attn.comp` shared `sq[256]` |

## Known divergences from CUDA (honest)

1. **KV cache is FP32**, not f16. `kv_store.comp` / `attn.comp` store and read
   floats. Same math, ~2× KV VRAM vs CUDA. (User-facing note: rotoquant is
   refused; the non-rotoquant path here is FP32 rather than CUDA's f16.)
2. **Always one fence wait per token**, including prefill with no D2H. CUDA can
   leave work async on a stream; Vulkan has nothing to order the next token
   against unless we submit.
3. **Attention scores** live in a global scratch SSBO (`attn_scratch`), not
   dynamic shared memory (GLSL fixed shared arrays).
4. **Coarse whole-pipeline barriers** between every dispatch (false
   dependencies; may cost tok/s).
5. **No multi-GPU**, no peer copies, no rotoquant, no MoE.
6. **Descriptor pool** is sized generously and reset once per token — untested
   against real set counts for deep models / large vocab.

## Push-constant / binding contracts

Must match the `.comp` files exactly (see `vk_kernels_init` comments). If a
shader is edited, update the host packs in `gemma4_vk.c` / `llama_vk.c` and the
`create_pipe(..., pushsize, nbufs, ...)` calls in `vk_common.c`.

## Risks (non-exhaustive)

- Memory-type selection may fail on some iGPUs (DEVICE_LOCAL + HOST_VISIBLE split).
- `std430` uint packing in `dequant.glsl` may not match GGUF row layout.
- Embed `row_off` is a byte offset into a `uint[]` buffer — alignment/endian
  assumptions are untested.
- RoPE `has_freqs` still binds a freqs buffer; dummy binding when unused.
- Argmax is a single 256-lane workgroup scan (fine for typical vocab; may be
  slow or wrong for pathological sizes).
- No validation layers were run. No numerical gate like `tests/cuda_equiv.c`.

## What “done” means here

Host forwards are **structurally complete**: init uploads DEVICE_LOCAL weights,
per-token recording dispatches the shader graph, hybrid step calls the CPU
tail. **Correctness is unverified.** Do not ship, do not default-build, do not
claim parity with CUDA until an equivalence test exists.
