# WebGPU backend — NOTES

**Status: UNVERIFIED.** Nothing under `src/webgpu/` has been compiled or run
against real weights. No claim of numerical agreement with the CPU or CUDA
forwards is made. Treat every line as a blind structural port.

## What this is

A WebGPU compute backend for oxidize-c that mirrors the CUDA resident-forward
shape (`gemma4_cuda` / `llama_cuda`):

- CPU loaders (`gemma4_load` / `llama_load`) own GGUF parsing and geometry.
- Weights stay GGUF-quantized on the GPU; WGSL `dqv()` (from `prelude.wgsl`)
  fuses dequant into matvec/embed — same idea as `cuda_dequant.cuh`.
- One device→host sync per token when logits/hidden/argmax are requested.
- KV cache is **f16** (packed two halves per `u32`), matching CUDA — not the
  Vulkan port’s FP32 divergence.

## What it is not

- **Not** wired into `oxidize-c/Makefile` or the CLI. Build only via
  `src/webgpu/Makefile`.
- **Not** multi-GPU. `--gpus != 1` is refused (no peer-copy pipeline).
- **Not** MoE-capable on the GPU. A MoE layer inside `[0, ngl)` is refused
  (same policy as `llama_cuda`).
- **Not** rotoquant KV. `m->kv_quant` is refused; use the f16 path.
- **Not** validated. There is no `tests/webgpu_equiv.c` yet.

## Layout

| Path | Role |
|------|------|
| `wgsl/prelude.wgsl` | Byte helpers + `dqv()` (prepended to matvec/embed) |
| `wgsl/*.wgsl` | Compute kernels |
| `webgpu_common.h/.c` | Dawn/emdawn host (ctx, buffers, pipelines, dispatch) |
| `gemma4_webgpu.h/.c` | Gemma 4 forward (GeGLU, softcap, argmax, sandwich norms) |
| `llama_webgpu.h/.c` | Llama-family forward (SwiGLU, biases, NeoX/NORMAL rope) |
| `Makefile` | Isolated; `make` prints help; `make wasm` needs `$EMSDK` |
| `index.html` | Minimal browser harness for the wasm smoke test |
| `wasm_smoke.c` | Adapter probe only |

## Build

```bash
cd oxidize-c/src/webgpu
make              # help only
make check-env
# after: source $EMSDK/emsdk_env.sh
make wasm         # -> build/oxidize_webgpu.js
# serve this directory and open index.html (not file://)
```

Native Dawn linking is intentionally unspecified — set `WEBGPU_CFLAGS` /
`WEBGPU_LDFLAGS` if you have a Dawn install. Full model objects also need the
rest of oxidize-c (`quant.o`, `model_*.o`, …); this Makefile does not pull them
in by default.

Shader directory override: `OXIDIZE_WEBGPU_WGSL=/path/to/wgsl`.

## Supported quants

Same gate as CUDA: `F32 / F16 / Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K / AL5_XS`.
Anything else fails loudly at init.

## Known blind risks (honesty list)

1. **Dawn/emdawn API drift.** `webgpu.h` callback-info shapes
   (`WGPURequestAdapterCallbackInfo`, `WGPUStringView`, etc.) vary by revision.
   Expect compile fixes on first real build.
2. **Uniform overwrite.** Dispatches use a pool of small UBOs (one per launch
   in a token). Pool exhaustion (`UBO_POOL=1024`) on huge stacks is possible.
3. **f16 pack/unpack.** `pack2x16float` / `unpack2x16float` for KV may not match
   CUDA `__float2half` rounding. Odd KV row lengths are refused.
4. **K=V gemma layers.** Re-run the K matvec into `v` instead of a D2D copy;
   should be equal, untested.
5. **Attention scratch.** Scores live in a global SSBO (like Vulkan), not
   dynamic shared memory. `head_dim > 256` refused.
6. **One compute pass per dispatch.** Correct ordering, likely slow.
7. **`wgpu_rec_submit_wait`.** Idle wait is a spin/`emscripten_sleep` pump —
   may return before GPU work finishes on some backends; downloads use
   `mapAsync` which should synchronize, but prefill-without-download is soft.
8. **Browser limits.** Large models will hit buffer/storage caps long before
   CUDA VRAM would.

## References used (read-only)

- `oxidize-c/src/cuda/*`
- `oxidize-c/src/vulkan/{vk_common.*,shaders/*}`
- `oxidize-c/src/webgpu/wgsl/*` (pre-existing prelude/matvec/embed/rmsnorm/rope)
- `oxidize-core/src/backends/webgpu.rs` (feature gating only; not a full forward)
