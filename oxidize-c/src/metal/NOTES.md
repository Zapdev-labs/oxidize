# Metal backend — NOTES

**UNVERIFIED.** Nothing in this directory has been compiled or run on real
hardware in the authoring environment. It was written blind against
`src/cuda/` (the verified resident-forward reference) and
`oxidize-core/src/backends/metal.rs`. It requires **macOS + Xcode** (`xcrun
metal`, Metal + Foundation frameworks) and an Apple GPU. **It MAY NOT COMPILE.**
No equivalence gate (CUDA's is `tests/cuda_equiv.c`) has ever been run against
it. Do not trust logits until a macOS validator proves them.

## Inventory

| File | Role |
|------|------|
| `metal_dequant.h` | MSL `dqv<T>` / `dh` / `ksm` (port of `cuda_dequant.cuh`) |
| `gemma4.metal` | `gk_*` decode kernels (f16 KV path) |
| `llama.metal` | `lk_*` decode kernels (both RoPE modes) |
| `metal_common.h` / `.mm` | Device/queue/library, buffers, PSO cache, encode, one commit+wait |
| `gemma4_metal.h` / `.mm` | API mirroring `gemma4_cuda.h`; layer graph; refuses multi-GPU / rotoquant / ngl 0 |
| `llama_metal.h` / `.mm` | API mirroring `llama_cuda.h`; refuses MoE in offload range; both RoPE modes |
| `Makefile` | Opt-in only — never part of the default `oxidize-c` build |
| `NOTES.md` | This file |

## Build (macOS only)

```bash
cd oxidize-c/src/metal
make
```

Produces:

- `build/oxidize.metallib` — combined `gk_*` + `lk_*`
- `build/metal_common.o`, `build/gemma4_metal.o`, `build/llama_metal.o`

Link a macOS binary with `-framework Metal -framework Foundation` and the
host `.o` files plus the rest of oxidize-c. At runtime set:

```bash
export OXIDIZE_METAL_LIB=/absolute/path/to/oxidize.metallib
```

or place the metallib at `src/metal/build/oxidize.metallib` relative to cwd.

## Design (mirrors CUDA)

- Weights stay **quantized** in shared `MTLBuffer`s; MSL fuses `dqv<T>` with
  matvec (same eight types as CUDA: F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/AL5_XS).
- One `MTLCommandBuffer` recorded per token; **one** `commit` +
  `waitUntilCompleted` when host results are needed (logits / hidden / argmax).
- Prefill with all outputs NULL commits without waiting.
- Hybrid `-ngl`: GPU layers `[0, ngl)`, then `gemma4_forward_from` /
  `llama_forward_from` on the CPU.

## Hard refuses

- `--gpus > 1` (no layer-split pipeline on Apple Silicon).
- Gemma4 `kv_quant` / rotoquant int4 KV (no `gk_fht` / `gk_attn_q4` in MSL).
- `--ngl 0` (pure-CPU path).
- Unsupported weight quants.
- Llama: any MoE layer inside the offload range.

## Known-blind risks (not exhaustive)

1. **`setBytes` vs `constant T&`** — each scalar constant is a separate buffer
   index; alignment / packing may differ from what the Metal shader compiler
   expects on some OS versions.
2. **Threadgroup memory for `gk_attn` / `lk_attn`** — large `--ctx` may exceed
   device threadgroup limits; CUDA raises dynamic smem attributes; Metal has
   no equivalent attribute call here.
3. **K=V blit mid-encoder** — gemma4 ends the compute encoder, blits K→V, then
   starts a new compute encoder on the same command buffer. Plausible but
   untested.
4. **Metallib / PSO lookup** — `host_name` strings must match the MSL
   `[[host_name(...)]]` instantiations exactly.
5. **ARC / `__bridge_retained` lifetime** — easy to get wrong; never validated.
6. **No wiring into `main.c` / Makefile of oxidize-c** — this tree is
   self-contained and intentionally unplugged from the default build.

## Honesty

This is a **production-shaped stub**: complete host orchestration matching the
CUDA API surface and layer graphs, but **unverified**. Shipping it as "works
on Mac" without a macOS equivalence run would be a lie.
