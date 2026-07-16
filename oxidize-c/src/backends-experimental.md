# Experimental GPU backends (UNVERIFIED)

Four blind ports live under `src/{mlx,metal,vulkan,webgpu}/`. They are **not**
part of `make` / `make test` / `make asan`. Each has its own Makefile and
`NOTES.md`. **None have been compiled or run in this environment.**

| Backend | Dir | Toolchain (on real hardware) | Status |
|---------|-----|------------------------------|--------|
| MLX | `src/mlx/` | macOS Apple Silicon + mlx-c | UNVERIFIED — host dequant to F32, then MLX ops |
| Metal | `src/metal/` | macOS + Xcode (`xcrun metal`) | UNVERIFIED — fused MSL `dqv` matvec like CUDA |
| Vulkan | `src/vulkan/` | Vulkan 1.1+ SDK + `glslangValidator` | UNVERIFIED — GLSL compute + libvulkan host |
| WebGPU | `src/webgpu/` | Emscripten/Dawn + WebGPU browser | UNVERIFIED — WGSL + `webgpu.h` / WASM |

The **verified** GPU path is CUDA only (`make cuda`, `make cuda-test`, see
`CUDA.md`). These four mirror that resident-forward shape for gemma4 + llama
family, but until an equivalence gate like `tests/cuda_equiv.c` is green on
real hardware, treat every logit as unproven.

```bash
make -C src/mlx      # needs MLX_INCLUDE / MLX_LIB
make -C src/metal    # needs Xcode Metal
make -C src/vulkan   # needs Vulkan SDK
make -C src/webgpu   # help; `make -C src/webgpu wasm` needs EMSDK
```

Do not link these into the default `oxidize-c` binary until validated.
