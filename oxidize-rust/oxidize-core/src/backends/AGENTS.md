# oxidize-core/src/backends/

**Generated:** 2026-06-03
**Domain:** Hardware compute backends (CUDA, Metal, Vulkan, WebGPU, MLX, Strix)

## OVERVIEW
Hardware-accelerated compute backend implementations. Each backend implements the `ComputeBackend` trait. Only Vulkan has a stub twin (`vulkan_stub.rs`) for compile-time fallback; all others compile-gate with `#[cfg(...)]`.

## STRUCTURE
```
backends/
├── cuda.rs         # CUDA kernels via cudarc, async stream management
├── metal.rs        # Apple Metal GPU backend
├── mlx.rs          # Apple MLX framework (macOS-only, full file gated)
├── strix.rs        # Strix custom accelerator backend
├── vulkan.rs       # Vulkan compute shaders (full impl)
├── vulkan_stub.rs  # No-op Vulkan for non-Vulkan builds
└── webgpu.rs       # WebGPU compute for WASM/browser targets
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add backend | `backend.rs` root + new `xxx.rs` | Implement `ComputeBackend`, add `XxxBuildInfo` |
| CUDA kernel launch | `cuda.rs` | `cudarc` for kernel loading, async stream sync |
| Metal shader mgmt | `metal.rs` | MTLDevice/MTLCommandBuffer, pipeline caching |
| Vulkan fallback | `vulkan_stub.rs` | Only backend with stub twin pattern |
| WebGPU in browser | `webgpu.rs` | `wasm-bindgen`, `web-sys` compute pass |
| MLX macOS path | `mlx.rs` | Every item `#[cfg(target_os = "macos")]`, no stubs |
| Backend feature gate | `Cargo.toml` | Add `"cuda"`, `"metal"`, etc. per backend |

## CONVENTIONS
- **Build info micro-pattern**: Every backend exposes `XxxBuildInfo` + `xxx_build_info()` for compile-time feature detection.
- **Trait boundary**: `ComputeBackend` in `backend.rs` is the single interface. All backends implement `allocate()`, `copy()`, `execute_kernel()`.
- **Async by default**: CUDA/Metal backends expose async methods; CPU fallback is synchronous.
- **Safety**: GPU backends contain `unsafe` blocks for FFI/raw pointer ops; keep isolated to backend files.

## ANTI-PATTERNS
- `vulkan.rs` + `vulkan_stub.rs` is the only backend with a stub twin — new backends should use `#[cfg(...)]` gating instead.
- Hardcoded device index (0) in CUDA init — should accept `CUDA_VISIBLE_DEVICES` or config.
- Metal shader compilation at runtime without caching — compile once, reuse pipeline state.

## BUILD FEATURES
| Feature | Backend | Platform |
|---------|---------|----------|
| `cuda` | NVIDIA CUDA | Linux/Windows |
| `metal` | Apple Metal | macOS/iOS |
| `vulkan` | Vulkan | Cross-platform |
| `webgpu` | WebGPU | WASM/Browser |
| `mlx` | Apple MLX | macOS-only |
| `strix` | Strix | Custom hardware |
