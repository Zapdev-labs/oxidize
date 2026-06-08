# Running 70B Models on 4GB GPUs — Architecture Overview

## The Problem

A 70B parameter model at FP16 requires **140 GB VRAM** — impossible on a 4GB GPU.  
Even at Q8_0 (1 byte/param) it's **70 GB** — still too large for most GPUs.

## The Solution: AirLLM-Style Layer Streaming

We split the problem into two parts:

### 1. Aggressive Quantization (CPU RAM Storage)

| Format | Bytes/Param | 70B Model Size | Location |
|--------|-------------|----------------|----------|
| FP16 | 2.0 | 140 GB | ❌ Too big |
| Q8_0 | 1.06 | ~74 GB | ✅ CPU RAM |
| Q4_0 | 0.56 | ~39 GB | ✅ CPU RAM |
| Q4_K | 0.56 | ~39 GB | ✅ CPU RAM |

**Key insight**: The model lives in CPU RAM in compressed form. Only the active layer is decompressed on the GPU.

### 2. Layer-by-Layer Streaming (GPU Compute)

```
CPU RAM (70B model in Q4_0 = 39 GB)
    │
    │  Copy Layer 0 weights (~2.5 GB Q4_0)
    ▼
GPU VRAM (4 GB)
    ├── Layer 0 weights: 2.5 GB
    ├── Activations: 0.5 GB
    └── Scratch: 0.5 GB
    │
    │  Compute Layer 0
    ▼
    Free Layer 0 weights
    │
    │  Copy Layer 1 weights
    ▼
    Compute Layer 1
    ...
```

**Per-layer VRAM footprint**: ~2-4 GB (compressed weights + activations)

## Implementation Details

### On-the-Fly GEMV Kernels (No F16 Materialization)

Traditional approach:
1. Upload Q8_0 bytes to GPU
2. Dequantize to F16 (2× size expansion)
3. Run F16 GEMV
4. Keep F16 resident for reuse

**Problem**: Step 2 turns 1 GB of Q8_0 into 2 GB of F16. On a 4GB GPU you can only fit 1-2 layers.

Our approach:
1. Upload Q8_0 bytes to GPU
2. **Run GEMV kernel that reads Q8_0 blocks directly**
3. Each thread dequantizes its own block: `value = scale * int8`
4. No F16 buffer ever allocated
5. Free Q8_0 bytes immediately after compute

**VRAM stays at compressed size throughout**.

### CUDA Kernels Added

| Kernel | Format | Bytes/Param | Materialization |
|--------|--------|-------------|-----------------|
| `gemv_f32_kernel` | F32 | 4.0 | N/A |
| `gemv_f16_kernel` | F16 | 2.0 | None (direct) |
| `gemv_q8_0_kernel` | Q8_0 | 1.06 | **None** (on-the-fly) |
| `gemv_q4_0_kernel` | Q4_0 | 0.56 | **None** (on-the-fly) |
| `dequant_q8_0_kernel` | Q8_0→F16 | 1.06→2.0 | Yes (legacy path) |

### Layer Management API

```rust
use oxidize_core::cuda::{set_layer_config, preload_layer, evict_layer, CudaLayerConfig};

// Set budget: keep only 2 layers in VRAM at once
set_layer_config(CudaLayerConfig {
    max_resident_layers: 2,
    max_vram_bytes: 0,
}).unwrap();

// Before each layer's forward pass:
preload_layer(layer_idx, &[
    (&q_proj_weights, hidden, hidden),
    (&k_proj_weights, kv_dim, hidden),
    (&v_proj_weights, kv_dim, hidden),
    (&o_proj_weights, hidden, hidden),
    (&ffn_gate, intermediate, hidden),
    (&ffn_up, intermediate, hidden),
    (&ffn_down, hidden, intermediate),
]).unwrap();

// Compute layer...

// Automatic: oldest layer is evicted when over budget
```

### VRAM Budget Math (Llama-70B on 4GB GPU)

**Model specs**:
- Layers: 80
- Hidden: 8192
- Intermediate: 28672
- Heads: 64

**Per-layer weight count**:
- Q/K/V/O proj: 4 × 8192 × 8192 = 268M params
- FFN gate/up/down: 2 × 28672 × 8192 + 8192 × 28672 = 704M params
- **Total**: ~972M params/layer

**Per-layer VRAM at Q4_0**:
- 972M params × 0.56 bytes = **544 MB compressed**
- Activations: ~100 MB
- Scratch: ~100 MB
- **Total per layer: ~750 MB**

**With 2-layer cache**:
- 2 × 750 MB = **1.5 GB** (fits in 4GB with room to spare)
- 78 layers in CPU RAM = **39 GB total** (reasonable for modern workstations)

### Throughput Expectations

**Single-token latency** (one full forward pass):
1. Upload layer 0: ~544 MB PCIe copy = **5-10 ms**
2. Compute layer 0: ~1 ms (GPU GEMV)
3. Free layer 0: ~0.1 ms
4. Repeat for 80 layers

**Total**: 80 × (7 ms) = **560 ms/token**  
**Throughput**: ~**1.8 tok/s**

**Optimizations**:
- **Async prefetch**: Upload layer N+1 while computing layer N (overlaps PCIe and compute)
- **Pinned memory**: Use CUDA pinned host memory for faster PCIe transfers
- **Kernel fusion**: Fuse RMSNorm + GEMV to reduce kernel launch overhead
- **Expected with async**: ~400-500 ms/token = **2-2.5 tok/s**

### Comparison Table

| Setup | VRAM Required | Speed | Notes |
|-------|--------------|-------|-------|
| FP16 full model | 140 GB | ~30 tok/s | Requires A100/H100 |
| Q8_0 full model | 74 GB | ~20 tok/s | Requires A100 80GB |
| **Q4_0 layer-stream** | **4 GB** | **~2 tok/s** | **Fits RTX 3050, GTX 1650** |
| CPU only | 0 GB | ~0.1 tok/s | Reference baseline |

## Files Changed

| File | Purpose |
|------|---------|
| `oxidize-core/kernels/gemv_f32.cu` | New on-the-fly Q8_0 and Q4_0 GEMV kernels |
| `oxidize-core/src/backends/cuda.rs` | Layer management API, LRU eviction, VRAM tracking |
| `oxidize-core/src/compute/tensor.rs` | Dispatch to on-the-fly kernels |
| `oxidize-core/build.rs` | nvcc PTX compilation at build time |

## How to Use

### 1. Convert your model to Q4_0

```bash
sfw cargo run -p oxidize-quantize -- \
  --input model-f32.gguf \
  --output model-q4_0.gguf \
  --source F32 --target Q4_0
```

### 2. Run with layer streaming

```bash
sfw cargo run -p oxidize-cli --features oxidize-core/cuda -- \
  --model model-q4_0.gguf \
  --prompt "Hello, world!" \
  --max-tokens 100
```

The inference engine will automatically:
1. Load the model metadata
2. Keep quantized weights in CPU RAM
3. Stream one layer at a time to GPU
4. Use on-the-fly Q4_0 GEMV kernels
5. Evict completed layers immediately

### 3. Tune the layer cache

For more VRAM headroom or faster multi-token generation:

```rust
// In your code before inference:
oxidize_core::cuda::set_layer_config(
    oxidize_core::cuda::CudaLayerConfig {
        max_resident_layers: 1,  // Minimal VRAM
        max_vram_bytes: 3 * 1024 * 1024 * 1024, // 3 GB cap
    }
).unwrap();
```

## Future Optimizations

1. **Async prefetch**: Double-buffer layer uploads with CUDA streams
2. **Flash Attention**: Replace O(n²) attention with O(n) for long contexts
3. **Speculative decoding**: Use small draft model to reduce forward passes
4. **CPU+GPU hybrid**: Run attention on GPU, FFN on CPU in parallel
5. **INT4x2 packing**: Pack 2 4-bit values per byte for even smaller transfers
