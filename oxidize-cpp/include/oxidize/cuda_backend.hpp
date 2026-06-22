#pragma once
// CUDA GPU backend exposing the same dense-decode op set as
// include/oxidize/tensor.hpp, so model_llama can run on the GPU when the
// OXIDIZE_CUDA env var is set.
//
// Ported from (math + dispatch semantics):
//   oxidize-core/src/backends/cuda.rs  (cuBLAS GEMV/GEMM, on-the-fly quantized
//     GEMV with resident-weight caching, f16 GEMV kernel accumulating in f32)
//   oxidize-core/src/compute/flash_attention.rs  (online-softmax decode
//     attention, GQA grouping, scale 1/sqrt(head_dim))
//   oxidize-cpp tensor.hpp / tensor_cpu.cpp  (rms_norm, apply_rope, swiglu,
//     geglu, softmax scalar reference math this backend stays faithful to)
//
// The whole file is compiled only when OXIDIZE_CUDA is defined (Modal build with
// nvcc + CUDA 12.x targeting sm_80 / sm_90). On a host without CUDA the header
// is an empty translation unit and nothing here is referenced.

#ifdef OXIDIZE_CUDA

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "oxidize/config.hpp"

namespace oxidize {

// GPU compute backend. One instance owns a CUDA stream, a cuBLASLt handle, and a
// content-addressed cache of resident weight matrices (uploaded once, reused
// every decode step). All public methods take/return host pointers and perform
// the host<->device transfers internally; this keeps the call sites in
// model_llama identical to the CPU tensor.hpp free functions.
//
// FP8 (e4m3) GEMM is used for F16 weight matmuls on sm_90 (H100) via cuBLASLt
// when both operands fit the e4m3 dynamic range; otherwise the path falls back
// to an F16 GEMM (A100 / sm_80 always use F16). Quantized weights are
// dequantized on the GPU to F16 once and cached; the GEMV then accumulates in
// f32 to match the Rust numerics (cuBLAS Hgemm's f16 accumulation drifts).
class CudaBackend {
 public:
  CudaBackend();
  ~CudaBackend();

  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  // True if a CUDA device is present and the backend initialized successfully.
  // model_llama checks this before routing ops to the GPU.
  static bool available();

  // Singleton accessor (one backend per process; the resident-weight cache is
  // keyed by host pointer + content hash so it is safe to share across layers).
  static CudaBackend& instance();

  // --- Op set mirroring tensor.hpp ------------------------------------------

  // out[i] = x[i] * inv_rms * scale_i  (see tensor.hpp::rms_norm).
  void rms_norm(float* out, const float* x, const float* weight, size_t n,
                float eps, bool weight_plus_one);

  // In-place partial RoPE over [num_heads * head_dim] (see tensor.hpp).
  void apply_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                  float theta, size_t rope_dim);

  // out[i] = silu(gate[i]) * up[i]; out may alias gate.
  void swiglu_inplace(float* gate, const float* up, float* out, size_t n);

  // out[i] = gelu_tanh(gate[i]) * up[i].
  void geglu_inplace(float* gate, const float* up, float* out, size_t n);

  // In-place numerically-stable softmax.
  void softmax_inplace(float* x, size_t n);

  // y = W * x ; W row-major [rows x cols] (f32 weights, cached resident).
  void matvec(float* y, const float* W, const float* x, size_t rows,
              size_t cols);

  // y = dequant(W) * x ; W is a packed block sequence of `quant` rows.
  void gemv_quantized(float* y, QuantType quant, const uint8_t* W, size_t rows,
                      size_t cols, const float* x);

  // Single-step GQA decode attention over a KV cache (see tensor.hpp).
  void attention_decode(float* out, const float* q, const float* k_cache,
                        const float* v_cache, size_t seq_len, size_t num_heads,
                        size_t kv_heads, size_t head_dim);

  // F16 GEMM: C[m x n] = A[m x k] * B[k x n], all row-major host f32 buffers.
  // Uses FP8 (e4m3) accumulation on sm_90 when enabled, else F16. Provided for
  // prefill / batched paths; the dense decode hot path uses matvec / gemv.
  void gemm(float* C, const float* A, const float* B, size_t m, size_t k,
            size_t n);

  // Argmax over `n` logits with total_cmp-style tie-break (first max wins),
  // computed on the GPU. Mirrors sampler.hpp::greedy semantics.
  uint32_t argmax(const float* logits, size_t n);

  // --- GPU-resident decode (WIP / UNVERIFIED) -------------------------------
  // The op set above bounces every activation host<->device and syncs per op
  // (~290 round-trips/token) — that, not the math, is why --cuda is slow. The
  // resident path keeps activations + KV cache on the device for the whole
  // forward and syncs once per token. Weights (host pointers) are uploaded and
  // cached on first use, identical to the op-set caches.
  //
  // A weight matrix as the model sees it: either f32 (use `f32`) or a packed
  // quantized / native-f16 block sequence (use `quant` + `data`).
  struct WeightView {
    bool quantized = false;
    QuantType quant = QuantType::F32;
    const uint8_t* data = nullptr;  // quantized blocks (host)
    const float* f32 = nullptr;     // f32 weights (host)
    size_t rows = 0;
    size_t cols = 0;
  };
  struct LayerView {
    const float* attn_norm = nullptr;
    WeightView wq, wk, wv, wo;
    const float* wq_bias = nullptr;
    const float* wk_bias = nullptr;
    const float* wv_bias = nullptr;
    const float* wo_bias = nullptr;
    size_t q_bias_len = 0, k_bias_len = 0, v_bias_len = 0, o_bias_len = 0;
    const float* attn_q_norm = nullptr;  // per-head (Qwen3), len head_dim or q_len
    const float* attn_k_norm = nullptr;
    const float* ffn_norm = nullptr;
    WeightView gate, up, down;
    const float* down_bias = nullptr;
    size_t down_bias_len = 0;
    const float* post_attn_norm = nullptr;  // Gemma sandwich
    const float* post_ffn_norm = nullptr;
    float rope_theta = 10000.0f;
    size_t sliding_window = 0;  // 0 = full attention
  };
  struct ModelView {
    InferenceConfig cfg;
    const float* final_norm = nullptr;
    WeightView lm_head;  // (tied embeddings already resolved by the caller)
    std::vector<LayerView> layers;
  };

  // Allocate device activation + KV buffers for `mv` dims (idempotent; only
  // reallocates when dims grow). Safe to call once after model load.
  void resident_setup(const ModelView& mv);

  // Run one decode step on the GPU. `embed_row` is the (already dequantized and
  // embedding-scaled) hidden-size embedding for `token`; `logits_out` receives
  // `cfg.vocab_size` f32 logits. One host<->device sync per call.
  void resident_forward(const ModelView& mv, const float* embed_row, size_t pos,
                        float* logits_out);

 private:
  // Reusable per-call device scratch (separate x / weight / out arenas so the
  // three operands of one op never alias).
  float* vec_buf_x(size_t n);
  float* vec_buf_w(size_t n);
  float* out_buf_(size_t n);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace oxidize

#endif  // OXIDIZE_CUDA
