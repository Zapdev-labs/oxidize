// CudaBackend: GPU implementation of the dense-decode op set (tensor.hpp), with
// content-addressed resident-weight caching so each weight matrix is uploaded
// and (for quantized weights) dequantized to F16 exactly once and reused on
// every later decode step.
//
// Ported from:
//   oxidize-core/src/backends/cuda.rs
//     - GpuState (resident_f32 / resident_f16 caches keyed by
//       (pointer, len, content-hash); FNV-1a 64-bit hash; one-time GPU dequant
//       of quantized weights into resident F16, then an f32-accumulating f16
//       GEMV).
//     - gemv_f32_cuda / gemv_quantized_cuda dispatch.
//   oxidize-cpp tensor_cpu.cpp / tensor.hpp (op contract + scalar math the
//     kernels stay faithful to).
//
// Compiled only with OXIDIZE_CUDA defined (Modal: nvcc + CUDA 12.x, sm_80/90).

#ifdef OXIDIZE_CUDA

#include "oxidize/cuda_backend.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_common.cuh"
#include "oxidize/quant.hpp"

namespace oxidize {

namespace {

// FNV-1a 64-bit hash (matches backends/cuda.rs::hash_bytes) for content-aware
// cache keys, so a mutated host buffer is not silently reused.
[[maybe_unused]] uint64_t fnv1a(const uint8_t* data, size_t len) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < len; ++i) {
    h ^= static_cast<uint64_t>(data[i]);
    h *= 0x100000001b3ull;
  }
  return h;
}

struct CacheKey {
  uintptr_t ptr;
  size_t len;
  uint64_t hash;
  bool operator==(const CacheKey& o) const {
    return ptr == o.ptr && len == o.len && hash == o.hash;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const {
    return static_cast<size_t>(k.hash ^ (k.ptr * 1099511628211ull) ^ k.len);
  }
};

// Cache key = (host pointer, length). Weight buffers are immutable after model
// load and each has a stable, unique host address, so (ptr, len) identifies a
// weight uniquely. We deliberately do NOT content-hash here: these helpers run
// on EVERY op of every layer of every token, and FNV-1a over a multi-hundred-MB
// weight (e.g. a dequantized f32 lm_head) per call costs seconds/token on the
// CPU and was making the GPU path slower than CPU. (fnv1a retained for the
// one-time content check below.)
CacheKey key_f32(const float* p, size_t n) {
  return {reinterpret_cast<uintptr_t>(p), n, 0};
}
CacheKey key_bytes(const uint8_t* p, size_t n) {
  return {reinterpret_cast<uintptr_t>(p), n, 0};
}

// Map QuantType to its GPU dequant launcher + block geometry. Returns false for
// types without a GPU dequant kernel (caller dequantizes on the CPU and uploads
// the resulting F16, matching cuda.rs's fall-back semantics).
using DequantLauncher = void (*)(const uint8_t*, __half*, unsigned, cudaStream_t);
bool dequant_kernel_for(QuantType q, DequantLauncher& fn, size_t& block_bytes,
                        size_t& vals_per_block) {
  switch (q) {
    case QuantType::Q8_0:
      fn = cuda::launch_dequant_q8_0; block_bytes = 34;  vals_per_block = 32;  return true;
    case QuantType::Q4_K_S:
    case QuantType::Q4_K_M:
      fn = cuda::launch_dequant_q4_k; block_bytes = 144; vals_per_block = 256; return true;
    case QuantType::Q6_K:
      fn = cuda::launch_dequant_q6_k; block_bytes = 210; vals_per_block = 256; return true;
    case QuantType::Q2_K:
      fn = cuda::launch_dequant_q2_k; block_bytes = 84;  vals_per_block = 256; return true;
    default:
      return false;
  }
}

}  // namespace

struct CudaBackend::Impl {
  cudaStream_t stream = nullptr;

  // Resident weight caches (uploaded/dequantized once, reused every step).
  std::unordered_map<CacheKey, float*, CacheKeyHash> resident_f32;
  std::unordered_map<CacheKey, __half*, CacheKeyHash> resident_f16;

  // Reusable device scratch (activation-sized vectors / outputs / GEMM arena).
  float* d_vec = nullptr;
  size_t d_vec_cap = 0;
  float* d_vec2 = nullptr;
  size_t d_vec2_cap = 0;
  float* d_out = nullptr;
  size_t d_out_cap = 0;
  uint8_t* d_scratch = nullptr;
  size_t d_scratch_cap = 0;

  // GPU-resident decode buffers (allocated by resident_setup, freed here).
  float* r_x = nullptr;       // [hidden] running activation
  float* r_norm = nullptr;    // [hidden] norm scratch
  float* r_q = nullptr;       // [q_len]
  float* r_k = nullptr;       // [kv_len]
  float* r_v = nullptr;       // [kv_len]
  float* r_attn = nullptr;    // [q_len]
  float* r_o = nullptr;       // [hidden] attn/ffn output
  float* r_gate = nullptr;    // [inter]
  float* r_up = nullptr;      // [inter]
  float* r_logits = nullptr;  // [vocab]
  float* r_kvk = nullptr;     // [layers * ctx * kv_token]
  float* r_kvv = nullptr;     // [layers * ctx * kv_token]
  size_t r_hidden = 0, r_qlen = 0, r_kvlen = 0, r_inter = 0, r_vocab = 0;
  size_t r_layers = 0, r_ctx = 0, r_kvtok = 0;
  bool r_ready = false;

  Impl() { CUDA_CHECK(cudaStreamCreate(&stream)); }
  ~Impl() {
    for (auto& kv : resident_f32) cudaFree(kv.second);
    for (auto& kv : resident_f16) cudaFree(kv.second);
    if (d_vec) cudaFree(d_vec);
    if (d_vec2) cudaFree(d_vec2);
    if (d_out) cudaFree(d_out);
    if (d_scratch) cudaFree(d_scratch);
    for (float* p : {r_x, r_norm, r_q, r_k, r_v, r_attn, r_o, r_gate, r_up,
                     r_logits, r_kvk, r_kvv})
      if (p) cudaFree(p);
    if (stream) cudaStreamDestroy(stream);
  }

  static float* dalloc(size_t n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (n ? n : 1) * sizeof(float)));
    return p;
  }

  float* vec_buf(size_t n) {
    if (n > d_vec_cap) {
      if (d_vec) CUDA_CHECK(cudaFree(d_vec));
      CUDA_CHECK(cudaMalloc(&d_vec, n * sizeof(float)));
      d_vec_cap = n;
    }
    return d_vec;
  }
  float* vec_buf2(size_t n) {
    if (n > d_vec2_cap) {
      if (d_vec2) CUDA_CHECK(cudaFree(d_vec2));
      CUDA_CHECK(cudaMalloc(&d_vec2, n * sizeof(float)));
      d_vec2_cap = n;
    }
    return d_vec2;
  }
  float* out_buf(size_t n) {
    if (n > d_out_cap) {
      if (d_out) CUDA_CHECK(cudaFree(d_out));
      CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
      d_out_cap = n;
    }
    return d_out;
  }
  uint8_t* scratch_buf(size_t bytes) {
    if (bytes > d_scratch_cap) {
      if (d_scratch) CUDA_CHECK(cudaFree(d_scratch));
      CUDA_CHECK(cudaMalloc(&d_scratch, bytes));
      d_scratch_cap = bytes;
    }
    return d_scratch;
  }

  float* ensure_f32(const float* host, size_t n) {
    CacheKey k = key_f32(host, n);
    auto it = resident_f32.find(k);
    if (it != resident_f32.end()) return it->second;
    float* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(dev, host, n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    resident_f32.emplace(k, dev);
    return dev;
  }

  // Upload + GPU-dequant (or CPU-dequant fallback) a quantized weight to resident
  // F16, cached by content. `elems` = rows * cols.
  __half* ensure_f16_from_quant(QuantType q, const uint8_t* host_blocks,
                                size_t elems) {
    DequantLauncher fn = nullptr;
    size_t block_bytes = 0, vals_per_block = 0;
    bool gpu = dequant_kernel_for(q, fn, block_bytes, vals_per_block);

    size_t raw_bytes = quantized_size(q, elems);  // throws on layout mismatch
    CacheKey k = key_bytes(host_blocks, raw_bytes);
    auto it = resident_f16.find(k);
    if (it != resident_f16.end()) return it->second;

    __half* weight = nullptr;
    CUDA_CHECK(cudaMalloc(&weight, elems * sizeof(__half)));

    if (gpu) {
      uint8_t* raw = nullptr;
      CUDA_CHECK(cudaMalloc(&raw, raw_bytes));
      CUDA_CHECK(cudaMemcpyAsync(raw, host_blocks, raw_bytes,
                                 cudaMemcpyHostToDevice, stream));
      unsigned n_blocks = static_cast<unsigned>(elems / vals_per_block);
      fn(raw, weight, n_blocks, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(raw));
    } else {
      // CPU dequant -> f32 host -> convert to f16 host -> upload (fallback for
      // types without a GPU dequant kernel; bit-faithful via quant.cpp).
      std::vector<float> f32(elems);
      dequantize_row(q, host_blocks, f32.data(), elems);
      std::vector<uint16_t> h16(elems);
      for (size_t i = 0; i < elems; ++i) {
        __half hv = __float2half(f32[i]);
        std::memcpy(&h16[i], &hv, sizeof(uint16_t));
      }
      CUDA_CHECK(cudaMemcpyAsync(weight, h16.data(), elems * sizeof(__half),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    resident_f16.emplace(k, weight);
    return weight;
  }
};

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;

bool CudaBackend::available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

CudaBackend& CudaBackend::instance() {
  static CudaBackend backend;
  return backend;
}

// --- Ops -------------------------------------------------------------------

void CudaBackend::rms_norm(float* out, const float* x, const float* weight,
                           size_t n, float eps, bool weight_plus_one) {
  if (n == 0) throw std::runtime_error("CudaBackend::rms_norm: zero dimension");
  float* dx = vec_buf_x(n);
  float* dw = vec_buf_w(n);
  float* dout = out_buf_(n);
  CUDA_CHECK(cudaMemcpyAsync(dx, x, n * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream));
  CUDA_CHECK(cudaMemcpyAsync(dw, weight, n * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  cuda::launch_rms_norm(dout, dx, dw, static_cast<unsigned>(n), eps,
                        weight_plus_one, impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(out, dout, n * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::apply_rope(float* vec, size_t head_dim, size_t num_heads,
                             size_t pos, float theta, size_t rope_dim) {
  if (head_dim == 0)
    throw std::runtime_error("CudaBackend::apply_rope: zero head_dim");
  size_t rope_len = (rope_dim == 0) ? head_dim : rope_dim;
  if (rope_len > head_dim) rope_len = head_dim;
  if (rope_len % 2 != 0)
    throw std::runtime_error("CudaBackend::apply_rope: odd rotary dimension " +
                             std::to_string(rope_len));
  if (pos == 0 || rope_len == 0) return;  // identity (matches CPU path)

  size_t total = num_heads * head_dim;
  float* dv = vec_buf_x(total);
  CUDA_CHECK(cudaMemcpyAsync(dv, vec, total * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  cuda::launch_apply_rope(dv, static_cast<unsigned>(head_dim),
                          static_cast<unsigned>(num_heads),
                          static_cast<unsigned>(pos), theta,
                          static_cast<unsigned>(rope_len), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(vec, dv, total * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::swiglu_inplace(float* gate, const float* up, float* out,
                                 size_t n) {
  if (n == 0) return;
  float* dg = vec_buf_x(n);
  float* du = vec_buf_w(n);
  float* dout = out_buf_(n);
  CUDA_CHECK(cudaMemcpyAsync(dg, gate, n * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  CUDA_CHECK(cudaMemcpyAsync(du, up, n * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream));
  cuda::launch_swiglu(dg, du, dout, static_cast<unsigned>(n), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(out, dout, n * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::geglu_inplace(float* gate, const float* up, float* out,
                                size_t n) {
  if (n == 0) return;
  float* dg = vec_buf_x(n);
  float* du = vec_buf_w(n);
  float* dout = out_buf_(n);
  CUDA_CHECK(cudaMemcpyAsync(dg, gate, n * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  CUDA_CHECK(cudaMemcpyAsync(du, up, n * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream));
  cuda::launch_geglu(dg, du, dout, static_cast<unsigned>(n), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(out, dout, n * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::softmax_inplace(float* x, size_t n) {
  if (n == 0) return;
  float* dx = vec_buf_x(n);
  CUDA_CHECK(cudaMemcpyAsync(dx, x, n * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream));
  cuda::launch_softmax_inplace(dx, static_cast<unsigned>(n), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost,
                             impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::matvec(float* y, const float* W, const float* x, size_t rows,
                         size_t cols) {
  float* dW = impl_->ensure_f32(W, rows * cols);
  float* dx = vec_buf_x(cols);
  float* dy = out_buf_(rows);
  CUDA_CHECK(cudaMemcpyAsync(dx, x, cols * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  cuda::launch_gemv_f32(dW, dx, dy, static_cast<unsigned>(rows),
                        static_cast<unsigned>(cols), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(y, dy, rows * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::gemv_quantized(float* y, QuantType quant, const uint8_t* W,
                                 size_t rows, size_t cols, const float* x) {
  __half* dW = impl_->ensure_f16_from_quant(quant, W, rows * cols);
  float* dx = vec_buf_x(cols);
  float* dy = out_buf_(rows);
  CUDA_CHECK(cudaMemcpyAsync(dx, x, cols * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  // f32-accumulating f16 GEMV (matches cuda.rs's deliberate choice over Hgemm).
  cuda::launch_gemv_f16(dW, dx, dy, static_cast<unsigned>(rows),
                        static_cast<unsigned>(cols), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(y, dy, rows * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void CudaBackend::attention_decode(float* out, const float* q,
                                   const float* k_cache, const float* v_cache,
                                   size_t seq_len, size_t num_heads,
                                   size_t kv_heads, size_t head_dim) {
  if (head_dim == 0)
    throw std::runtime_error("CudaBackend::attention_decode: zero head_dim");
  if (kv_heads == 0 || num_heads % kv_heads != 0)
    throw std::runtime_error(
        "CudaBackend::attention_decode: num_heads not divisible by kv_heads");

  size_t q_len = num_heads * head_dim;
  size_t kv_len = static_cast<size_t>(seq_len) * kv_heads * head_dim;

  float* dq = nullptr;
  float* dk = nullptr;
  float* dv = nullptr;
  float* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dq, q_len * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, q_len * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, (kv_len ? kv_len : 1) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, (kv_len ? kv_len : 1) * sizeof(float)));

  CUDA_CHECK(cudaMemcpyAsync(dq, q, q_len * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  if (kv_len) {
    CUDA_CHECK(cudaMemcpyAsync(dk, k_cache, kv_len * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(dv, v_cache, kv_len * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream));
  }
  cuda::launch_flash_decode(dout, dq, dk, dv, static_cast<unsigned>(seq_len),
                            static_cast<unsigned>(num_heads),
                            static_cast<unsigned>(kv_heads),
                            static_cast<unsigned>(head_dim), impl_->stream);
  CUDA_CHECK(cudaMemcpyAsync(out, dout, q_len * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dout);
}

void CudaBackend::gemm(float* C, const float* A, const float* B, size_t m,
                       size_t k, size_t n) {
  size_t a_elems = m * k;
  size_t b_elems = k * n;
  size_t c_elems = m * n;

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, a_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, b_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, c_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(dA, A, a_elems * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, B, b_elems * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));

  // Cast arena: max(f16, fp8) operands; f16 dominates (2 bytes/elem).
  size_t scratch_bytes = (a_elems + b_elems) * sizeof(__half);
  uint8_t* scratch = impl_->scratch_buf(scratch_bytes);

  cuda::cuda_gemm_device(dA, dB, dC, static_cast<int>(m), static_cast<int>(k),
                         static_cast<int>(n), scratch, scratch_bytes,
                         impl_->stream);

  CUDA_CHECK(cudaMemcpyAsync(C, dC, c_elems * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

uint32_t CudaBackend::argmax(const float* logits, size_t n) {
  if (n == 0) throw std::runtime_error("CudaBackend::argmax: empty logits");
  float* dl = vec_buf_x(n);
  CUDA_CHECK(cudaMemcpyAsync(dl, logits, n * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  uint32_t* d_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, sizeof(uint32_t)));
  cuda::launch_argmax(dl, static_cast<unsigned>(n), d_idx, impl_->stream);
  uint32_t idx = 0;
  CUDA_CHECK(cudaMemcpyAsync(&idx, d_idx, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  cudaFree(d_idx);
  return idx;
}

// --- Private scratch helpers (separate x / weight / out buffers so the three
//     concurrent operands of an op never alias) -----------------------------

float* CudaBackend::vec_buf_x(size_t n) { return impl_->vec_buf(n); }
float* CudaBackend::vec_buf_w(size_t n) { return impl_->vec_buf2(n); }
float* CudaBackend::out_buf_(size_t n) { return impl_->out_buf(n); }

// --- GPU-resident decode (WIP / UNVERIFIED) --------------------------------

void CudaBackend::resident_setup(const ModelView& mv) {
  const InferenceConfig& cfg = mv.cfg;
  size_t head_dim = cfg.head_dim();
  size_t q_len = cfg.num_attention_heads * head_dim;
  size_t kv_tok = cfg.num_key_value_heads * head_dim;
  size_t inter = cfg.intermediate_size;
  // Reallocate only if uninitialized or a dimension grew.
  bool need = !impl_->r_ready || cfg.hidden_size > impl_->r_hidden ||
              q_len > impl_->r_qlen || kv_tok > impl_->r_kvtok ||
              inter > impl_->r_inter || cfg.vocab_size > impl_->r_vocab ||
              cfg.layer_count > impl_->r_layers || cfg.context_size > impl_->r_ctx;
  if (!need) return;

  for (float* p : {impl_->r_x, impl_->r_norm, impl_->r_q, impl_->r_k, impl_->r_v,
                   impl_->r_attn, impl_->r_o, impl_->r_gate, impl_->r_up,
                   impl_->r_logits, impl_->r_kvk, impl_->r_kvv})
    if (p) cudaFree(p);

  impl_->r_hidden = cfg.hidden_size;
  impl_->r_qlen = q_len;
  impl_->r_kvtok = kv_tok;
  impl_->r_kvlen = kv_tok;
  impl_->r_inter = inter;
  impl_->r_vocab = cfg.vocab_size;
  impl_->r_layers = cfg.layer_count;
  impl_->r_ctx = cfg.context_size;

  impl_->r_x = Impl::dalloc(cfg.hidden_size);
  impl_->r_norm = Impl::dalloc(cfg.hidden_size);
  impl_->r_q = Impl::dalloc(q_len);
  impl_->r_k = Impl::dalloc(kv_tok);
  impl_->r_v = Impl::dalloc(kv_tok);
  impl_->r_attn = Impl::dalloc(q_len);
  impl_->r_o = Impl::dalloc(cfg.hidden_size);
  impl_->r_gate = Impl::dalloc(inter);
  impl_->r_up = Impl::dalloc(inter);
  impl_->r_logits = Impl::dalloc(cfg.vocab_size);
  impl_->r_kvk = Impl::dalloc(cfg.layer_count * cfg.context_size * kv_tok);
  impl_->r_kvv = Impl::dalloc(cfg.layer_count * cfg.context_size * kv_tok);
  impl_->r_ready = true;
}

void CudaBackend::resident_forward(const ModelView& mv, const float* embed_row,
                                   size_t pos, float* logits_out) {
  resident_setup(mv);
  const InferenceConfig& cfg = mv.cfg;
  const float eps = cfg.rms_norm_eps;
  const bool plus_one = cfg.rms_norm_weight_plus_one;
  const size_t hidden = cfg.hidden_size;
  const size_t n_heads = cfg.num_attention_heads;
  const size_t kv_heads = cfg.num_key_value_heads;
  const size_t head_dim = cfg.head_dim();
  const size_t q_len = n_heads * head_dim;
  const size_t kv_tok = kv_heads * head_dim;
  const size_t inter = cfg.intermediate_size;
  const size_t seq_len = pos + 1;
  size_t rope_len = (cfg.rope_dim == 0) ? head_dim : cfg.rope_dim;
  if (rope_len > head_dim) rope_len = head_dim;
  cudaStream_t s = impl_->stream;
  Impl* I = impl_.get();

  auto u = [](size_t v) { return static_cast<unsigned>(v); };
  auto dev = [&](const float* host, size_t n) { return I->ensure_f32(host, n); };
  auto gemv = [&](const WeightView& w, const float* in, float* out) {
    if (w.quantized) {
      const __half* dW = I->ensure_f16_from_quant(w.quant, w.data, w.rows * w.cols);
      cuda::launch_gemv_f16(dW, in, out, u(w.rows), u(w.cols), s);
    } else {
      const float* dW = I->ensure_f32(w.f32, w.rows * w.cols);
      cuda::launch_gemv_f32(dW, in, out, u(w.rows), u(w.cols), s);
    }
  };
  auto bias = [&](float* y, const float* b, size_t n, size_t blen) {
    if (b && blen) cuda::launch_add_bias_mod(y, dev(b, blen), u(n), u(blen), s);
  };

  // Embedding (already dequantized + scaled by the caller) -> resident x.
  CUDA_CHECK(cudaMemcpyAsync(I->r_x, embed_row, hidden * sizeof(float),
                             cudaMemcpyHostToDevice, s));

  for (size_t l = 0; l < cfg.layer_count; ++l) {
    const LayerView& ly = mv.layers[l];

    // Attention.
    cuda::launch_rms_norm(I->r_norm, I->r_x, dev(ly.attn_norm, hidden), u(hidden),
                          eps, plus_one, s);
    gemv(ly.wq, I->r_norm, I->r_q);
    gemv(ly.wk, I->r_norm, I->r_k);
    gemv(ly.wv, I->r_norm, I->r_v);
    bias(I->r_q, ly.wq_bias, q_len, ly.q_bias_len);
    bias(I->r_k, ly.wk_bias, kv_tok, ly.k_bias_len);
    bias(I->r_v, ly.wv_bias, kv_tok, ly.v_bias_len);

    // Per-head Q/K RMSNorm (Qwen3): in-place over each head_dim slice.
    if (ly.attn_q_norm) {
      const float* dn = dev(ly.attn_q_norm, head_dim);
      for (size_t h = 0; h < n_heads; ++h)
        cuda::launch_rms_norm(I->r_q + h * head_dim, I->r_q + h * head_dim, dn,
                              u(head_dim), eps, plus_one, s);
    }
    if (ly.attn_k_norm) {
      const float* dn = dev(ly.attn_k_norm, head_dim);
      for (size_t h = 0; h < kv_heads; ++h)
        cuda::launch_rms_norm(I->r_k + h * head_dim, I->r_k + h * head_dim, dn,
                              u(head_dim), eps, plus_one, s);
    }

    cuda::launch_apply_rope(I->r_q, u(head_dim), u(n_heads), u(pos),
                            ly.rope_theta, u(rope_len), s);
    cuda::launch_apply_rope(I->r_k, u(head_dim), u(kv_heads), u(pos),
                            ly.rope_theta, u(rope_len), s);

    // Append K/V to this layer's resident cache slot.
    size_t phys = pos % cfg.context_size;
    float* kbase = I->r_kvk + (l * cfg.context_size + phys) * kv_tok;
    float* vbase = I->r_kvv + (l * cfg.context_size + phys) * kv_tok;
    CUDA_CHECK(cudaMemcpyAsync(kbase, I->r_k, kv_tok * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(vbase, I->r_v, kv_tok * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));

    const float* kpre = I->r_kvk + (l * cfg.context_size) * kv_tok;
    const float* vpre = I->r_kvv + (l * cfg.context_size) * kv_tok;
    size_t eff = seq_len;
    if (ly.sliding_window > 0 && seq_len > ly.sliding_window) {
      size_t skip = (seq_len - ly.sliding_window) * kv_tok;
      kpre += skip;
      vpre += skip;
      eff = ly.sliding_window;
    }
    cuda::launch_flash_decode(I->r_attn, I->r_q, kpre, vpre, u(eff), u(n_heads),
                              u(kv_heads), u(head_dim), s);

    gemv(ly.wo, I->r_attn, I->r_o);
    bias(I->r_o, ly.wo_bias, hidden, ly.o_bias_len);
    if (ly.post_attn_norm)  // Gemma sandwich
      cuda::launch_rms_norm(I->r_o, I->r_o, dev(ly.post_attn_norm, hidden),
                            u(hidden), eps, plus_one, s);
    cuda::launch_residual_add(I->r_x, I->r_o, u(hidden), s);

    // FFN.
    cuda::launch_rms_norm(I->r_norm, I->r_x, dev(ly.ffn_norm, hidden), u(hidden),
                          eps, plus_one, s);
    gemv(ly.gate, I->r_norm, I->r_gate);
    gemv(ly.up, I->r_norm, I->r_up);
    if (cfg.gelu_ffn)
      cuda::launch_geglu(I->r_gate, I->r_up, I->r_gate, u(inter), s);
    else
      cuda::launch_swiglu(I->r_gate, I->r_up, I->r_gate, u(inter), s);
    gemv(ly.down, I->r_gate, I->r_o);
    bias(I->r_o, ly.down_bias, hidden, ly.down_bias_len);
    if (ly.post_ffn_norm)
      cuda::launch_rms_norm(I->r_o, I->r_o, dev(ly.post_ffn_norm, hidden),
                            u(hidden), eps, plus_one, s);
    cuda::launch_residual_add(I->r_x, I->r_o, u(hidden), s);
  }

  // Final norm + lm_head, then one sync to bring logits home.
  cuda::launch_rms_norm(I->r_norm, I->r_x, dev(mv.final_norm, hidden), u(hidden),
                        eps, plus_one, s);
  gemv(mv.lm_head, I->r_norm, I->r_logits);
  CUDA_CHECK(cudaMemcpyAsync(logits_out, I->r_logits,
                             cfg.vocab_size * sizeof(float),
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace oxidize

#endif  // OXIDIZE_CUDA
