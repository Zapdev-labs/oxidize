#pragma once
// Llama-family DENSE inference hot path.
//
// Ported from:
//   oxidize-core/src/model/inference.rs
//     - InferenceModel::load_from_gguf  (weight loading / layout)
//     - InferenceModel::forward_single + embed_token_into_workspace
//       + run_layer_range_in_workspace  (per-layer dense decode)
//     - InferenceModel::final_head_from_workspace (final norm + lm_head)
//     - impl Model for InferenceModel    (forward / forward_many)
//   oxidize-core/src/compute/kv_cache.rs (F32 layer-major KV cache + RoPE
//       absolute positions, GQA token layout [layer][position][head][head_dim])
//
// Numerically faithful to the Rust SCALAR reference path: f32 accumulation,
// per-layer rms_norm -> QKV (gemv_quantized) -> per-head RoPE -> KV append ->
// GQA causal attention (online softmax, scale 1/sqrt(head_dim)) -> output proj
// -> residual -> rms_norm -> SwiGLU/GeGLU FFN -> residual; final norm -> lm_head.
//
// Out-of-Phase-1 architectures (MoE / MLA / shortconv / Mamba /
// sliding-window-interleave) throw std::runtime_error. Dense Llama / Mistral /
// Qwen3-dense / Gemma-dense are fully implemented, including GQA, partial RoPE,
// Qwen (1+w) RMSNorm, Gemma GeGLU + embedding scale, tied embeddings, and
// uniform sliding-window (Mistral/Qwen) attention masking.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "oxidize/config.hpp"
#include "oxidize/gguf.hpp"
#include "oxidize/model.hpp"
#include "oxidize/quant.hpp"
#ifdef OXIDIZE_CUDA
#include "oxidize/cuda_backend.hpp"
#endif

namespace oxidize {

// Storage for one weight matrix: either dequantized f32, or a borrowed pointer
// into the mmap'd GGUF data kept in its quantized block layout (decoded on the
// fly by gemv_quantized). Mirrors WeightStorage::{F32, MmapQuantized}.
struct LlamaWeight {
  bool quantized = false;
  QuantType quant = QuantType::F32;
  const uint8_t* data = nullptr;   // valid when quantized: mmap-backed blocks
  std::vector<float> f32;          // valid when !quantized
  size_t rows = 0;                 // output features
  size_t cols = 0;                 // input features

  bool empty() const { return quantized ? (data == nullptr) : f32.empty(); }
};

// Per-layer dense weights (Llama/Mistral/Qwen/Gemma). Mirrors the dense subset
// of inference.rs::LayerWeights.
struct LlamaLayer {
  std::vector<float> attn_norm;
  LlamaWeight attn_q;
  std::vector<float> attn_q_bias;
  LlamaWeight attn_k;
  std::vector<float> attn_k_bias;
  LlamaWeight attn_v;
  std::vector<float> attn_v_bias;
  LlamaWeight attn_output;
  std::vector<float> attn_output_bias;
  std::vector<float> attn_q_norm;   // per-head (Qwen3) optional
  std::vector<float> attn_k_norm;   // per-head (Qwen3) optional

  std::vector<float> ffn_norm;
  std::vector<float> post_attention_norm;  // Gemma sandwich norm
  std::vector<float> post_ffn_norm;        // Gemma sandwich norm
  LlamaWeight ffn_gate;
  LlamaWeight ffn_up;
  LlamaWeight ffn_down;
  std::vector<float> ffn_down_bias;
};

// Dense Llama-family model. Owns the GgufModel (for mmap lifetime) and the
// decoded/borrowed weights. Implements oxidize::Model.
class LlamaModel : public Model {
 public:
  explicit LlamaModel(GgufModel gguf);

  // Route the dense-decode hot path (matmuls, rms_norm, rope, attention, FFN)
  // through the CUDA backend when `on` and a device is available. Weights stay
  // resident on the GPU; only activation vectors transfer per op. Returns the
  // effective state (false if requested but no CUDA device / non-CUDA build).
  bool set_cuda(bool on);
  bool cuda_enabled() const { return use_cuda_; }

  Logits forward(const std::vector<Token>& tokens, Session& session) override;
  size_t vocab_size() const override { return config_.vocab_size; }
  size_t context_size() const override { return config_.context_size; }
  size_t layer_count() const override { return config_.layer_count; }
  void rewind_to(size_t consumed_tokens) override;

  const InferenceConfig& config() const { return config_; }
  const GgufModel& gguf() const { return gguf_; }

 private:
  // One decode step for `token` at absolute position `pos`. When need_logits is
  // false the lm_head is skipped (returns empty logits).
  Logits forward_single(Token token, size_t pos, bool need_logits);
  void embed_token(Token token, float* x) const;
  void run_layers(size_t pos);
  Logits final_head();

  // Op dispatch: CUDA backend when use_cuda_, else the CPU tensor.hpp kernels.
  // Signatures mirror tensor.hpp / cuda_backend.hpp exactly (host pointers).
  void d_rms_norm(float* out, const float* x, const float* w, size_t n,
                  float eps, bool plus_one);
  void d_apply_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                    float theta, size_t rope_dim);
  void d_swiglu(float* gate, const float* up, float* out, size_t n);
  void d_geglu(float* gate, const float* up, float* out, size_t n);
  void d_attention(float* out, const float* q, const float* k_cache,
                   const float* v_cache, size_t seq_len, size_t num_heads,
                   size_t kv_heads, size_t head_dim);
  void d_gemv_weight(const LlamaWeight& w, size_t rows, size_t cols,
                     const float* x, float* y);

#ifdef OXIDIZE_CUDA
  // Build a device-resident decode view of this model's weights (WIP).
  CudaBackend::ModelView build_cuda_view() const;
#endif

  bool use_cuda_ = false;

  static void reject_unsupported(const InferenceConfig& cfg);
  LlamaWeight load_weight(const GgufModel& g, const std::string& name,
                          bool keep_quantized);
  std::vector<float> load_vector(const GgufModel& g, const std::string& name);

  GgufModel gguf_;
  InferenceConfig config_;

  LlamaWeight tok_embeddings_;
  size_t tok_embeddings_cols_ = 0;
  std::vector<float> norm_weight_;
  LlamaWeight output_weight_;  // lm_head (may alias tok_embeddings_ when tied)
  bool output_tied_ = false;
  std::vector<LlamaLayer> layers_;

  // Layer-major F32 KV cache: [attn_layer][position][kv_heads*head_dim].
  // Mirrors kv_cache.rs F32 storage with token_slot_index = layer*ctx + pos.
  std::vector<float> kv_keys_;
  std::vector<float> kv_values_;
  size_t kv_token_size_ = 0;  // kv_heads * head_dim

  // Persistent activation + scratch (mirrors inference.rs::Workspace, dense subset).
  std::vector<float> x_;
};

// Factory: mmap + parse a .gguf and build a dense LlamaModel. When `want_cuda`
// the model routes the decode hot path to the GPU if a CUDA device is available
// (silently falls back to CPU otherwise).
// Throws std::runtime_error on parse failure or unsupported architecture.
std::unique_ptr<Model> load_llama_gguf(const std::string& path,
                                       bool want_cuda = false);

}  // namespace oxidize
