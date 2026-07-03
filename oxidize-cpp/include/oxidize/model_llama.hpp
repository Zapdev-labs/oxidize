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
#ifdef OXIDIZE_GPU
#include "oxidize/cuda_backend.hpp"
#endif

namespace oxidize {

// Storage for one weight matrix: either dequantized f32, or a borrowed pointer
// into the mmap'd GGUF data kept in its quantized block layout (decoded on the
// fly by gemv_quantized). Mirrors WeightStorage::{F32, MmapQuantized}.
struct LlamaWeight {
  bool quantized = false;
  QuantType quant = QuantType::F32;
  const uint8_t* data = nullptr;   // valid when quantized + mmap-backed
  std::vector<uint8_t> owned;      // valid when quantized + on-the-fly quantized
  std::vector<float> f32;          // valid when !quantized
  size_t rows = 0;                 // output features
  size_t cols = 0;                 // input features

  // Quantized block bytes (owned takes precedence over the borrowed mmap ptr).
  const uint8_t* qbytes() const { return owned.empty() ? data : owned.data(); }
  bool empty() const {
    return quantized ? (data == nullptr && owned.empty()) : f32.empty();
  }
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

  // MoE (Mixtral / Qwen-MoE): routed-expert FFN. Expert tensors are 3D
  // [n_experts, *, *]; gate_inp is the router [n_experts, hidden]. Empty on
  // dense layers. Mirrors inference.rs::moe_ffn_forward_weights.
  LlamaWeight ffn_gate_exps;  // [n_experts, expert_inter, hidden]
  LlamaWeight ffn_up_exps;    // [n_experts, expert_inter, hidden]
  LlamaWeight ffn_down_exps;  // [n_experts, hidden, expert_inter]
  LlamaWeight ffn_gate_inp;   // router [n_experts, hidden]
  std::vector<float> ffn_exp_probs_b;  // sigmoid-gating per-expert bias (LFM2MoE)

  // Shared (always-on) expert FFN (GLM-5.2 / DeepSeek-V2). Runs unconditionally
  // alongside the routed experts and is summed into the MoE output. Empty when
  // the model has no shared expert. Mirrors the shexp tensors in the GGUF.
  LlamaWeight ffn_gate_shexp;
  LlamaWeight ffn_up_shexp;
  LlamaWeight ffn_down_shexp;

  // MLA (GLM-5.2 glm-dsa / DeepSeek-V2 compressed attention). These replace the
  // dense attn_q/attn_k/attn_v projections. Empty on dense-attention layers.
  //   attn_q_a       : x -> q_lora_rank          (q down-proj)
  //   attn_q_a_norm  : RMSNorm over q_lora_rank
  //   attn_q_b       : q_lora_rank -> n_heads*mla_key_dim  (q up-proj)
  //   attn_kv_a_mqa  : x -> kv_lora_rank + rope  (kv down-proj, MQA)
  //   attn_kv_a_norm : RMSNorm over kv_lora_rank
  //   attn_k_b       : kv_lora_rank -> per-head key   (k up-proj, 3D)
  //   attn_v_b       : kv_lora_rank -> per-head value (v up-proj, 3D)
  LlamaWeight attn_q_a;
  std::vector<float> attn_q_a_norm;
  LlamaWeight attn_q_b;
  LlamaWeight attn_kv_a_mqa;
  std::vector<float> attn_kv_a_norm;
  LlamaWeight attn_k_b;
  LlamaWeight attn_v_b;

  bool is_moe() const { return !ffn_gate_exps.empty() || !ffn_gate_inp.empty(); }
  // MLA layer: uses compressed q/kv projections instead of dense attn_q/k/v.
  bool is_mla() const { return !attn_q_a.empty() || !attn_kv_a_mqa.empty(); }
  bool has_shared_expert() const {
    return !ffn_gate_shexp.empty() || !ffn_down_shexp.empty();
  }
};

// Dense Llama-family model. Owns the GgufModel (for mmap lifetime) and the
// decoded/borrowed weights. Implements oxidize::Model.
class LlamaModel : public Model {
 public:
  // `quantize_to` (Q8_0) quantizes F16/BF16/F32 weight matrices to Q8_0 at load
  // for ~1.3x faster, near-lossless decode; QuantType::F32 = keep as-is.
  explicit LlamaModel(GgufModel gguf, QuantType quantize_to = QuantType::F32);

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
  Logits forward_batched(const std::vector<Token>& tokens, size_t start_pos,
                         bool need_logits);
  void embed_token(Token token, float* x) const;
  void embed_tokens_batched(const std::vector<Token>& tokens, float* x_batch) const;
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
  void d_gemm_weight(const LlamaWeight& w, size_t rows, size_t cols,
                     const float* inputs, float* outputs, size_t batch);

#ifdef OXIDIZE_GPU
  void resident_sync_kv_to_gpu(size_t seq_len);
  // Build a device-resident decode view of this model's weights (WIP).
  CudaBackend::ModelView build_cuda_view() const;
#endif

  bool use_cuda_ = false;
  bool any_moe_ = false;  // true if any layer is MoE (resident GPU path is dense-only)
  QuantType quantize_to_ = QuantType::F32;  // on-the-fly weight quant (F32 = none)

  // Routed-expert MoE FFN: ffn_out += sum over top-k experts of w_e * FFN_e(normed).
  void moe_ffn(const LlamaLayer& layer, const float* normed, float* ffn_out);

  // MLA (GLM-5.2 glm-dsa / DeepSeek-V2) compressed-attention forward for one
  // layer at absolute position `pos`. Reads x_ (post-attn_norm done inside),
  // writes the attention output (pre-residual) into attn_out[hidden_size].
  // Uses kv_keys_ as the 576-dim compressed-latent cache (kv_heads=1); V aliases
  // the first kv_lora_rank dims of each cached row (no separate V cache).
  void mla_attention(const LlamaLayer& layer, size_t l, size_t pos,
                     float* attn_out);

  static void reject_unsupported(const InferenceConfig& cfg);
  LlamaWeight load_weight(const GgufModel& g, const std::string& name,
                          bool keep_quantized, bool allow_quant = true,
                          bool force_keep_quantized = false);
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
  // Allocated KV-cache context (== context_size for normal models; capped below
  // context_size only for models whose advertised context would require an
  // impractically large up-front F32 cache, e.g. GLM-5.2's 1M-token context).
  // Used as the layer stride for KV slot indexing; positions must be < this.
  size_t kv_context_ = 0;

  // Persistent activation + scratch (mirrors inference.rs::Workspace, dense subset).
  std::vector<float> x_;
};

// Factory: mmap + parse a .gguf and build a dense LlamaModel. When `want_cuda`
// the model routes the decode hot path to the GPU if a CUDA device is available
// (silently falls back to CPU otherwise).
// Throws std::runtime_error on parse failure or unsupported architecture.
std::unique_ptr<Model> load_llama_gguf(const std::string& path,
                                       bool want_cuda = false,
                                       QuantType quantize_to = QuantType::F32,
                                       GgufMmapAdvice mmap_advice =
                                           GgufMmapAdvice::SequentialPrefetch);

}  // namespace oxidize
