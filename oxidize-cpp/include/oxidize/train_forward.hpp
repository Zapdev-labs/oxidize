#pragma once
// fp32 training forward pass with autograd tape.
//
// TrainModel wraps a const LlamaModel* for weight access only. It holds its
// own fp32 master weights (full-FT) or LoRA adapters, and a separate Tape.
// The inference LlamaModel is NEVER mutated.
//
// Architecture: Qwen2.5 dense (hidden=896, heads=14, kv_heads=2, head_dim=64,
// inter=4864, layers=24, vocab=151936, weight_plus_one RMSNorm).

#include <cstddef>
#include <memory>
#include <vector>
#include <random>

#include "oxidize/config.hpp"
#include "oxidize/model_llama.hpp"
#include "oxidize/autograd.hpp"
#include "oxidize/lora.hpp"
#include "oxidize/train_types.hpp"

namespace oxidize {

// Per-layer saved activations for backward pass (materialized attention).
struct LayerActivations {
  std::vector<float> x_in;         // input to layer (before attn_norm), [h]
  std::vector<float> normed_attn;  // after attn_norm, [h]
  std::vector<float> q_rope;       // q after rope, [q_len]
  std::vector<float> k_rope;       // k after rope per position [T x kv_len]
  std::vector<float> v_save;       // v per position [T x kv_len]
  std::vector<float> attn_w;       // attn weights [n_heads x T]
  std::vector<float> attn_out_proj;// output proj input = concat heads, [q_len]
  std::vector<float> x_after_attn; // x after residual, [h]
  std::vector<float> normed_ffn;   // after ffn_norm, [h]
  std::vector<float> gate_pre;     // ffn gate before activation, [inter]
  std::vector<float> up_pre;       // ffn up, [inter]
  std::vector<float> ffn_act;      // silu(gate)*up, [inter]
};

// Full-FT master weight for one projection (dequantized fp32 copy of frozen).
struct FTWeight {
  std::vector<float> W;    // [rows x cols]
  std::vector<float> grad; // gradient (same shape)
  std::vector<float> m, v; // AdamW moments
  size_t rows = 0;
  size_t cols = 0;

  void init_from(const LlamaWeight& src, size_t rows_, size_t cols_);
  void zero_grad();
  void adamw_step(float lr, float beta1, float beta2, float eps,
                  float weight_decay, int t, bool skip_wd);
};

// Per-layer fp32 master weights (full-FT).
struct FTLayer {
  FTWeight attn_q, attn_k, attn_v, attn_o;
  FTWeight ffn_gate, ffn_up, ffn_down;
  // Norm weights (small, always fp32 in inference too).
  std::vector<float> attn_norm, ffn_norm;
  std::vector<float> dattn_norm, dffn_norm;
  std::vector<float> m_attn_norm, v_attn_norm;
  std::vector<float> m_ffn_norm, v_ffn_norm;
  int norm_t = 0;
  void zero_grad();
};

class TrainModel {
 public:
  // Construct from a loaded inference model. mode selects LoRA or full-FT.
  TrainModel(const LlamaModel* base, const TrainConfig& cfg, uint64_t seed);

  // Forward pass over `tokens` (seq_len), returns per-token logits [T x vocab].
  // Saves activations for backward. Causal: position i sees tokens [0..i].
  std::vector<float> forward(const std::vector<Token>& tokens);

  // Backward pass. logits_grad: [T x vocab] upstream gradient (from cross_entropy).
  // Accumulates parameter gradients. Does NOT step the optimizer.
  void backward(const std::vector<float>& logits_grad,
                const std::vector<Token>& tokens,
                const std::vector<float>& loss_mask);

  // AdamW step. t = optimizer step count (1-based, after grad_accum batches).
  void optimizer_step(float lr, int t);

  // Zero all gradients.
  void zero_grads();

  // Compute global gradient L2 norm. Returns the norm.
  float grad_norm() const;

  // Clip gradients so global norm <= max_norm (in-place scale).
  void clip_grad_norm(float max_norm);

  // Return peak saved activation bytes (for memory reporting).
  size_t activation_bytes() const;

  const InferenceConfig& config() const { return base_->config(); }

 private:
  const LlamaModel* base_;
  TrainConfig cfg_;

  // LoRA adapters (mode == LoRA). Indexed by layer.
  // Order per layer: [attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down]
  static constexpr int N_LORA_PER_LAYER = 7;
  std::vector<std::vector<LoraAdapter>> lora_adapters_;  // [layer][proj]

  // Frozen fp32 weight cache for LoRA mode (dequantized once at construction).
  // Per layer: [Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, norm_attn, norm_ffn].
  struct LayerWeightCache {
    std::vector<float> Wq, Wk, Wv, Wo, Wg, Wu, Wd;
    std::vector<float> norm_attn, norm_ffn;
  };
  std::vector<LayerWeightCache> lora_weight_cache_;  // only used in LoRA mode
  // Also cache lm_head and final norm for LoRA mode.
  std::vector<float> lora_norm_;       // final norm [h]
  std::vector<float> lora_lm_head_;   // [vocab x h]
  std::vector<float> lora_tok_emb_;   // [vocab x h]

  // Full-FT weights (mode == FullFT).
  std::vector<FTLayer> ft_layers_;
  std::vector<float> ft_tok_emb_;    // [vocab x h]
  std::vector<float> d_tok_emb_;
  std::vector<float> m_tok_emb_, v_tok_emb_;
  std::vector<float> ft_norm_;       // final norm weight [h]
  std::vector<float> d_norm_;
  std::vector<float> m_norm_, v_norm_;
  std::vector<float> ft_lm_head_;   // [vocab x h]
  std::vector<float> d_lm_head_;
  std::vector<float> m_lm_head_, v_lm_head_;
  bool lm_head_tied_ = false;

  // Saved activations per layer per forward pass.
  std::vector<LayerActivations> saved_;
  // x at each position [T x h] (residual stream)
  std::vector<float> x_stream_;    // [T x h]
  // logits [T x vocab]
  std::vector<float> logits_save_; // for backward
  size_t T_saved_ = 0;

  // AdamW step counter.
  int adamw_t_ = 0;

  // Helper: get fp32 pointer for a frozen weight (dequantized on the fly or f32).
  void dequant_weight(const LlamaWeight& w, std::vector<float>& out) const;
  // Get W_q in fp32 for layer l (FT: from ft_layers_, LoRA: dequant gguf).
  const float* get_W_q(size_t l, std::vector<float>& scratch) const;
};

}  // namespace oxidize
