// fp32 training forward + backward pass.
// Wraps a const LlamaModel* for weight access; holds its own fp32 weights
// (full-FT) or LoRA adapters. Never mutates the inference model.

#include "oxidize/train_forward.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "oxidize/autograd.hpp"
#include "oxidize/tensor.hpp"
#include "oxidize/quant.hpp"

namespace oxidize {


void FTWeight::init_from(const LlamaWeight& src, size_t rows_, size_t cols_) {
  rows = rows_;
  cols = cols_;
  size_t n = rows * cols;
  W.assign(n, 0.0f);
  grad.assign(n, 0.0f);
  m.assign(n, 0.0f);
  v.assign(n, 0.0f);
  if (src.quantized) {
    dequantize_row(src.quant, src.qbytes(), W.data(), n);
  } else if (!src.f32.empty()) {
    W = src.f32;
  }
}

void FTWeight::zero_grad() {
  std::fill(grad.begin(), grad.end(), 0.0f);
}

void FTWeight::adamw_step(float lr, float beta1, float beta2, float eps,
                           float weight_decay, int t, bool skip_wd) {
  float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
  float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
  size_t n = W.size();
  for (size_t i = 0; i < n; ++i) {
    float g = grad[i] + (skip_wd ? 0.0f : weight_decay * W[i]);
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    W[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
  }
}


void FTLayer::zero_grad() {
  attn_q.zero_grad(); attn_k.zero_grad(); attn_v.zero_grad(); attn_o.zero_grad();
  ffn_gate.zero_grad(); ffn_up.zero_grad(); ffn_down.zero_grad();
  std::fill(dattn_norm.begin(), dattn_norm.end(), 0.0f);
  std::fill(dffn_norm.begin(), dffn_norm.end(), 0.0f);
}


void TrainModel::dequant_weight(const LlamaWeight& w,
                                 std::vector<float>& out) const {
  size_t n = w.rows * w.cols;
  out.resize(n);
  if (w.quantized) {
    dequantize_row(w.quant, w.qbytes(), out.data(), n);
  } else {
    out = w.f32;
  }
}

const float* TrainModel::get_W_q(size_t l, std::vector<float>& scratch) const {
  if (cfg_.mode == TrainMode::FullFT) return ft_layers_[l].attn_q.W.data();
  const InferenceConfig& ic = base_->config();
  size_t n = ic.num_attention_heads * ic.head_dim() * ic.hidden_size;
  TensorView tv = base_->gguf().tensor("blk." + std::to_string(l) + ".attn_q.weight");
  scratch.assign(n, 0.0f);
  dequantize_row(tv.quant, tv.data, scratch.data(), n);
  return scratch.data();
}


TrainModel::TrainModel(const LlamaModel* base, const TrainConfig& cfg,
                       uint64_t seed)
    : base_(base), cfg_(cfg) {
  const InferenceConfig& ic = base->config();
  const size_t h    = ic.hidden_size;
  const size_t q_len  = ic.num_attention_heads * ic.head_dim();
  const size_t kv_len = ic.num_key_value_heads  * ic.head_dim();
  const size_t inter  = ic.intermediate_size;
  const size_t vocab  = ic.vocab_size;
  const size_t L      = ic.layer_count;

  std::mt19937_64 rng(seed);

  if (cfg_.mode == TrainMode::LoRA) {
    size_t rank = cfg_.lora.rank;
    float alpha = cfg_.lora.alpha;

    lora_adapters_.resize(L);
    for (size_t l = 0; l < L; ++l) {
      lora_adapters_[l].resize(N_LORA_PER_LAYER);
      // [0]=attn_q, [1]=attn_k, [2]=attn_v, [3]=attn_o,
      // [4]=ffn_gate, [5]=ffn_up, [6]=ffn_down
      lora_adapters_[l][0].init(q_len,  h,     rank, alpha, rng);
      lora_adapters_[l][1].init(kv_len, h,     rank, alpha, rng);
      lora_adapters_[l][2].init(kv_len, h,     rank, alpha, rng);
      lora_adapters_[l][3].init(h,      q_len, rank, alpha, rng);
      lora_adapters_[l][4].init(inter,  h,     rank, alpha, rng);
      lora_adapters_[l][5].init(inter,  h,     rank, alpha, rng);
      lora_adapters_[l][6].init(h,      inter, rank, alpha, rng);
    }

    // Dequantize frozen weights once at construction (avoid per-step overhead).
    printf("[train] Dequantizing frozen weights for LoRA cache...\n"); fflush(stdout);
    auto t_dq_start = std::chrono::steady_clock::now();
    lora_weight_cache_.resize(L);
    for (size_t l = 0; l < L; ++l) {
      LayerWeightCache& wc = lora_weight_cache_[l];
      std::string p = "blk." + std::to_string(l) + ".";
      auto load_w = [&](const std::string& name, size_t rows, size_t cols,
                         std::vector<float>& buf) {
        buf.assign(rows * cols, 0.0f);
        if (!base->gguf().has_tensor(name)) return;
        TensorView tv = base->gguf().tensor(name);
        dequantize_row(tv.quant, tv.data, buf.data(), rows * cols);
      };
      auto t_l = std::chrono::steady_clock::now();
      load_w(p + "attn_q.weight",      q_len,  h,     wc.Wq);
      load_w(p + "attn_k.weight",      kv_len, h,     wc.Wk);
      load_w(p + "attn_v.weight",      kv_len, h,     wc.Wv);
      load_w(p + "attn_output.weight", h,      q_len, wc.Wo);
      load_w(p + "ffn_gate.weight",    inter,  h,     wc.Wg);
      load_w(p + "ffn_up.weight",      inter,  h,     wc.Wu);
      load_w(p + "ffn_down.weight",    h,      inter, wc.Wd);
      if (l == 0 || l == L-1) {
        auto t_e = std::chrono::steady_clock::now();
        printf("[train] Layer %zu dequant: %.2f s\n", l,
               std::chrono::duration<double>(t_e-t_l).count()); fflush(stdout);
      }
      wc.norm_attn.assign(h, 1.0f);
      if (base->gguf().has_tensor(p + "attn_norm.weight")) {
        TensorView tv = base->gguf().tensor(p + "attn_norm.weight");
        dequantize_row(tv.quant, tv.data, wc.norm_attn.data(), h);
      }
      wc.norm_ffn.assign(h, 1.0f);
      if (base->gguf().has_tensor(p + "ffn_norm.weight")) {
        TensorView tv = base->gguf().tensor(p + "ffn_norm.weight");
        dequantize_row(tv.quant, tv.data, wc.norm_ffn.data(), h);
      }
    }
    // Embeddings.
    lora_tok_emb_.assign(vocab * h, 0.0f);
    {
      std::string en = base->gguf().has_tensor("token_embd.weight")
                     ? "token_embd.weight" : "tok_embeddings.weight";
      TensorView tv = base->gguf().tensor(en);
      dequantize_row(tv.quant, tv.data, lora_tok_emb_.data(), vocab * h);
    }
    // Final norm.
    lora_norm_.assign(h, 0.0f);
    {
      std::string nm = base->gguf().has_tensor("output_norm.weight")
                     ? "output_norm.weight" : "norm.weight";
      TensorView tv = base->gguf().tensor(nm);
      dequantize_row(tv.quant, tv.data, lora_norm_.data(), h);
    }
    // LM head.
    bool has_out = base->gguf().has_tensor("output.weight");
    lora_lm_head_.assign(vocab * h, 0.0f);
    if (has_out) {
      TensorView tv = base->gguf().tensor("output.weight");
      dequantize_row(tv.quant, tv.data, lora_lm_head_.data(), vocab * h);
    } else {
      lora_lm_head_ = lora_tok_emb_;
    }
    {
      auto t_dq_end = std::chrono::steady_clock::now();
      printf("[train] Weight cache ready. Total dequant: %.2f s\n",
             std::chrono::duration<double>(t_dq_end-t_dq_start).count()); fflush(stdout);
    }
  } else {
    // Full-FT: dequantize all weight matrices into fp32 masters.
    // Read layers from the LlamaModel via gguf().
    ft_layers_.resize(L);
    const auto& layers = base->config();  // just for dims
    (void)layers;

    // We need access to LlamaLayer weights. Use the fact that LlamaModel
    // stores them as private. We instead load from gguf directly.
    // Build a LlamaWeight dequant helper from gguf tensor views.
    auto dq_from_gguf = [&](const std::string& name, size_t rows, size_t cols,
                              FTWeight& dst) {
      if (!base->gguf().has_tensor(name)) {
        dst.W.assign(rows * cols, 0.0f);
        dst.grad.assign(rows * cols, 0.0f);
        dst.m.assign(rows * cols, 0.0f);
        dst.v.assign(rows * cols, 0.0f);
        dst.rows = rows; dst.cols = cols;
        return;
      }
      TensorView tv = base->gguf().tensor(name);
      size_t n = rows * cols;
      dst.W.assign(n, 0.0f);
      dequantize_row(tv.quant, tv.data, dst.W.data(), n);
      dst.grad.assign(n, 0.0f);
      dst.m.assign(n, 0.0f);
      dst.v.assign(n, 0.0f);
      dst.rows = rows; dst.cols = cols;
    };

    for (size_t l = 0; l < L; ++l) {
      FTLayer& fl = ft_layers_[l];
      std::string p = "blk." + std::to_string(l) + ".";
      dq_from_gguf(p + "attn_q.weight",      q_len,  h,     fl.attn_q);
      dq_from_gguf(p + "attn_k.weight",      kv_len, h,     fl.attn_k);
      dq_from_gguf(p + "attn_v.weight",      kv_len, h,     fl.attn_v);
      dq_from_gguf(p + "attn_output.weight", h,      q_len, fl.attn_o);
      dq_from_gguf(p + "ffn_gate.weight",    inter,  h,     fl.ffn_gate);
      dq_from_gguf(p + "ffn_up.weight",      inter,  h,     fl.ffn_up);
      dq_from_gguf(p + "ffn_down.weight",    h,      inter, fl.ffn_down);

      // Norm weights (fp32 in both modes).
      if (base->gguf().has_tensor(p + "attn_norm.weight")) {
        TensorView tv = base->gguf().tensor(p + "attn_norm.weight");
        fl.attn_norm.assign(h, 0.0f);
        dequantize_row(tv.quant, tv.data, fl.attn_norm.data(), h);
      } else {
        fl.attn_norm.assign(h, 1.0f);
      }
      if (base->gguf().has_tensor(p + "ffn_norm.weight")) {
        TensorView tv = base->gguf().tensor(p + "ffn_norm.weight");
        fl.ffn_norm.assign(h, 0.0f);
        dequantize_row(tv.quant, tv.data, fl.ffn_norm.data(), h);
      } else {
        fl.ffn_norm.assign(h, 1.0f);
      }
      fl.dattn_norm.assign(h, 0.0f);
      fl.dffn_norm.assign(h, 0.0f);
      fl.m_attn_norm.assign(h, 0.0f); fl.v_attn_norm.assign(h, 0.0f);
      fl.m_ffn_norm.assign(h, 0.0f);  fl.v_ffn_norm.assign(h, 0.0f);
      fl.norm_t = 0;
    }

    // Embedding.
    ft_tok_emb_.assign(vocab * h, 0.0f);
    {
      std::string name = base->gguf().has_tensor("token_embd.weight")
                       ? "token_embd.weight" : "tok_embeddings.weight";
      TensorView tv = base->gguf().tensor(name);
      dequantize_row(tv.quant, tv.data, ft_tok_emb_.data(), vocab * h);
    }
    d_tok_emb_.assign(vocab * h, 0.0f);
    m_tok_emb_.assign(vocab * h, 0.0f);
    v_tok_emb_.assign(vocab * h, 0.0f);

    // Final norm.
    ft_norm_.assign(h, 0.0f);
    {
      std::string name = base->gguf().has_tensor("output_norm.weight")
                       ? "output_norm.weight" : "norm.weight";
      TensorView tv = base->gguf().tensor(name);
      dequantize_row(tv.quant, tv.data, ft_norm_.data(), h);
    }
    d_norm_.assign(h, 0.0f);
    m_norm_.assign(h, 0.0f);
    v_norm_.assign(h, 0.0f);

    // LM head (tied or separate).
    lm_head_tied_ = !base->gguf().has_tensor("output.weight");
    if (!lm_head_tied_) {
      ft_lm_head_.assign(vocab * h, 0.0f);
      TensorView tv = base->gguf().tensor("output.weight");
      dequantize_row(tv.quant, tv.data, ft_lm_head_.data(), vocab * h);
      d_lm_head_.assign(vocab * h, 0.0f);
      m_lm_head_.assign(vocab * h, 0.0f);
      v_lm_head_.assign(vocab * h, 0.0f);
    }
  }
}


std::vector<float> TrainModel::forward(const std::vector<Token>& tokens) {
  const InferenceConfig& ic = base_->config();
  const size_t h    = ic.hidden_size;
  const size_t q_len  = ic.num_attention_heads * ic.head_dim();
  const size_t kv_len = ic.num_key_value_heads  * ic.head_dim();
  const size_t inter  = ic.intermediate_size;
  const size_t vocab  = ic.vocab_size;
  const size_t L      = ic.layer_count;
  const size_t T      = tokens.size();
  const float  eps    = ic.rms_norm_eps;
  const bool   plus1  = ic.rms_norm_weight_plus_one;
  const size_t n_heads  = ic.num_attention_heads;
  const size_t kv_heads = ic.num_key_value_heads;
  const size_t hd       = ic.head_dim();
  const float  rope_th  = ic.rope_theta;
  const size_t rope_dim = ic.rope_dim;

  T_saved_ = T;
  x_stream_.assign(T * h, 0.0f);
  saved_.resize(L);

  // Embed tokens.
  if (cfg_.mode == TrainMode::FullFT) {
    for (size_t t = 0; t < T; ++t) {
      size_t tid = std::min<size_t>(tokens[t], vocab - 1);
      const float* row = ft_tok_emb_.data() + tid * h;
      float* xrow = x_stream_.data() + t * h;
      for (size_t i = 0; i < h; ++i) xrow[i] = row[i];
    }
  } else {
    // LoRA mode: use pre-dequantized embedding cache.
    for (size_t t = 0; t < T; ++t) {
      size_t tid = std::min<size_t>(tokens[t], vocab - 1);
      const float* row = lora_tok_emb_.data() + tid * h;
      float* xrow = x_stream_.data() + t * h;
      for (size_t i = 0; i < h; ++i) xrow[i] = row[i];
      if (ic.embedding_scale != 1.0f)
        for (size_t i = 0; i < h; ++i) xrow[i] *= ic.embedding_scale;
    }
  }

  // Process each layer.
  std::vector<float> scratch_W;  // for LoRA mode dequant

  for (size_t l = 0; l < L; ++l) {
    LayerActivations& sa = saved_[l];
    sa.x_in.assign(T * h, 0.0f);
    sa.normed_attn.assign(T * h, 0.0f);
    sa.q_rope.assign(T * q_len, 0.0f);
    sa.k_rope.assign(T * kv_len, 0.0f);
    sa.v_save.assign(T * kv_len, 0.0f);
    sa.attn_w.assign(n_heads * T * T, 0.0f);  // [n_heads x T x T] (causal)
    sa.attn_out_proj.assign(T * q_len, 0.0f);
    sa.x_after_attn.assign(T * h, 0.0f);
    sa.normed_ffn.assign(T * h, 0.0f);
    sa.gate_pre.assign(T * inter, 0.0f);
    sa.up_pre.assign(T * inter, 0.0f);
    sa.ffn_act.assign(T * inter, 0.0f);

    // Copy x_in.
    std::copy(x_stream_.data(), x_stream_.data() + T * h, sa.x_in.data());

    // Get weight matrices (fp32).
    const float* Wq; const float* Wk; const float* Wv; const float* Wo;
    const float* Wg; const float* Wu; const float* Wd;
    const float* norm_attn; const float* norm_ffn;

    if (cfg_.mode == TrainMode::FullFT) {
      Wq = ft_layers_[l].attn_q.W.data();
      Wk = ft_layers_[l].attn_k.W.data();
      Wv = ft_layers_[l].attn_v.W.data();
      Wo = ft_layers_[l].attn_o.W.data();
      Wg = ft_layers_[l].ffn_gate.W.data();
      Wu = ft_layers_[l].ffn_up.W.data();
      Wd = ft_layers_[l].ffn_down.W.data();
      norm_attn = ft_layers_[l].attn_norm.data();
      norm_ffn  = ft_layers_[l].ffn_norm.data();
    } else {
      // LoRA: use pre-dequantized cache (fast path).
      const LayerWeightCache& wc = lora_weight_cache_[l];
      Wq = wc.Wq.data(); Wk = wc.Wk.data(); Wv = wc.Wv.data(); Wo = wc.Wo.data();
      Wg = wc.Wg.data(); Wu = wc.Wu.data(); Wd = wc.Wd.data();
      norm_attn = wc.norm_attn.data();
      norm_ffn  = wc.norm_ffn.data();
    }

    // --- Attention block (batched over T) ---

    // RMSNorm (per-token, sequential — cheap, n=h per token).
    for (size_t t = 0; t < T; ++t) {
      rms_norm(sa.normed_attn.data() + t * h,
               x_stream_.data() + t * h, norm_attn, h, eps, plus1);
    }

    // QKV projections — batch [T x h] -> [T x q_len], [T x kv_len].
    train_batch_matvec(sa.q_rope.data(), Wq, sa.normed_attn.data(), T, q_len, h);
    train_batch_matvec(sa.k_rope.data(), Wk, sa.normed_attn.data(), T, kv_len, h);
    train_batch_matvec(sa.v_save.data(), Wv, sa.normed_attn.data(), T, kv_len, h);

    // LoRA contributions to Q, K, V (per-token; LoRA is small rank, cheap).
    if (cfg_.mode == TrainMode::LoRA) {
      std::vector<float> ax(cfg_.lora.rank, 0.0f);
      for (size_t t = 0; t < T; ++t) {
        const float* normed_t = sa.normed_attn.data() + t * h;
        lora_adapters_[l][0].forward(normed_t, sa.q_rope.data() + t * q_len, ax.data());
        lora_adapters_[l][1].forward(normed_t, sa.k_rope.data() + t * kv_len, ax.data());
        lora_adapters_[l][2].forward(normed_t, sa.v_save.data() + t * kv_len, ax.data());
      }
    }

    // RoPE (per-token, cheap).
    for (size_t t = 0; t < T; ++t) {
      apply_rope(sa.q_rope.data() + t * q_len, hd, n_heads,  t, rope_th, rope_dim);
      apply_rope(sa.k_rope.data() + t * kv_len, hd, kv_heads, t, rope_th, rope_dim);
    }

    // Materialized causal attention (parallelized over heads).
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
#pragma omp parallel for schedule(static)
    for (long long h_idx_ll = 0; h_idx_ll < (long long)n_heads; ++h_idx_ll) {
      size_t h_idx = static_cast<size_t>(h_idx_ll);
      size_t kv_h = h_idx / (n_heads / kv_heads);
      for (size_t t = 0; t < T; ++t) {
        const float* q_t = sa.q_rope.data() + t * q_len + h_idx * hd;
        float* row = sa.attn_w.data() + h_idx * T * T + t * T;
        float max_s = -1e38f;
        for (size_t s = 0; s <= t; ++s) {
          const float* k_s = sa.k_rope.data() + s * kv_len + kv_h * hd;
          float dot = 0.0f;
          for (size_t d = 0; d < hd; ++d) dot += q_t[d] * k_s[d];
          row[s] = dot * scale; max_s = std::max(max_s, row[s]);
        }
        float sum = 0.0f;
        for (size_t s = 0; s <= t; ++s) { row[s] = std::exp(row[s] - max_s); sum += row[s]; }
        if (sum > 0.0f) for (size_t s = 0; s <= t; ++s) row[s] /= sum;
      }
    }

    // Attention output (parallelized over n_heads * T).
    std::fill(sa.attn_out_proj.begin(), sa.attn_out_proj.end(), 0.0f);
#pragma omp parallel for schedule(static)
    for (long long ht = 0; ht < (long long)(n_heads * T); ++ht) {
      size_t h_idx = static_cast<size_t>(ht) / T;
      size_t t     = static_cast<size_t>(ht) % T;
      size_t kv_h = h_idx / (n_heads / kv_heads);
      const float* row = sa.attn_w.data() + h_idx * T * T + t * T;
      float* ao_h = sa.attn_out_proj.data() + t * q_len + h_idx * hd;
      for (size_t s = 0; s <= t; ++s) {
        const float* v_s = sa.v_save.data() + s * kv_len + kv_h * hd;
        for (size_t d = 0; d < hd; ++d) ao_h[d] += row[s] * v_s[d];
      }
    }

    // Output projection: batch [T x q_len] -> [T x h].
    std::vector<float> attn_out_batch(T * h, 0.0f);
    train_batch_matvec(attn_out_batch.data(), Wo, sa.attn_out_proj.data(), T, h, q_len);

    // LoRA attn_o (per-token).
    if (cfg_.mode == TrainMode::LoRA) {
      std::vector<float> ax(cfg_.lora.rank, 0.0f);
      for (size_t t = 0; t < T; ++t) {
        lora_adapters_[l][3].forward(sa.attn_out_proj.data() + t * q_len,
                                     attn_out_batch.data() + t * h, ax.data());
      }
    }

    // Residual + save x_after_attn.
    for (size_t t = 0; t < T; ++t) {
      float* x_t = x_stream_.data() + t * h;
      const float* ao_t = attn_out_batch.data() + t * h;
      for (size_t i = 0; i < h; ++i) x_t[i] += ao_t[i];
    }
    std::copy(x_stream_.data(), x_stream_.data() + T * h, sa.x_after_attn.data());

    // --- FFN block (batched) ---

    // Pre-FFN RMSNorm.
    for (size_t t = 0; t < T; ++t) {
      rms_norm(sa.normed_ffn.data() + t * h,
               x_stream_.data() + t * h, norm_ffn, h, eps, plus1);
    }

    // Gate + Up projections (batch).
    train_batch_matvec(sa.gate_pre.data(), Wg, sa.normed_ffn.data(), T, inter, h);
    train_batch_matvec(sa.up_pre.data(),   Wu, sa.normed_ffn.data(), T, inter, h);

    // LoRA gate/up (per-token).
    if (cfg_.mode == TrainMode::LoRA) {
      std::vector<float> ax(cfg_.lora.rank, 0.0f);
      for (size_t t = 0; t < T; ++t) {
        const float* normed_f = sa.normed_ffn.data() + t * h;
        lora_adapters_[l][4].forward(normed_f, sa.gate_pre.data() + t * inter, ax.data());
        lora_adapters_[l][5].forward(normed_f, sa.up_pre.data() + t * inter, ax.data());
      }
    }

    // SwiGLU activation (per-token).
    for (size_t t = 0; t < T; ++t) {
      float* gate_t = sa.gate_pre.data() + t * inter;
      float* up_t   = sa.up_pre.data()   + t * inter;
      float* act_t  = sa.ffn_act.data()  + t * inter;
      std::copy(gate_t, gate_t + inter, act_t);
      swiglu_inplace(act_t, up_t, act_t, inter);
    }

    // Down projection (batch).
    std::vector<float> ffn_out_batch(T * h, 0.0f);
    train_batch_matvec(ffn_out_batch.data(), Wd, sa.ffn_act.data(), T, h, inter);

    // LoRA down (per-token, input=ffn_act, output+=ffn_out_batch).
    if (cfg_.mode == TrainMode::LoRA) {
      std::vector<float> ax(cfg_.lora.rank, 0.0f);
      for (size_t t = 0; t < T; ++t) {
        lora_adapters_[l][6].forward(sa.ffn_act.data() + t * inter,
                                     ffn_out_batch.data() + t * h, ax.data());
      }
    }

    // FFN residual.
    for (size_t t = 0; t < T; ++t) {
      float* x_t = x_stream_.data() + t * h;
      const float* fo_t = ffn_out_batch.data() + t * h;
      for (size_t i = 0; i < h; ++i) x_t[i] += fo_t[i];
    }
  }

  // Final norm + lm_head.
  const float* norm_w = (cfg_.mode == TrainMode::FullFT) ? ft_norm_.data() : lora_norm_.data();
  const float* lm_head_w = (cfg_.mode == TrainMode::FullFT)
    ? (lm_head_tied_ ? ft_tok_emb_.data() : ft_lm_head_.data())
    : lora_lm_head_.data();

  // Final norm (per-token) then batch lm_head projection.
  std::vector<float> normed_final_batch(T * h, 0.0f);
  for (size_t t = 0; t < T; ++t) {
    rms_norm(normed_final_batch.data() + t * h,
             x_stream_.data() + t * h, norm_w, h, ic.rms_norm_eps, plus1);
  }
  // lm_head: [vocab x h] * [T x h]^T -> [T x vocab].
  logits_save_.assign(T * vocab, 0.0f);
  train_batch_matvec(logits_save_.data(), lm_head_w,
                     normed_final_batch.data(), T, vocab, h);

  return logits_save_;
}


void TrainModel::backward(const std::vector<float>& logits_grad,
                          const std::vector<Token>& tokens,
                          const std::vector<float>& loss_mask) {
  const InferenceConfig& ic = base_->config();
  const size_t h    = ic.hidden_size;
  const size_t q_len  = ic.num_attention_heads * ic.head_dim();
  const size_t kv_len = ic.num_key_value_heads  * ic.head_dim();
  const size_t inter  = ic.intermediate_size;
  const size_t vocab  = ic.vocab_size;
  const size_t L      = ic.layer_count;
  const size_t T      = T_saved_;
  const float  eps    = ic.rms_norm_eps;
  const bool   plus1  = ic.rms_norm_weight_plus_one;
  const size_t n_heads  = ic.num_attention_heads;
  const size_t kv_heads = ic.num_key_value_heads;
  const size_t hd       = ic.head_dim();
  const float  rope_th  = ic.rope_theta;
  const size_t rope_dim = ic.rope_dim;

  // Gradient of residual stream.
  std::vector<float> dx_stream(T * h, 0.0f);

  // Gradient from logits (already computed by caller via cross_entropy_backward).
  // logits_grad: [T x vocab]. Backprop through lm_head and final norm.

  const float* norm_w = (cfg_.mode == TrainMode::FullFT) ? ft_norm_.data() : lora_norm_.data();
  const float* lm_head_w = (cfg_.mode == TrainMode::FullFT)
    ? (lm_head_tied_ ? ft_tok_emb_.data() : ft_lm_head_.data())
    : lora_lm_head_.data();

  // Recompute normed_final_batch for lm_head backward.
  std::vector<float> normed_final_back(T * h, 0.0f);
  for (size_t t = 0; t < T; ++t) {
    rms_norm(normed_final_back.data() + t * h,
             x_stream_.data() + t * h, norm_w, h, eps, plus1);
  }

  // lm_head backward (batch): dx_normed[T x h] = dlogits[T x vocab] * lm_head[vocab x h]
  std::vector<float> dx_normed_all(T * h, 0.0f);
  auto t_lmh_bwd0 = std::chrono::steady_clock::now();
  if (cfg_.mode == TrainMode::FullFT) {
    float* dW_lm = lm_head_tied_ ? d_tok_emb_.data() : d_lm_head_.data();
    train_wt_dy(dx_normed_all.data(), lm_head_w, logits_grad.data(), T, vocab, h);
    train_outer_accum(dW_lm, logits_grad.data(), normed_final_back.data(), T, vocab, h);
  } else {
    // LoRA: frozen lm_head; just propagate dx.
    train_wt_dy(dx_normed_all.data(), lm_head_w, logits_grad.data(), T, vocab, h);
  }

  {
    auto t_lmh_bwd1 = std::chrono::steady_clock::now();
    static bool once = true;
    if (once) {
      printf("[bwd_timing] lm_head_bwd=%.3fs\n",
             std::chrono::duration<double>(t_lmh_bwd1-t_lmh_bwd0).count());
      fflush(stdout);
      once = false;
    }
  }
  // Backward through final norm (per-token).
  for (size_t t = 0; t < T; ++t) {
    const float* x_t = x_stream_.data() + t * h;
    std::vector<float> dx_t(h, 0.0f), dnorm_w_t(h, 0.0f);
    rmsnorm_backward(dx_t.data(), dnorm_w_t.data(),
                     x_t, norm_w, dx_normed_all.data() + t * h, h, eps, plus1);
    if (cfg_.mode == TrainMode::FullFT) {
      for (size_t i = 0; i < h; ++i) d_norm_[i] += dnorm_w_t[i];
    }
    for (size_t i = 0; i < h; ++i) dx_stream[t * h + i] += dx_t[i];
  }

  // Backprop through layers in reverse.
  for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
    LayerActivations& sa = saved_[l];

    // Load weight matrices for this layer.
    const float* Wq; const float* Wk; const float* Wv; const float* Wo;
    const float* Wg; const float* Wu; const float* Wd;
    const float* norm_attn; const float* norm_ffn;

    if (cfg_.mode == TrainMode::FullFT) {
      Wq = ft_layers_[l].attn_q.W.data();
      Wk = ft_layers_[l].attn_k.W.data();
      Wv = ft_layers_[l].attn_v.W.data();
      Wo = ft_layers_[l].attn_o.W.data();
      Wg = ft_layers_[l].ffn_gate.W.data();
      Wu = ft_layers_[l].ffn_up.W.data();
      Wd = ft_layers_[l].ffn_down.W.data();
      norm_attn = ft_layers_[l].attn_norm.data();
      norm_ffn  = ft_layers_[l].ffn_norm.data();
    } else {
      // LoRA: use pre-dequantized cache.
      const LayerWeightCache& wc = lora_weight_cache_[l];
      Wq = wc.Wq.data(); Wk = wc.Wk.data(); Wv = wc.Wv.data(); Wo = wc.Wo.data();
      Wg = wc.Wg.data(); Wu = wc.Wu.data(); Wd = wc.Wd.data();
      norm_attn = wc.norm_attn.data();
      norm_ffn  = wc.norm_ffn.data();
    }

    // --- FFN backward (batched over T) ---
    std::vector<float> dx_ffn(T * h, 0.0f);   // gradient into FFN block

    // Down proj backward: dact[T x inter] = dx_stream[T x h] * Wd[h x inter]^T
    // i.e., dact[t,c] = sum_r Wd[r,inter]*dx_stream[t,r]
    std::vector<float> dact_all(T * inter, 0.0f);
    if (cfg_.mode == TrainMode::FullFT) {
      // Full-FT: also accumulate weight gradient dWd += dx_stream^T * ffn_act
      train_wt_dy(dact_all.data(), Wd, dx_stream.data(), T, h, inter);
      train_outer_accum(ft_layers_[l].ffn_down.grad.data(),
                        dx_stream.data(), sa.ffn_act.data(), T, h, inter);
    } else {
      // LoRA: frozen Wd, just propagate dx.
      train_wt_dy(dact_all.data(), Wd, dx_stream.data(), T, h, inter);
    }

    // LoRA down backward (per-token, rank is small).
    if (cfg_.mode == TrainMode::LoRA) {
      LoraAdapter& la6 = lora_adapters_[l][6];
      for (size_t t = 0; t < T; ++t) {
        const float* act_t = sa.ffn_act.data() + t * inter;
        const float* dx_t = dx_stream.data() + t * h;
        std::vector<float> ax_lora(la6.rank, 0.0f);
        for (size_t r = 0; r < la6.rank; ++r) {
          float s = 0.0f;
          for (size_t c = 0; c < la6.cols; ++c) s += la6.A[r * la6.cols + c] * act_t[c];
          ax_lora[r] = s;
        }
        std::vector<float> dx_lora(inter, 0.0f);
        la6.backward(act_t, dx_t, ax_lora.data(), dx_lora.data(), la6.rank);
        for (size_t i = 0; i < inter; ++i) dact_all[t * inter + i] += dx_lora[i];
      }
    }

    // SwiGLU backward (per-token, cheap).
    std::vector<float> dgate_all(T * inter, 0.0f), dup_all(T * inter, 0.0f);
    for (size_t t = 0; t < T; ++t) {
      swiglu_backward(dgate_all.data() + t * inter, dup_all.data() + t * inter,
                      sa.gate_pre.data() + t * inter, sa.up_pre.data() + t * inter,
                      dact_all.data() + t * inter, inter);
    }

    // Gate + Up proj backward (batched): dx_normed_ffn[T x h].
    std::vector<float> dx_normed_ffn_all(T * h, 0.0f);
    if (cfg_.mode == TrainMode::FullFT) {
      train_wt_dy(dx_normed_ffn_all.data(), Wg, dgate_all.data(), T, inter, h);
      train_wt_dy(dx_normed_ffn_all.data(), Wu, dup_all.data(), T, inter, h);
      train_outer_accum(ft_layers_[l].ffn_gate.grad.data(),
                        dgate_all.data(), sa.normed_ffn.data(), T, inter, h);
      train_outer_accum(ft_layers_[l].ffn_up.grad.data(),
                        dup_all.data(), sa.normed_ffn.data(), T, inter, h);
    } else {
      train_wt_dy(dx_normed_ffn_all.data(), Wg, dgate_all.data(), T, inter, h);
      train_wt_dy(dx_normed_ffn_all.data(), Wu, dup_all.data(), T, inter, h);
    }

    // LoRA gate/up backward (per-token, rank small).
    if (cfg_.mode == TrainMode::LoRA) {
      for (size_t t = 0; t < T; ++t) {
        const float* normed_f = sa.normed_ffn.data() + t * h;
        auto lora_back_ffn = [&](LoraAdapter& la, const float* dout_r) {
          std::vector<float> ax(la.rank, 0.0f);
          for (size_t r = 0; r < la.rank; ++r) {
            float s = 0.0f;
            for (size_t c = 0; c < la.cols; ++c) s += la.A[r * la.cols + c] * normed_f[c];
            ax[r] = s;
          }
          std::vector<float> dx_la(h, 0.0f);
          la.backward(normed_f, dout_r, ax.data(), dx_la.data(), la.rank);
          for (size_t i = 0; i < h; ++i) dx_normed_ffn_all[t * h + i] += dx_la[i];
        };
        lora_back_ffn(lora_adapters_[l][4], dgate_all.data() + t * inter);
        lora_back_ffn(lora_adapters_[l][5], dup_all.data() + t * inter);
      }
    }

    // FFN norm backward (per-token) and compute dx_ffn.
    for (size_t t = 0; t < T; ++t) {
      const float* x_after_attn_t = sa.x_after_attn.data() + t * h;
      std::vector<float> dx_res_ffn(h, 0.0f), dnorm_ffn_t(h, 0.0f);
      rmsnorm_backward(dx_res_ffn.data(), dnorm_ffn_t.data(),
                       x_after_attn_t, norm_ffn,
                       dx_normed_ffn_all.data() + t * h, h, eps, plus1);
      if (cfg_.mode == TrainMode::FullFT) {
        for (size_t i = 0; i < h; ++i) ft_layers_[l].dffn_norm[i] += dnorm_ffn_t[i];
      }
      for (size_t i = 0; i < h; ++i)
        dx_ffn[t * h + i] = dx_stream[t * h + i] + dx_res_ffn[i];
    }

    // --- Attention backward ---
    // Gradient into x_after_attn is dx_ffn.
    std::vector<float> dx_attn_block(T * h, 0.0f);  // gradient into attention residual

    // Output proj backward (batch): dao_all[T x q_len] = dx_ffn[T x h] * Wo[h x q_len]^T.
    std::vector<float> dao_all(T * q_len, 0.0f);
    if (cfg_.mode == TrainMode::FullFT) {
      train_wt_dy(dao_all.data(), Wo, dx_ffn.data(), T, h, q_len);
      train_outer_accum(ft_layers_[l].attn_o.grad.data(),
                        dx_ffn.data(), sa.attn_out_proj.data(), T, h, q_len);
    } else {
      train_wt_dy(dao_all.data(), Wo, dx_ffn.data(), T, h, q_len);
    }

    // LoRA attn_o backward (per-token, rank small).
    if (cfg_.mode == TrainMode::LoRA) {
      LoraAdapter& la3 = lora_adapters_[l][3];
      for (size_t t = 0; t < T; ++t) {
        const float* ao_t = sa.attn_out_proj.data() + t * q_len;
        const float* dx_t = dx_ffn.data() + t * h;
        std::vector<float> ax(la3.rank, 0.0f);
        for (size_t r = 0; r < la3.rank; ++r) {
          float s = 0.0f;
          for (size_t c = 0; c < la3.cols; ++c) s += la3.A[r * la3.cols + c] * ao_t[c];
          ax[r] = s;
        }
        std::vector<float> dx_la(q_len, 0.0f);
        la3.backward(ao_t, dx_t, ax.data(), dx_la.data(), la3.rank);
        for (size_t i = 0; i < q_len; ++i) dao_all[t * q_len + i] += dx_la[i];
      }
    }

    // Residual passes through.
    for (size_t i = 0; i < T * h; ++i) dx_attn_block[i] += dx_ffn[i];

    // Attention weight backward.
    // dq, dk, dv: [T x q_len], [T x kv_len], [T x kv_len].
    std::vector<float> dq_all(T * q_len, 0.0f);
    std::vector<float> dk_all(T * kv_len, 0.0f);
    std::vector<float> dv_all(T * kv_len, 0.0f);

    float scale_attn = 1.0f / std::sqrt(static_cast<float>(hd));

    for (size_t h_idx = 0; h_idx < n_heads; ++h_idx) {
      size_t kv_h = h_idx / (n_heads / kv_heads);
      for (size_t t = 0; t < T; ++t) {
        const float* attn_row = sa.attn_w.data() + h_idx * T * T + t * T;
        const float* dao_h = dao_all.data() + t * q_len + h_idx * hd;

        // dattn[s] = dot(dao_h, v[s, kv_h]) for s <= t.
        std::vector<float> dattn(t + 1, 0.0f);
        for (size_t s = 0; s <= t; ++s) {
          const float* v_s = sa.v_save.data() + s * kv_len + kv_h * hd;
          float dot = 0.0f;
          for (size_t d = 0; d < hd; ++d) dot += dao_h[d] * v_s[d];
          dattn[s] = dot;
        }
        float sum_dattn = 0.0f;
        for (size_t s = 0; s <= t; ++s) sum_dattn += attn_row[s] * dattn[s];

        for (size_t s = 0; s <= t; ++s) {
          float ds = attn_row[s] * (dattn[s] - sum_dattn);
          const float* q_t = sa.q_rope.data() + t * q_len + h_idx * hd;
          const float* k_s = sa.k_rope.data() + s * kv_len + kv_h * hd;
          const float* v_s = sa.v_save.data() + s * kv_len + kv_h * hd;

          // dq[t, h] += scale * ds * k[s, kv_h]
          float* dq_t_h = dq_all.data() + t * q_len + h_idx * hd;
          for (size_t d = 0; d < hd; ++d) dq_t_h[d] += scale_attn * ds * k_s[d];

          // dk[s, kv_h] += scale * ds * q[t, h]
          float* dk_s_h = dk_all.data() + s * kv_len + kv_h * hd;
          for (size_t d = 0; d < hd; ++d) dk_s_h[d] += scale_attn * ds * q_t[d];

          // dv[s, kv_h] += attn[s] * dao_h
          float* dv_s_h = dv_all.data() + s * kv_len + kv_h * hd;
          for (size_t d = 0; d < hd; ++d) dv_s_h[d] += attn_row[s] * dao_h[d];
        }
      }
    }

    // RoPE backward for Q and K.
    for (size_t t = 0; t < T; ++t) {
      // rope_backward overwrites dx (not accumulates) — it's an inverse rotation.
      // dq_rope -> dq_pre_rope
      std::vector<float> dq_pre(q_len, 0.0f);
      rope_backward(dq_pre.data(), dq_all.data() + t * q_len,
                    hd, n_heads, t, rope_th, rope_dim);
      for (size_t i = 0; i < q_len; ++i) dq_all[t * q_len + i] = dq_pre[i];

      std::vector<float> dk_pre(kv_len, 0.0f);
      rope_backward(dk_pre.data(), dk_all.data() + t * kv_len,
                    hd, kv_heads, t, rope_th, rope_dim);
      for (size_t i = 0; i < kv_len; ++i) dk_all[t * kv_len + i] = dk_pre[i];
    }

    // Q, K, V proj backward (batched).
    std::vector<float> dx_normed_attn_all(T * h, 0.0f);
    if (cfg_.mode == TrainMode::FullFT) {
      train_wt_dy(dx_normed_attn_all.data(), Wq, dq_all.data(), T, q_len, h);
      train_wt_dy(dx_normed_attn_all.data(), Wk, dk_all.data(), T, kv_len, h);
      train_wt_dy(dx_normed_attn_all.data(), Wv, dv_all.data(), T, kv_len, h);
      train_outer_accum(ft_layers_[l].attn_q.grad.data(),
                        dq_all.data(), sa.normed_attn.data(), T, q_len, h);
      train_outer_accum(ft_layers_[l].attn_k.grad.data(),
                        dk_all.data(), sa.normed_attn.data(), T, kv_len, h);
      train_outer_accum(ft_layers_[l].attn_v.grad.data(),
                        dv_all.data(), sa.normed_attn.data(), T, kv_len, h);
    } else {
      train_wt_dy(dx_normed_attn_all.data(), Wq, dq_all.data(), T, q_len, h);
      train_wt_dy(dx_normed_attn_all.data(), Wk, dk_all.data(), T, kv_len, h);
      train_wt_dy(dx_normed_attn_all.data(), Wv, dv_all.data(), T, kv_len, h);
    }

    // LoRA Q/K/V backward (per-token, rank small).
    if (cfg_.mode == TrainMode::LoRA) {
      for (size_t t = 0; t < T; ++t) {
        const float* normed_t = sa.normed_attn.data() + t * h;
        auto lora_back_qkv = [&](LoraAdapter& la, const float* dout_r) {
          std::vector<float> ax(la.rank, 0.0f);
          for (size_t r = 0; r < la.rank; ++r) {
            float s = 0.0f;
            for (size_t c = 0; c < la.cols; ++c) s += la.A[r * la.cols + c] * normed_t[c];
            ax[r] = s;
          }
          std::vector<float> dx_la(h, 0.0f);
          la.backward(normed_t, dout_r, ax.data(), dx_la.data(), la.rank);
          for (size_t i = 0; i < h; ++i) dx_normed_attn_all[t * h + i] += dx_la[i];
        };
        lora_back_qkv(lora_adapters_[l][0], dq_all.data() + t * q_len);
        lora_back_qkv(lora_adapters_[l][1], dk_all.data() + t * kv_len);
        lora_back_qkv(lora_adapters_[l][2], dv_all.data() + t * kv_len);
      }
    }

    // Attn norm backward (per-token) and final dx_stream.
    for (size_t t = 0; t < T; ++t) {
      const float* x_in_t = sa.x_in.data() + t * h;
      std::vector<float> dx_res_attn(h, 0.0f), dnorm_attn_t(h, 0.0f);
      rmsnorm_backward(dx_res_attn.data(), dnorm_attn_t.data(),
                       x_in_t, norm_attn, dx_normed_attn_all.data() + t * h, h, eps, plus1);
      if (cfg_.mode == TrainMode::FullFT) {
        for (size_t i = 0; i < h; ++i) ft_layers_[l].dattn_norm[i] += dnorm_attn_t[i];
      }
      for (size_t i = 0; i < h; ++i)
        dx_stream[t * h + i] = dx_attn_block[t * h + i] + dx_res_attn[i];
    }
  }

  // Embedding backward (full-FT only).
  if (cfg_.mode == TrainMode::FullFT) {
    for (size_t t = 0; t < T; ++t) {
      size_t tid = std::min<size_t>(tokens[t], ic.vocab_size - 1);
      const float* dx_t = dx_stream.data() + t * h;
      float* demb_row = d_tok_emb_.data() + tid * h;
      for (size_t i = 0; i < h; ++i) demb_row[i] += dx_t[i];
    }
  }
}


float TrainModel::grad_norm() const {
  double sum_sq = 0.0;
  if (cfg_.mode == TrainMode::LoRA) {
    for (const auto& layer : lora_adapters_) {
      for (const auto& la : layer) {
        for (float g : la.dA) sum_sq += static_cast<double>(g) * g;
        for (float g : la.dB) sum_sq += static_cast<double>(g) * g;
      }
    }
  } else {
    for (const auto& fl : ft_layers_) {
      for (float g : fl.attn_q.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.attn_k.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.attn_v.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.attn_o.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.ffn_gate.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.ffn_up.grad)   sum_sq += static_cast<double>(g) * g;
      for (float g : fl.ffn_down.grad) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.dattn_norm) sum_sq += static_cast<double>(g) * g;
      for (float g : fl.dffn_norm)  sum_sq += static_cast<double>(g) * g;
    }
    for (float g : d_tok_emb_) sum_sq += static_cast<double>(g) * g;
    for (float g : d_norm_)    sum_sq += static_cast<double>(g) * g;
    if (!lm_head_tied_)
      for (float g : d_lm_head_) sum_sq += static_cast<double>(g) * g;
  }
  return static_cast<float>(std::sqrt(sum_sq));
}

void TrainModel::clip_grad_norm(float max_norm) {
  if (max_norm <= 0.0f) return;
  float norm = grad_norm();
  if (norm <= max_norm) return;
  float scale = max_norm / (norm + 1e-6f);
  if (cfg_.mode == TrainMode::LoRA) {
    for (auto& layer : lora_adapters_) {
      for (auto& la : layer) {
        for (float& g : la.dA) g *= scale;
        for (float& g : la.dB) g *= scale;
      }
    }
  } else {
    for (auto& fl : ft_layers_) {
      for (float& g : fl.attn_q.grad) g *= scale;
      for (float& g : fl.attn_k.grad) g *= scale;
      for (float& g : fl.attn_v.grad) g *= scale;
      for (float& g : fl.attn_o.grad) g *= scale;
      for (float& g : fl.ffn_gate.grad) g *= scale;
      for (float& g : fl.ffn_up.grad)   g *= scale;
      for (float& g : fl.ffn_down.grad) g *= scale;
      for (float& g : fl.dattn_norm) g *= scale;
      for (float& g : fl.dffn_norm)  g *= scale;
    }
    for (float& g : d_tok_emb_) g *= scale;
    for (float& g : d_norm_)    g *= scale;
    if (!lm_head_tied_)
      for (float& g : d_lm_head_) g *= scale;
  }
}

void TrainModel::zero_grads() {
  if (cfg_.mode == TrainMode::LoRA) {
    for (auto& layer : lora_adapters_)
      for (auto& la : layer) la.zero_grads();
  } else {
    for (auto& fl : ft_layers_) fl.zero_grad();
    std::fill(d_tok_emb_.begin(), d_tok_emb_.end(), 0.0f);
    std::fill(d_norm_.begin(),    d_norm_.end(),    0.0f);
    if (!lm_head_tied_) std::fill(d_lm_head_.begin(), d_lm_head_.end(), 0.0f);
  }
}

void TrainModel::optimizer_step(float lr, int t) {
  const InferenceConfig& ic = base_->config();
  float b1 = cfg_.adamw.beta1, b2 = cfg_.adamw.beta2;
  float eps_a = cfg_.adamw.eps, wd = cfg_.adamw.weight_decay;

  if (cfg_.mode == TrainMode::LoRA) {
    for (auto& layer : lora_adapters_)
      for (auto& la : layer) la.adamw_step(lr, b1, b2, eps_a, wd, t);
  } else {
    ++adamw_t_;
    float bc1 = 1.0f - std::pow(b1, static_cast<float>(adamw_t_));
    float bc2 = 1.0f - std::pow(b2, static_cast<float>(adamw_t_));
    auto adam_step = [&](float* p, float* g, float* m, float* v, size_t n,
                          bool skip_wd) {
      for (size_t i = 0; i < n; ++i) {
        float gi = g[i] + (skip_wd ? 0.0f : wd * p[i]);
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps_a);
      }
    };

    const size_t h = ic.hidden_size;
    for (auto& fl : ft_layers_) {
      auto step_w = [&](FTWeight& w, bool skip_wd) {
        adam_step(w.W.data(), w.grad.data(), w.m.data(), w.v.data(),
                   w.W.size(), skip_wd);
      };
      step_w(fl.attn_q, false);
      step_w(fl.attn_k, false);
      step_w(fl.attn_v, false);
      step_w(fl.attn_o, false);
      step_w(fl.ffn_gate, false);
      step_w(fl.ffn_up, false);
      step_w(fl.ffn_down, false);
      // Norm weights: skip WD.
      adam_step(fl.attn_norm.data(), fl.dattn_norm.data(),
                fl.m_attn_norm.data(), fl.v_attn_norm.data(), h, true);
      adam_step(fl.ffn_norm.data(), fl.dffn_norm.data(),
                fl.m_ffn_norm.data(), fl.v_ffn_norm.data(), h, true);
    }
    // Embedding + norm.
    adam_step(ft_tok_emb_.data(), d_tok_emb_.data(),
              m_tok_emb_.data(), v_tok_emb_.data(), ft_tok_emb_.size(), false);
    adam_step(ft_norm_.data(), d_norm_.data(),
              m_norm_.data(), v_norm_.data(), h, true);
    if (!lm_head_tied_) {
      adam_step(ft_lm_head_.data(), d_lm_head_.data(),
                m_lm_head_.data(), v_lm_head_.data(), ft_lm_head_.size(), false);
    }
  }
}

size_t TrainModel::activation_bytes() const {
  size_t total = 0;
  for (const auto& sa : saved_) {
    total += sa.x_in.size() + sa.normed_attn.size() + sa.q_rope.size()
           + sa.k_rope.size() + sa.v_save.size() + sa.attn_w.size()
           + sa.attn_out_proj.size() + sa.x_after_attn.size()
           + sa.normed_ffn.size() + sa.gate_pre.size() + sa.up_pre.size()
           + sa.ffn_act.size();
  }
  total += x_stream_.size() + logits_save_.size();
  return total * sizeof(float);
}

}  // namespace oxidize
