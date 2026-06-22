// Llama-family DENSE inference hot path.
//
// Ported from oxidize-core/src/model/inference.rs (InferenceModel::
// load_from_gguf, embed_token_into_workspace, run_layer_range_in_workspace,
// final_head_from_workspace, impl Model for InferenceModel) and
// oxidize-core/src/compute/kv_cache.rs (layer-major F32 KV cache layout,
// token_slot_index = layer*context_size + position).
//
// Numerically faithful to the Rust scalar reference path. Out-of-Phase-1
// architectures throw std::runtime_error; dense Llama/Mistral/Qwen3-dense/
// Gemma-dense are fully implemented (GQA, partial RoPE, Qwen (1+w) RMSNorm,
// Gemma GeGLU + embedding scale, tied embeddings, uniform sliding window).

#include "oxidize/model_llama.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "oxidize/tensor.hpp"

#ifdef OXIDIZE_CUDA
#include "oxidize/cuda_backend.hpp"
#endif

namespace oxidize {

namespace {

// llama.cpp / inference.rs "keep quantized" rule: weight matrices stay in their
// block layout (decoded on the fly by gemv_quantized); norms/biases dequantize.
bool is_supported_quant_gemv(QuantType q) {
  switch (q) {
    // Keep 16-bit float weights NATIVE (2 bytes/param) and dequantize per row in
    // gemv_quantized, instead of expanding to f32 at load (4 bytes/param). Decode
    // is memory-bandwidth bound, so halving the weight bytes read per token is a
    // large speedup for F16/BF16 models.
    case QuantType::F16:
    case QuantType::BF16:
    case QuantType::Q8_0:
    case QuantType::Q4_K_S:
    case QuantType::Q4_K_M:
    case QuantType::Q6_K:
    case QuantType::IQ1_S:
    case QuantType::IQ1_M:
    case QuantType::NVFP4:
      return true;
    default:
      return false;
  }
}

// True if `layer_idx` (0-based) is a global-attention layer. Mirrors
// InferenceConfig::layer_is_global. For dense Llama/Mistral/Qwen there is no
// interleave pattern, so uniform-SWA models report every layer local.
bool layer_is_global(const InferenceConfig& cfg, size_t layer_idx) {
  if (cfg.sliding_window == 0) return true;
  if (cfg.sliding_window_pattern == 0) return false;
  return (layer_idx + 1) % cfg.sliding_window_pattern == 0;
}

// Effective sliding-window size for a layer (0 = full attention). Mirrors
// InferenceConfig::layer_sliding_window.
size_t layer_sliding_window(const InferenceConfig& cfg, size_t layer_idx) {
  if (cfg.sliding_window > 0 && !layer_is_global(cfg, layer_idx)) {
    return cfg.sliding_window;
  }
  return 0;
}

// Per-layer RoPE theta. Mirrors InferenceConfig::layer_rope_theta. Dense
// Llama/Mistral/Qwen leave rope_theta_swa == 0, so this is always rope_theta.
float layer_rope_theta(const InferenceConfig& cfg, size_t layer_idx) {
  if (cfg.rope_theta_swa > 0.0f && !layer_is_global(cfg, layer_idx)) {
    return cfg.rope_theta_swa;
  }
  return cfg.rope_theta;
}

// Effective RoPE dimension. Mirrors InferenceConfig::effective_rope_dim.
size_t effective_rope_dim(const InferenceConfig& cfg) {
  size_t kvhd = cfg.head_dim();  // config.hpp head_dim() already honors key_value_head_dim
  if (cfg.rope_dim > 0) {
    return cfg.rope_dim < kvhd ? cfg.rope_dim : kvhd;
  }
  return kvhd;
}

// y = W * x, dispatching on storage kind. rows = output features, cols = input.
void gemv_weight(const LlamaWeight& w, size_t rows, size_t cols, const float* x,
                 float* y) {
  if (w.quantized) {
    gemv_quantized(y, w.quant, w.data, rows, cols, x);
  } else {
    matvec(y, w.f32.data(), x, rows, cols);
  }
}

}  // namespace

// --- backend dispatch -------------------------------------------------------
// Each helper routes to the CUDA backend when use_cuda_ (weights resident on the
// GPU, activation vectors transferred per op) or the CPU tensor.hpp kernels.

bool LlamaModel::set_cuda(bool on) {
#ifdef OXIDIZE_CUDA
  use_cuda_ = on && CudaBackend::available();
#else
  (void)on;
  use_cuda_ = false;
#endif
  return use_cuda_;
}

void LlamaModel::d_rms_norm(float* out, const float* x, const float* w, size_t n,
                            float eps, bool plus_one) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    CudaBackend::instance().rms_norm(out, x, w, n, eps, plus_one);
    return;
  }
#endif
  rms_norm(out, x, w, n, eps, plus_one);
}

void LlamaModel::d_apply_rope(float* vec, size_t head_dim, size_t num_heads,
                              size_t pos, float theta, size_t rope_dim) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    CudaBackend::instance().apply_rope(vec, head_dim, num_heads, pos, theta,
                                       rope_dim);
    return;
  }
#endif
  apply_rope(vec, head_dim, num_heads, pos, theta, rope_dim);
}

void LlamaModel::d_swiglu(float* gate, const float* up, float* out, size_t n) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    CudaBackend::instance().swiglu_inplace(gate, up, out, n);
    return;
  }
#endif
  swiglu_inplace(gate, up, out, n);
}

void LlamaModel::d_geglu(float* gate, const float* up, float* out, size_t n) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    CudaBackend::instance().geglu_inplace(gate, up, out, n);
    return;
  }
#endif
  geglu_inplace(gate, up, out, n);
}

void LlamaModel::d_attention(float* out, const float* q, const float* k_cache,
                             const float* v_cache, size_t seq_len,
                             size_t num_heads, size_t kv_heads,
                             size_t head_dim) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    CudaBackend::instance().attention_decode(out, q, k_cache, v_cache, seq_len,
                                             num_heads, kv_heads, head_dim);
    return;
  }
#endif
  attention_decode(out, q, k_cache, v_cache, seq_len, num_heads, kv_heads,
                   head_dim);
}

void LlamaModel::d_gemv_weight(const LlamaWeight& w, size_t rows, size_t cols,
                               const float* x, float* y) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_) {
    if (w.quantized) {
      CudaBackend::instance().gemv_quantized(y, w.quant, w.data, rows, cols, x);
    } else {
      CudaBackend::instance().matvec(y, w.f32.data(), x, rows, cols);
    }
    return;
  }
#endif
  gemv_weight(w, rows, cols, x, y);
}

void LlamaModel::reject_unsupported(const InferenceConfig& cfg) {
  switch (cfg.architecture) {
    case Architecture::DeepSeek:  // MLA compressed attention (unsupported)
    case Architecture::MiniMax:   // lightning attention
    case Architecture::Lfm2:      // short-conv token mixing
    case Architecture::Lfm2Moe:
      throw std::runtime_error(
          "LlamaModel: MLA / lightning-attn / shortconv architecture is unsupported");
    default:
      break;
  }
  // Mixtral / Qwen-MoE (standard attention + routed-expert FFN) ARE supported
  // via moe_ffn(); only MoE that also needs MLA/shortconv attention is rejected
  // by the arch switch above.
  if (cfg.shortconv_l_cache > 0) {
    throw std::runtime_error("LlamaModel: LFM2 shortconv is out of Phase 1 scope");
  }
  if (cfg.nextn_predict_layers > 0) {
    throw std::runtime_error("LlamaModel: MTP/nextn draft layers are out of Phase 1 scope");
  }
  // Gemma-family interleaved global/local sliding window is out of Phase 1.
  // Uniform sliding window (Mistral/Qwen, pattern == 0) IS supported below.
  if (cfg.sliding_window > 0 && cfg.sliding_window_pattern > 0) {
    throw std::runtime_error(
        "LlamaModel: interleaved sliding-window attention (Gemma2/3) is out of Phase 1 scope");
  }
}

std::vector<float> LlamaModel::load_vector(const GgufModel& g, const std::string& name) {
  TensorView tv = g.tensor(name);
  size_t count = 1;
  for (uint64_t d : tv.dims) count *= static_cast<size_t>(d);
  std::vector<float> out(count, 0.0f);
  dequantize_row(tv.quant, tv.data, out.data(), count);
  return out;
}

LlamaWeight LlamaModel::load_weight(const GgufModel& g, const std::string& name,
                                    bool keep_quantized) {
  TensorView tv = g.tensor(name);
  size_t count = 1;
  for (uint64_t d : tv.dims) count *= static_cast<size_t>(d);

  LlamaWeight w;
  // GGUF stores linear weights as [in_features, out_features] (dims[0] = cols
  // = input, dims[1] = rows = output). cols/rows are recovered by callers from
  // the known input dimension; here we keep the total element count and let the
  // forward pass derive rows = count / cols. We store cols/rows when both dims
  // are present for clarity / validation.
  if (tv.dims.size() >= 2) {
    w.cols = static_cast<size_t>(tv.dims[0]);
    w.rows = static_cast<size_t>(tv.dims[1]);
  } else if (tv.dims.size() == 1) {
    w.cols = static_cast<size_t>(tv.dims[0]);
    w.rows = 1;
  }

  if (keep_quantized && is_supported_quant_gemv(tv.quant)) {
    w.quantized = true;
    w.quant = tv.quant;
    w.data = tv.data;
  } else {
    w.quantized = false;
    w.quant = QuantType::F32;
    w.f32.assign(count, 0.0f);
    dequantize_row(tv.quant, tv.data, w.f32.data(), count);
  }
  return w;
}

LlamaModel::LlamaModel(GgufModel gguf) : gguf_(std::move(gguf)) {
  config_ = build_inference_config(gguf_);
  reject_unsupported(config_);

  const GgufModel& g = gguf_;
  const size_t h = config_.hidden_size;

  // ---- Embeddings (token_embd / tok_embeddings) ----
  std::string embd_name = g.has_tensor("token_embd.weight") ? "token_embd.weight"
                                                            : "tok_embeddings.weight";
  if (!g.has_tensor(embd_name)) {
    throw std::runtime_error("LlamaModel: missing token embedding tensor");
  }
  {
    TensorView tv = g.tensor(embd_name);
    tok_embeddings_cols_ = tv.dims.size() >= 2 ? static_cast<size_t>(tv.dims[1]) : h;
  }
  // Keep embeddings quantized when supported (decoded per-token at lookup), to
  // mirror inference.rs lookup_quantized_embedding.
  tok_embeddings_ = load_weight(g, embd_name, /*keep_quantized=*/true);

  // ---- Final norm (output_norm / norm / token_embd_norm) ----
  if (g.has_tensor("output_norm.weight")) {
    norm_weight_ = load_vector(g, "output_norm.weight");
  } else if (g.has_tensor("norm.weight")) {
    norm_weight_ = load_vector(g, "norm.weight");
  } else if (g.has_tensor("token_embd_norm.weight")) {
    norm_weight_ = load_vector(g, "token_embd_norm.weight");
  } else {
    throw std::runtime_error("LlamaModel: missing final norm weight");
  }

  // ---- Output head (lm_head). Tied to embeddings when absent. ----
  if (g.has_tensor("output.weight")) {
    output_weight_ = load_weight(g, "output.weight", /*keep_quantized=*/true);
    output_tied_ = false;
  } else {
    output_tied_ = true;  // reuse tok_embeddings_ as lm_head
  }

  // ---- Per-layer dense weights ----
  layers_.resize(config_.layer_count);
  for (size_t l = 0; l < config_.layer_count; ++l) {
    LlamaLayer& layer = layers_[l];
    const std::string p = "blk." + std::to_string(l) + ".";

    auto opt_vec = [&](const std::string& suffix, std::vector<float>& dst) {
      if (g.has_tensor(p + suffix)) dst = load_vector(g, p + suffix);
    };
    auto opt_w = [&](const std::string& suffix, LlamaWeight& dst) {
      if (g.has_tensor(p + suffix)) dst = load_weight(g, p + suffix, /*keep_quantized=*/true);
    };

    opt_vec("attn_norm.weight", layer.attn_norm);
    opt_w("attn_q.weight", layer.attn_q);
    opt_vec("attn_q.bias", layer.attn_q_bias);
    opt_w("attn_k.weight", layer.attn_k);
    opt_vec("attn_k.bias", layer.attn_k_bias);
    opt_w("attn_v.weight", layer.attn_v);
    opt_vec("attn_v.bias", layer.attn_v_bias);
    opt_w("attn_output.weight", layer.attn_output);
    opt_vec("attn_output.bias", layer.attn_output_bias);
    opt_vec("attn_q_norm.weight", layer.attn_q_norm);
    opt_vec("attn_k_norm.weight", layer.attn_k_norm);

    opt_vec("ffn_norm.weight", layer.ffn_norm);
    opt_vec("post_attention_norm.weight", layer.post_attention_norm);
    if (g.has_tensor(p + "post_ffw_norm.weight")) {
      layer.post_ffn_norm = load_vector(g, p + "post_ffw_norm.weight");
    } else {
      opt_vec("post_ffn_norm.weight", layer.post_ffn_norm);
    }
    opt_w("ffn_gate.weight", layer.ffn_gate);
    opt_w("ffn_up.weight", layer.ffn_up);
    opt_w("ffn_down.weight", layer.ffn_down);
    opt_vec("ffn_down.bias", layer.ffn_down_bias);

    // MoE routed experts (Mixtral / Qwen-MoE). Present only on MoE layers.
    opt_w("ffn_gate_exps.weight", layer.ffn_gate_exps);
    opt_w("ffn_up_exps.weight", layer.ffn_up_exps);
    opt_w("ffn_down_exps.weight", layer.ffn_down_exps);
    opt_w("ffn_gate_inp.weight", layer.ffn_gate_inp);
    opt_vec("exp_probs_b.bias", layer.ffn_exp_probs_b);

    if (layer.attn_q.empty()) {
      throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                               " has no attn_q (non-dense-attention layer unsupported in Phase 1)");
    }
    if (layer.is_moe()) any_moe_ = true;
  }

  // ---- KV cache (F32, layer-major). Only attention layers are stored; in the
  // dense path every layer is an attention layer, so attn_layer_count ==
  // layer_count and the layer index maps directly. ----
  kv_token_size_ = config_.num_key_value_heads * config_.head_dim();
  size_t kv_elems = config_.layer_count * config_.context_size * kv_token_size_;
  kv_keys_.assign(kv_elems, 0.0f);
  kv_values_.assign(kv_elems, 0.0f);

  x_.assign(h, 0.0f);
}

void LlamaModel::embed_token(Token token, float* x) const {
  const size_t h = config_.hidden_size;
  for (size_t i = 0; i < h; ++i) x[i] = 0.0f;
  size_t token_idx = static_cast<size_t>(token);
  if (config_.vocab_size > 0 && token_idx >= config_.vocab_size) {
    token_idx = config_.vocab_size - 1;
  }
  if (tok_embeddings_.quantized) {
    size_t bvals = quant_block_values(tok_embeddings_.quant);
    size_t bbytes = quant_block_bytes(tok_embeddings_.quant);
    size_t blocks_per_row = h / bvals;
    size_t row_bytes = blocks_per_row * bbytes;
    const uint8_t* row = tok_embeddings_.data + token_idx * row_bytes;
    dequantize_row(tok_embeddings_.quant, row, x, h);
  } else {
    const float* row = tok_embeddings_.f32.data() + token_idx * h;
    for (size_t i = 0; i < h; ++i) x[i] = row[i];
  }
  if (config_.embedding_scale != 1.0f) {
    for (size_t i = 0; i < h; ++i) x[i] *= config_.embedding_scale;
  }
}

void LlamaModel::moe_ffn(const LlamaLayer& layer, const float* normed,
                         float* ffn_out) {
  const InferenceConfig& cfg = config_;
  const size_t h = cfg.hidden_size;
  const size_t i_size = cfg.expert_intermediate_size > 0
                            ? cfg.expert_intermediate_size
                            : cfg.intermediate_size;
  const size_t n_experts = cfg.num_experts;
  if (n_experts == 0)
    throw std::runtime_error("moe_ffn: layer has experts but num_experts==0");
  const size_t n_sel =
      std::min(std::max<size_t>(cfg.num_experts_per_tok, 1), n_experts);
  const bool sigmoid = cfg.expert_gating_sigmoid;

  // Expert e's [rows,cols] slice of a 3D [n_experts,rows,cols] weight.
  auto gemv_expert = [&](const LlamaWeight& w, size_t e, size_t rows,
                         size_t cols, const float* x, float* y) {
    if (w.quantized) {
      size_t rb = quantized_size(w.quant, cols);
      gemv_quantized(y, w.quant, w.data + e * rows * rb, rows, cols, x);
    } else {
      matvec(y, w.f32.data() + e * rows * cols, x, rows, cols);
    }
  };

  // 1. Router logits.
  std::vector<float> rw(n_experts, 0.0f);  // routing weight (renorm source)
  gemv_weight(layer.ffn_gate_inp, n_experts, h, normed, rw.data());

  // 2. Gating: softmax (Mixtral/Qwen) or sigmoid+bias (LFM2MoE). sel = selection
  // score, rw = routing weight.
  std::vector<std::pair<size_t, float>> sel(n_experts);  // (idx, selection score)
  if (sigmoid) {
    for (size_t i = 0; i < n_experts; ++i) rw[i] = 1.0f / (1.0f + std::exp(-rw[i]));
    for (size_t i = 0; i < n_experts; ++i) {
      float bias = i < layer.ffn_exp_probs_b.size() ? layer.ffn_exp_probs_b[i] : 0.0f;
      sel[i] = {i, rw[i] + bias};
    }
  } else {
    float mx = -std::numeric_limits<float>::infinity();
    for (float v : rw) mx = std::max(mx, v);
    float sum = 0.0f;
    for (float& v : rw) { v = std::exp(v - mx); sum += v; }
    if (sum > 0.0f) for (float& v : rw) v /= sum;
    for (size_t i = 0; i < n_experts; ++i) sel[i] = {i, rw[i]};
  }

  // 3. Top-k by selection score (desc).
  auto by_score = [](const std::pair<size_t, float>& a,
                     const std::pair<size_t, float>& b) { return a.second > b.second; };
  if (n_sel < n_experts)
    std::partial_sort(sel.begin(), sel.begin() + n_sel, sel.end(), by_score);
  else
    std::sort(sel.begin(), sel.end(), by_score);

  // 4. Renormalize routing weights over selected; apply routed scale.
  float wnorm = 0.0f;
  for (size_t s = 0; s < n_sel; ++s) wnorm += rw[sel[s].first];
  if (wnorm <= 0.0f) wnorm = 1.0f;
  float scale = cfg.expert_weights_scale > 0.0f ? cfg.expert_weights_scale : 1.0f;

  // 5. Per-expert FFN, accumulated into ffn_out.
  std::vector<float> gate(i_size), up(i_size), down(h);
  for (size_t s = 0; s < n_sel; ++s) {
    size_t e = sel[s].first;
    float w = scale * rw[e] / wnorm;
    gemv_expert(layer.ffn_gate_exps, e, i_size, h, normed, gate.data());
    gemv_expert(layer.ffn_up_exps, e, i_size, h, normed, up.data());
    swiglu_inplace(gate.data(), up.data(), gate.data(), i_size);
    gemv_expert(layer.ffn_down_exps, e, h, i_size, gate.data(), down.data());
    for (size_t i = 0; i < h; ++i) ffn_out[i] += w * down[i];
  }
}

void LlamaModel::run_layers(size_t pos) {
  const InferenceConfig& cfg = config_;
  const size_t h = cfg.hidden_size;
  const size_t n_heads = cfg.num_attention_heads;
  const size_t kv_heads = cfg.num_key_value_heads;
  const size_t head_dim = cfg.head_dim();
  const size_t q_len = n_heads * head_dim;
  const size_t kv_len = kv_heads * head_dim;
  const size_t inter = cfg.intermediate_size;
  const float eps = cfg.rms_norm_eps;
  const bool plus_one = cfg.rms_norm_weight_plus_one;
  const size_t seq_len = pos + 1;
  const size_t rope_dim = cfg.rope_dim;  // 0 => full head_dim (apply_rope convention)

  // Reused scratch buffers (one decode step).
  std::vector<float> normed(h);
  std::vector<float> q(q_len), k(kv_len), v(kv_len);
  std::vector<float> attn_result(q_len);
  std::vector<float> attn_out(h);
  std::vector<float> gate(inter), up(inter);
  std::vector<float> ffn_out(h);
  std::vector<float> head_scratch(head_dim);

  for (size_t l = 0; l < cfg.layer_count; ++l) {
    const LlamaLayer& layer = layers_[l];
    const float rope_theta = layer_rope_theta(cfg, l);
    const size_t window = layer_sliding_window(cfg, l);

    // ---- Attention block ----
    // pre-attention RMSNorm
    d_rms_norm(normed.data(), x_.data(), layer.attn_norm.data(), h, eps, plus_one);

    // QKV projections (each consumes the same normed input).
    d_gemv_weight(layer.attn_q, q_len, h, normed.data(), q.data());
    d_gemv_weight(layer.attn_k, kv_len, h, normed.data(), k.data());
    d_gemv_weight(layer.attn_v, kv_len, h, normed.data(), v.data());

    if (!layer.attn_q_bias.empty()) {
      size_t bl = layer.attn_q_bias.size();
      for (size_t i = 0; i < q_len; ++i) q[i] += layer.attn_q_bias[i % bl];
    }
    if (!layer.attn_k_bias.empty()) {
      size_t bl = layer.attn_k_bias.size();
      for (size_t i = 0; i < kv_len; ++i) k[i] += layer.attn_k_bias[i % bl];
    }
    if (!layer.attn_v_bias.empty()) {
      size_t bl = layer.attn_v_bias.size();
      for (size_t i = 0; i < kv_len; ++i) v[i] += layer.attn_v_bias[i % bl];
    }

    // Per-head Q/K RMSNorm (Qwen3). Applied over head_dim when the norm weight
    // matches the per-head width (the dense case used by Qwen3).
    if (!layer.attn_q_norm.empty() && layer.attn_q_norm.size() == head_dim) {
      for (size_t head = 0; head < n_heads; ++head) {
        float* hp = q.data() + head * head_dim;
        d_rms_norm(head_scratch.data(), hp, layer.attn_q_norm.data(), head_dim, eps, plus_one);
        for (size_t i = 0; i < head_dim; ++i) hp[i] = head_scratch[i];
      }
    } else if (!layer.attn_q_norm.empty() && layer.attn_q_norm.size() == q_len) {
      std::vector<float> tmp(q_len);
      d_rms_norm(tmp.data(), q.data(), layer.attn_q_norm.data(), q_len, eps, plus_one);
      q = tmp;
    }
    if (!layer.attn_k_norm.empty() && layer.attn_k_norm.size() == head_dim) {
      for (size_t head = 0; head < kv_heads; ++head) {
        float* hp = k.data() + head * head_dim;
        d_rms_norm(head_scratch.data(), hp, layer.attn_k_norm.data(), head_dim, eps, plus_one);
        for (size_t i = 0; i < head_dim; ++i) hp[i] = head_scratch[i];
      }
    } else if (!layer.attn_k_norm.empty() && layer.attn_k_norm.size() == kv_len) {
      std::vector<float> tmp(kv_len);
      d_rms_norm(tmp.data(), k.data(), layer.attn_k_norm.data(), kv_len, eps, plus_one);
      k = tmp;
    }

    // RoPE (partial-aware via apply_rope's rope_dim parameter; 0 => full head).
    d_apply_rope(q.data(), head_dim, n_heads, pos, rope_theta, rope_dim);
    d_apply_rope(k.data(), head_dim, kv_heads, pos, rope_theta, rope_dim);

    // Append K/V to the layer-major cache at physical position pos.
    // token_slot_index = l*context_size + (pos % context_size); F32 storage.
    size_t phys = pos % cfg.context_size;
    size_t base = (l * cfg.context_size + phys) * kv_token_size_;
    for (size_t i = 0; i < kv_len; ++i) {
      kv_keys_[base + i] = k[i];
      kv_values_[base + i] = v[i];
    }

    // GQA causal attention over the contiguous [0, seq_len) prefix.
    // Layer prefix start is at position 0 within this layer's slice.
    size_t layer_start = (l * cfg.context_size) * kv_token_size_;
    size_t eff_seq_len = seq_len;
    const float* key_prefix = kv_keys_.data() + layer_start;
    const float* val_prefix = kv_values_.data() + layer_start;
    if (window > 0 && seq_len > window) {
      // Uniform sliding window: attend only to the most recent `window`
      // positions. RoPE encodes absolute positions, so slicing the oldest rows
      // yields the windowed-causal mask with relative positions preserved.
      size_t skip = (seq_len - window) * kv_token_size_;
      key_prefix += skip;
      val_prefix += skip;
      eff_seq_len = window;
    }
    d_attention(attn_result.data(), q.data(), key_prefix, val_prefix,
                eff_seq_len, n_heads, kv_heads, head_dim);

    // Output projection: [q_len] -> [h].
    d_gemv_weight(layer.attn_output, h, q_len, attn_result.data(), attn_out.data());
    if (!layer.attn_output_bias.empty()) {
      size_t bl = layer.attn_output_bias.size();
      for (size_t i = 0; i < h; ++i) attn_out[i] += layer.attn_output_bias[i % bl];
    }

    // Gemma "sandwich" norm: post-attention RMSNorm applied to the attention
    // output *before* the residual add (inference.rs sandwich_norm path).
    if (cfg.sandwich_norm && !layer.post_attention_norm.empty()) {
      d_rms_norm(attn_out.data(), attn_out.data(), layer.post_attention_norm.data(),
                 h, eps, plus_one);
    }

    // Residual add.
    for (size_t i = 0; i < h; ++i) x_[i] += attn_out[i];

    // ---- FFN block ----
    // Pre-FFN RMSNorm weight. Only fall back to post_attention_norm when NOT in
    // sandwich-norm mode, where post_attention_norm is the sandwich weight and
    // must not be reused as the pre-FFN norm.
    const std::vector<float>& ffn_norm_w =
        !layer.ffn_norm.empty()
            ? layer.ffn_norm
            : (cfg.sandwich_norm ? layer.ffn_norm : layer.post_attention_norm);
    if (ffn_norm_w.empty()) {
      throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                               " missing ffn_norm");
    }
    d_rms_norm(normed.data(), x_.data(), ffn_norm_w.data(), h, eps, plus_one);

    if (layer.is_moe()) {
      // Routed-expert MoE FFN (Mixtral / Qwen-MoE).
      std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
      moe_ffn(layer, normed.data(), ffn_out.data());
    } else {
      if (layer.ffn_gate.empty() || layer.ffn_up.empty() ||
          layer.ffn_down.empty()) {
        throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                                 " missing dense FFN weights");
      }
      d_gemv_weight(layer.ffn_gate, inter, h, normed.data(), gate.data());
      d_gemv_weight(layer.ffn_up, inter, h, normed.data(), up.data());
      if (cfg.gelu_ffn) {
        d_geglu(gate.data(), up.data(), gate.data(), inter);
      } else {
        d_swiglu(gate.data(), up.data(), gate.data(), inter);
      }
      d_gemv_weight(layer.ffn_down, h, inter, gate.data(), ffn_out.data());
      if (!layer.ffn_down_bias.empty()) {
        size_t bl = layer.ffn_down_bias.size();
        for (size_t i = 0; i < h; ++i) ffn_out[i] += layer.ffn_down_bias[i % bl];
      }
    }

    // Gemma "sandwich" norm: post-FFN RMSNorm applied to the FFN output *before*
    // the residual add (inference.rs sandwich_norm path).
    if (cfg.sandwich_norm && !layer.post_ffn_norm.empty()) {
      d_rms_norm(ffn_out.data(), ffn_out.data(), layer.post_ffn_norm.data(),
                 h, eps, plus_one);
    }

    for (size_t i = 0; i < h; ++i) x_[i] += ffn_out[i];
  }
}

Logits LlamaModel::final_head() {
  const size_t h = config_.hidden_size;
  const size_t vocab = config_.vocab_size;
  std::vector<float> normed(h, 0.0f);
  d_rms_norm(normed.data(), x_.data(), norm_weight_.data(), h, config_.rms_norm_eps,
             config_.rms_norm_weight_plus_one);

  Logits logits(vocab, 0.0f);
  const LlamaWeight& head = output_tied_ ? tok_embeddings_ : output_weight_;
  d_gemv_weight(head, vocab, h, normed.data(), logits.data());
  return logits;
}

#ifdef OXIDIZE_CUDA
// Build a CudaBackend::ModelView referencing this model's (host) weights for the
// GPU-resident decode. Pointers are borrowed; the backend caches device copies.
// WIP / UNVERIFIED — see cuda_backend.hpp::resident_forward.
CudaBackend::ModelView LlamaModel::build_cuda_view() const {
  const InferenceConfig& cfg = config_;
  const size_t h = cfg.hidden_size;
  const size_t head_dim = cfg.head_dim();
  const size_t q_len = cfg.num_attention_heads * head_dim;
  const size_t kv_len = cfg.num_key_value_heads * head_dim;
  const size_t inter = cfg.intermediate_size;

  auto mkw = [](const LlamaWeight& w, size_t rows, size_t cols) {
    CudaBackend::WeightView v;
    v.quantized = w.quantized;
    v.quant = w.quant;
    v.data = w.data;
    v.f32 = w.f32.empty() ? nullptr : w.f32.data();
    v.rows = rows;
    v.cols = cols;
    return v;
  };
  auto vptr = [](const std::vector<float>& v) {
    return v.empty() ? nullptr : v.data();
  };

  CudaBackend::ModelView mv;
  mv.cfg = cfg;
  mv.final_norm = norm_weight_.data();
  mv.lm_head = mkw(output_tied_ ? tok_embeddings_ : output_weight_,
                   cfg.vocab_size, h);
  mv.layers.resize(cfg.layer_count);
  for (size_t l = 0; l < cfg.layer_count; ++l) {
    const LlamaLayer& s = layers_[l];
    CudaBackend::LayerView& d = mv.layers[l];
    d.attn_norm = vptr(s.attn_norm);
    d.wq = mkw(s.attn_q, q_len, h);
    d.wk = mkw(s.attn_k, kv_len, h);
    d.wv = mkw(s.attn_v, kv_len, h);
    d.wo = mkw(s.attn_output, h, q_len);
    d.wq_bias = vptr(s.attn_q_bias);
    d.wk_bias = vptr(s.attn_k_bias);
    d.wv_bias = vptr(s.attn_v_bias);
    d.wo_bias = vptr(s.attn_output_bias);
    d.q_bias_len = s.attn_q_bias.size();
    d.k_bias_len = s.attn_k_bias.size();
    d.v_bias_len = s.attn_v_bias.size();
    d.o_bias_len = s.attn_output_bias.size();
    d.attn_q_norm = vptr(s.attn_q_norm);
    d.attn_k_norm = vptr(s.attn_k_norm);
    d.ffn_norm = !s.ffn_norm.empty() ? s.ffn_norm.data() : vptr(s.post_attention_norm);
    d.gate = mkw(s.ffn_gate, inter, h);
    d.up = mkw(s.ffn_up, inter, h);
    d.down = mkw(s.ffn_down, h, inter);
    d.down_bias = vptr(s.ffn_down_bias);
    d.down_bias_len = s.ffn_down_bias.size();
    d.post_attn_norm =
        (cfg.sandwich_norm && !s.post_attention_norm.empty()) ? s.post_attention_norm.data() : nullptr;
    d.post_ffn_norm =
        (cfg.sandwich_norm && !s.post_ffn_norm.empty()) ? s.post_ffn_norm.data() : nullptr;
    d.rope_theta = layer_rope_theta(cfg, l);
    d.sliding_window = layer_sliding_window(cfg, l);
  }
  return mv;
}
#endif

Logits LlamaModel::forward_single(Token token, size_t pos, bool need_logits) {
#ifdef OXIDIZE_CUDA
  if (use_cuda_ && !any_moe_) {
    // Resident GPU decode: one host<->device sync per token (vs ~290 in the
    // per-op path). embed_token dequantizes + scales the row on the host; the
    // rest of the forward stays on the device. (MoE layers fall back to the CPU
    // path below — the resident forward computes a dense FFN only.)
    embed_token(token, x_.data());
    Logits logits(config_.vocab_size, 0.0f);
    CudaBackend::instance().resident_forward(build_cuda_view(), x_.data(), pos,
                                             logits.data());
    return need_logits ? logits : Logits{};
  }
#endif
  embed_token(token, x_.data());
  run_layers(pos);
  if (!need_logits) return Logits{};
  return final_head();
}

Logits LlamaModel::forward(const std::vector<Token>& tokens, Session& session) {
  if (tokens.empty()) {
    throw std::runtime_error("LlamaModel::forward: empty input");
  }
  size_t requested_total = session.consumed_tokens() + tokens.size();
  if (requested_total > config_.context_size) {
    throw std::runtime_error("LlamaModel::forward: context exceeded (requested " +
                             std::to_string(requested_total) + " > context " +
                             std::to_string(config_.context_size) + ")");
  }
  size_t start_pos = session.consumed_tokens();
  Logits logits;
  for (size_t i = 0; i < tokens.size(); ++i) {
    size_t pos = start_pos + i;
    bool need = (i + 1 == tokens.size());
    Logits step = forward_single(tokens[i], pos, need);
    if (need) logits = std::move(step);
  }
  session.record_tokens(tokens.size());
  return logits;
}

void LlamaModel::rewind_to(size_t consumed_tokens) {
  // KV positions are overwritten on the next forward at the corresponding
  // physical slot; the F32 cache requires no explicit truncation because reads
  // are bounded by seq_len = pos + 1. Nothing to do for the dense F32 cache.
  (void)consumed_tokens;
}

std::unique_ptr<Model> load_llama_gguf(const std::string& path, bool want_cuda) {
  GgufModel gguf = GgufModel::load(path);
  auto model = std::make_unique<LlamaModel>(std::move(gguf));
  model->set_cuda(want_cuda);
  return model;
}

}  // namespace oxidize
