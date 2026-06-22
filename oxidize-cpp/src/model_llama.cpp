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

#include <cmath>
#include <stdexcept>
#include <string>

#include "oxidize/tensor.hpp"

namespace oxidize {

namespace {

// llama.cpp / inference.rs "keep quantized" rule: weight matrices stay in their
// block layout (decoded on the fly by gemv_quantized); norms/biases dequantize.
bool is_supported_quant_gemv(QuantType q) {
  switch (q) {
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

void LlamaModel::reject_unsupported(const InferenceConfig& cfg) {
  switch (cfg.architecture) {
    case Architecture::Mixtral:
    case Architecture::DeepSeek:
    case Architecture::MiniMax:
    case Architecture::Lfm2:
    case Architecture::Lfm2Moe:
      throw std::runtime_error(
          "LlamaModel: MoE / MLA / shortconv architecture is out of Phase 1 scope");
    default:
      break;
  }
  if (cfg.num_experts > 0) {
    throw std::runtime_error("LlamaModel: MoE FFN (num_experts > 0) is out of Phase 1 scope");
  }
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

    if (layer.attn_q.empty()) {
      throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                               " has no attn_q (non-dense-attention layer unsupported in Phase 1)");
    }
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
    rms_norm(normed.data(), x_.data(), layer.attn_norm.data(), h, eps, plus_one);

    // QKV projections (each consumes the same normed input).
    gemv_weight(layer.attn_q, q_len, h, normed.data(), q.data());
    gemv_weight(layer.attn_k, kv_len, h, normed.data(), k.data());
    gemv_weight(layer.attn_v, kv_len, h, normed.data(), v.data());

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
        rms_norm(head_scratch.data(), hp, layer.attn_q_norm.data(), head_dim, eps, plus_one);
        for (size_t i = 0; i < head_dim; ++i) hp[i] = head_scratch[i];
      }
    } else if (!layer.attn_q_norm.empty() && layer.attn_q_norm.size() == q_len) {
      std::vector<float> tmp(q_len);
      rms_norm(tmp.data(), q.data(), layer.attn_q_norm.data(), q_len, eps, plus_one);
      q = tmp;
    }
    if (!layer.attn_k_norm.empty() && layer.attn_k_norm.size() == head_dim) {
      for (size_t head = 0; head < kv_heads; ++head) {
        float* hp = k.data() + head * head_dim;
        rms_norm(head_scratch.data(), hp, layer.attn_k_norm.data(), head_dim, eps, plus_one);
        for (size_t i = 0; i < head_dim; ++i) hp[i] = head_scratch[i];
      }
    } else if (!layer.attn_k_norm.empty() && layer.attn_k_norm.size() == kv_len) {
      std::vector<float> tmp(kv_len);
      rms_norm(tmp.data(), k.data(), layer.attn_k_norm.data(), kv_len, eps, plus_one);
      k = tmp;
    }

    // RoPE (partial-aware via apply_rope's rope_dim parameter; 0 => full head).
    apply_rope(q.data(), head_dim, n_heads, pos, rope_theta, rope_dim);
    apply_rope(k.data(), head_dim, kv_heads, pos, rope_theta, rope_dim);

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
    attention_decode(attn_result.data(), q.data(), key_prefix, val_prefix,
                     eff_seq_len, n_heads, kv_heads, head_dim);

    // Output projection: [q_len] -> [h].
    gemv_weight(layer.attn_output, h, q_len, attn_result.data(), attn_out.data());
    if (!layer.attn_output_bias.empty()) {
      size_t bl = layer.attn_output_bias.size();
      for (size_t i = 0; i < h; ++i) attn_out[i] += layer.attn_output_bias[i % bl];
    }

    // Gemma "sandwich" norm: post-attention RMSNorm applied to the attention
    // output *before* the residual add (inference.rs sandwich_norm path).
    if (cfg.sandwich_norm && !layer.post_attention_norm.empty()) {
      rms_norm(attn_out.data(), attn_out.data(), layer.post_attention_norm.data(),
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
    if (ffn_norm_w.empty() || layer.ffn_gate.empty() || layer.ffn_up.empty() ||
        layer.ffn_down.empty()) {
      throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                               " missing dense FFN weights");
    }

    rms_norm(normed.data(), x_.data(), ffn_norm_w.data(), h, eps, plus_one);
    gemv_weight(layer.ffn_gate, inter, h, normed.data(), gate.data());
    gemv_weight(layer.ffn_up, inter, h, normed.data(), up.data());

    if (cfg.gelu_ffn) {
      geglu_inplace(gate.data(), up.data(), gate.data(), inter);
    } else {
      swiglu_inplace(gate.data(), up.data(), gate.data(), inter);
    }

    gemv_weight(layer.ffn_down, h, inter, gate.data(), ffn_out.data());
    if (!layer.ffn_down_bias.empty()) {
      size_t bl = layer.ffn_down_bias.size();
      for (size_t i = 0; i < h; ++i) ffn_out[i] += layer.ffn_down_bias[i % bl];
    }

    // Gemma "sandwich" norm: post-FFN RMSNorm applied to the FFN output *before*
    // the residual add (inference.rs sandwich_norm path).
    if (cfg.sandwich_norm && !layer.post_ffn_norm.empty()) {
      rms_norm(ffn_out.data(), ffn_out.data(), layer.post_ffn_norm.data(),
               h, eps, plus_one);
    }

    for (size_t i = 0; i < h; ++i) x_[i] += ffn_out[i];
  }
}

Logits LlamaModel::final_head() {
  const size_t h = config_.hidden_size;
  const size_t vocab = config_.vocab_size;
  std::vector<float> normed(h, 0.0f);
  rms_norm(normed.data(), x_.data(), norm_weight_.data(), h, config_.rms_norm_eps,
           config_.rms_norm_weight_plus_one);

  Logits logits(vocab, 0.0f);
  const LlamaWeight& head = output_tied_ ? tok_embeddings_ : output_weight_;
  gemv_weight(head, vocab, h, normed.data(), logits.data());
  return logits;
}

Logits LlamaModel::forward_single(Token token, size_t pos, bool need_logits) {
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

std::unique_ptr<Model> load_llama_gguf(const std::string& path) {
  GgufModel gguf = GgufModel::load(path);
  return std::make_unique<LlamaModel>(std::move(gguf));
}

}  // namespace oxidize
