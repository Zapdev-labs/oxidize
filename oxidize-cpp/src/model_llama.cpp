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
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Debug helper: print first N floats from a named vector. Gated by OXIDIZE_DEBUG env.
static bool g_debug = false;
static void dbg_vec(const char* label, const float* v, size_t n, size_t show = 8) {
  if (!g_debug) return;
  std::fprintf(stderr, "DBG %-40s [", label);
  for (size_t i = 0; i < std::min(n, show); ++i)
    std::fprintf(stderr, "%s%.4f", i ? " " : "", v[i]);
  std::fprintf(stderr, "]\n");
}
static void dbg_topk(const char* label, const float* v, size_t n, size_t k = 5) {
  if (!g_debug) return;
  // find top-k indices by value
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + std::min(k, n), idx.end(),
                    [&](size_t a, size_t b){ return v[a] > v[b]; });
  std::fprintf(stderr, "DBG %-40s top%zu: ", label, k);
  for (size_t i = 0; i < std::min(k, n); ++i)
    std::fprintf(stderr, "[%zu]=%.4f ", idx[i], v[idx[i]]);
  std::fprintf(stderr, "\n");
}

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
    // F32 is included so that F32 weight tensors stay mmap'd (w.data = mmap ptr,
    // w.quantized=true) rather than being heap-copied at load time. For a 213 GB
    // model like GLM-5.2 this avoids OOM from duplicate anonymous pages.
    case QuantType::F32:
    case QuantType::F16:
    case QuantType::BF16:
    case QuantType::Q4_0:
    case QuantType::Q5_0:
    case QuantType::Q5_1:
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
    gemv_quantized(y, w.quant, w.qbytes(), rows, cols, x);
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
      CudaBackend::instance().gemv_quantized(y, w.quant, w.qbytes(), rows, cols, x);
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
  // GLM-5.2 (glm-dsa) ships 1 nextn/MTP head; we simply skip it (layer_count
  // already excludes it via from_gguf). Other architectures with MTP heads stay
  // out of Phase 1 scope.
  if (cfg.nextn_predict_layers > 0 && cfg.architecture != Architecture::GlmDsa) {
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
                                    bool keep_quantized, bool allow_quant,
                                    bool force_keep_quantized) {
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

  // On-the-fly F16/BF16/F32 -> Q8_0 quantization (near-lossless, halves weight
  // bytes -> ~1.3x decode). Only for high-precision sources with 32-aligned rows.
  bool can_q8 = allow_quant && quantize_to_ == QuantType::Q8_0 &&
                (tv.quant == QuantType::F16 || tv.quant == QuantType::BF16 ||
                 tv.quant == QuantType::F32) &&
                count % 32 == 0 && w.cols % 32 == 0;
  if (can_q8) {
    std::vector<float> tmp(count, 0.0f);
    dequantize_row(tv.quant, tv.data, tmp.data(), count);
    w.owned.resize(quantized_size(QuantType::Q8_0, count));
    quantize_row_q8_0(tmp.data(), w.owned.data(), count);
    w.quantized = true;
    w.quant = QuantType::Q8_0;
    w.data = nullptr;
  } else if (keep_quantized &&
             (is_supported_quant_gemv(tv.quant) || force_keep_quantized)) {
    // Borrow the mmap'd quantized bytes in their native block layout. With
    // force_keep_quantized we keep types not yet wired into gemv (e.g. the IQ
    // MoE experts and Q5_K MLA projections of GLM-5.2) quantized rather than
    // eagerly dequantizing — required to mmap the 213GB model without expanding
    // every tensor to f32 in RAM. The MLA/MoE forward path is responsible for
    // decoding these on demand.
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

LlamaModel::LlamaModel(GgufModel gguf, QuantType quantize_to)
    : gguf_(std::move(gguf)), quantize_to_(quantize_to) {
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
  // Embeddings are gathered per-token via dequantize_row (not gemv), so keep
  // them in their original layout — never on-the-fly quantized.
  tok_embeddings_ = load_weight(g, embd_name, /*keep_quantized=*/true,
                                /*allow_quant=*/false);

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
    // MoE expert tensors are 3D and addressed per-expert via the borrowed mmap
    // pointer, so they must not be on-the-fly quantized into an owned buffer.
    auto opt_w_raw = [&](const std::string& suffix, LlamaWeight& dst) {
      if (g.has_tensor(p + suffix))
        dst = load_weight(g, p + suffix, /*keep_quantized=*/true, /*allow_quant=*/false);
    };
    // Keep the tensor in its native mmap'd quantized layout regardless of
    // whether gemv supports the type yet (used for MLA + MoE/shared-expert
    // tensors so the 213GB GLM-5.2 maps without dequantizing into RAM).
    auto opt_w_keepq = [&](const std::string& suffix, LlamaWeight& dst) {
      if (g.has_tensor(p + suffix))
        dst = load_weight(g, p + suffix, /*keep_quantized=*/true,
                          /*allow_quant=*/false, /*force_keep_quantized=*/true);
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

    // MoE routed experts (Mixtral / Qwen-MoE / GLM-5.2). Present only on MoE
    // layers. Use keepq so IQ-quantized GLM experts stay mmap'd (not dequantized).
    opt_w_keepq("ffn_gate_exps.weight", layer.ffn_gate_exps);
    opt_w_keepq("ffn_up_exps.weight", layer.ffn_up_exps);
    opt_w_keepq("ffn_down_exps.weight", layer.ffn_down_exps);
    opt_w_raw("ffn_gate_inp.weight", layer.ffn_gate_inp);
    opt_vec("exp_probs_b.bias", layer.ffn_exp_probs_b);

    // Shared (always-on) expert FFN (GLM-5.2 / DeepSeek-V2).
    opt_w_keepq("ffn_gate_shexp.weight", layer.ffn_gate_shexp);
    opt_w_keepq("ffn_up_shexp.weight", layer.ffn_up_shexp);
    opt_w_keepq("ffn_down_shexp.weight", layer.ffn_down_shexp);

    // MLA compressed-attention projections (GLM-5.2 glm-dsa). Replace dense
    // attn_q/attn_k/attn_v. attn_q_b / attn_k_b / attn_v_b can be Q8_0 (gemv-ok)
    // or other quant; keepq keeps them mmap'd regardless. 3D k_b/v_b must NOT be
    // on-the-fly quantized, hence keepq (allow_quant=false).
    opt_w_keepq("attn_q_a.weight", layer.attn_q_a);
    opt_vec("attn_q_a_norm.weight", layer.attn_q_a_norm);
    opt_w_keepq("attn_q_b.weight", layer.attn_q_b);
    opt_w_keepq("attn_kv_a_mqa.weight", layer.attn_kv_a_mqa);
    opt_vec("attn_kv_a_norm.weight", layer.attn_kv_a_norm);
    opt_w_keepq("attn_k_b.weight", layer.attn_k_b);
    opt_w_keepq("attn_v_b.weight", layer.attn_v_b);

    // A layer must have either dense attn_q OR the MLA q/kv down-projections.
    if (layer.attn_q.empty() && !layer.is_mla()) {
      throw std::runtime_error("LlamaModel: layer " + std::to_string(l) +
                               " has no attn_q (non-dense-attention layer unsupported in Phase 1)");
    }
    if (layer.is_moe()) any_moe_ = true;
  }

  // ---- KV cache (F32, layer-major). Only attention layers are stored; in the
  // dense path every layer is an attention layer, so attn_layer_count ==
  // layer_count and the layer index maps directly. ----
  kv_token_size_ = config_.num_key_value_heads * config_.head_dim();
  // Up-front F32 KV cache. Cap the allocated context only when the full
  // advertised context would need an impractically large cache (GLM-5.2 reports
  // a 1,048,576-token context => a 377 GB F32 cache). Below the budget the cache
  // is allocated for the full context, so qwen/llama/etc. behavior is unchanged
  // (kv_context_ == context_size). kv_context_ is the KV slot layer-stride and
  // forward() rejects positions >= kv_context_.
  constexpr size_t kKvBudgetBytes = static_cast<size_t>(8) << 30;  // 8 GB
  constexpr size_t kKvMinPositions = 4096;
  kv_context_ = config_.context_size;
  size_t bytes_per_pos = config_.layer_count * kv_token_size_ * sizeof(float) * 2;
  if (bytes_per_pos > 0 &&
      config_.context_size * bytes_per_pos > kKvBudgetBytes) {
    kv_context_ = std::max(kKvMinPositions, kKvBudgetBytes / bytes_per_pos);
    kv_context_ = std::min(kv_context_, config_.context_size);
  }
  if (kv_context_ == 0) kv_context_ = config_.context_size;
  size_t kv_elems = config_.layer_count * kv_context_ * kv_token_size_;
  if (kv_context_ < config_.context_size) {
    std::fprintf(stderr,
                 "warning: KV cache capped to %zu positions (model advertises "
                 "%zu); a full F32 cache would need %.1f GB\n",
                 kv_context_, config_.context_size,
                 (double)config_.layer_count * config_.context_size *
                     kv_token_size_ * 2 * 4 / 1e9);
  }
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
  dbg_vec("moe router logits (raw)", rw.data(), n_experts, 8);

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

  // 4. Renormalize routing weights over selected; apply routed scale. The
  // routing *weights* are gathered from the UNBIASED probs `rw` (the per-expert
  // bias only affects top-k selection above). For softmax gating rw already sums
  // to 1 over all experts; GLM/DeepSeek sigmoid gating renormalizes the selected
  // gates (expert_weights_norm) and scales by expert_weights_scale (2.5).
  // Mixtral/Qwen (softmax, norm=false) keep the per-expert softmax probs as-is.
  float wnorm = 1.0f;
  if (cfg.expert_weights_norm || !sigmoid) {
    wnorm = 0.0f;
    for (size_t s = 0; s < n_sel; ++s) wnorm += rw[sel[s].first];
    // llama.cpp clamps the divisor to avoid blow-up on tiny sigmoid gates.
    wnorm = std::max(wnorm, 6.103515625e-5f);
  }
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

  dbg_vec("moe routed ffn_out (pre-shexp)", ffn_out, h);
  // 6. Shared (always-on) expert (GLM-5.2 / DeepSeek-V2): a standard SwiGLU FFN
  // over the SAME normed input, added with weight 1.0 (NOT routed, NOT scaled by
  // expert_weights_scale). n_ff_shexp = expert_inter * num_shared_experts.
  if (layer.has_shared_expert()) {
    size_t n_shared = std::max<size_t>(cfg.num_shared_experts, 1);
    size_t sh_inter = i_size * n_shared;
    std::vector<float> sgate(sh_inter), sup(sh_inter), sdown(h);
    gemv_weight(layer.ffn_gate_shexp, sh_inter, h, normed, sgate.data());
    gemv_weight(layer.ffn_up_shexp, sh_inter, h, normed, sup.data());
    swiglu_inplace(sgate.data(), sup.data(), sgate.data(), sh_inter);
    gemv_weight(layer.ffn_down_shexp, h, sh_inter, sgate.data(), sdown.data());
    for (size_t i = 0; i < h; ++i) ffn_out[i] += sdown[i];
  }
}

void LlamaModel::mla_attention(const LlamaLayer& layer, size_t l, size_t pos,
                               float* attn_out) {
  const InferenceConfig& cfg = config_;
  const size_t h = cfg.hidden_size;
  const size_t n_heads = cfg.num_attention_heads;       // 64
  const size_t q_rank = cfg.q_lora_rank;                // 2048
  const size_t kv_rank = cfg.kv_lora_rank;              // 512 (latent)
  const size_t mla_key = cfg.mla_key_dim;               // 256 per-head q/k width
  const size_t mla_val = cfg.mla_val_dim;               // 256 per-head v width
  const size_t n_rot = cfg.rope_dim;                    // 64 (partial rope)
  const size_t n_nope = mla_key - n_rot;                // 192
  const size_t latent_row = kv_rank + n_rot;            // 576 (cache row)
  const float theta = cfg.rope_theta;                   // 8e6
  const float eps = cfg.rms_norm_eps;
  const float kq_scale = 1.0f / std::sqrt(static_cast<float>(mla_key));  // 1/16
  const size_t seq_len = pos + 1;

  // Per-head gemv on a 3D [n_heads, rows, cols] absorb tensor (k_b / v_b). Head h
  // occupies a contiguous rows*cols quantized block.
  auto gemv_head = [&](const LlamaWeight& w, size_t head, size_t rows,
                       size_t cols, const float* x, float* y) {
    if (w.quantized) {
      size_t rb = quantized_size(w.quant, cols);  // bytes per row
      const uint8_t* base = w.qbytes() + head * rows * rb;
      gemv_quantized(y, w.quant, base, rows, cols, x);
    } else {
      matvec(y, w.f32.data() + head * rows * cols, x, rows, cols);
    }
  };

  bool is_l0p0 = (l == 0 && pos == 0);

  // 0. pre-attention RMSNorm.
  std::vector<float> normed(h);
  d_rms_norm(normed.data(), x_.data(), layer.attn_norm.data(), h, eps, /*plus_one=*/false);
  if (is_l0p0) {
    dbg_vec("x (embed)", x_.data(), h);
    dbg_vec("normed (l0 attn_norm)", normed.data(), h);
  }

  // 1. Query: x -> q_a(2048) -> norm -> q(n_heads*mla_key).
  std::vector<float> q_a(q_rank);
  gemv_weight(layer.attn_q_a, q_rank, h, normed.data(), q_a.data());
  if (is_l0p0) dbg_vec("q_a (pre-norm)", q_a.data(), q_rank);
  {
    std::vector<float> tmp(q_rank);
    rms_norm(tmp.data(), q_a.data(), layer.attn_q_a_norm.data(), q_rank, eps, false);
    q_a.swap(tmp);
  }
  if (is_l0p0) dbg_vec("q_a (post-norm)", q_a.data(), q_rank);
  std::vector<float> q(n_heads * mla_key);
  gemv_weight(layer.attn_q_b, n_heads * mla_key, q_rank, q_a.data(), q.data());
  if (is_l0p0) dbg_vec("q[h0] (pre-rope)", q.data(), mla_key);

  // 2. Partial RoPE on the rope part (last n_rot dims) of each q head.
  // GLM-DSA uses LLAMA_ROPE_TYPE_NORM (adjacent-pair rotation), NOT NeoX
  // split-half. apply_rope_norm rotates pairs (h[2i], h[2i+1]).
  for (size_t hd = 0; hd < n_heads; ++hd) {
    float* q_pe = q.data() + hd * mla_key + n_nope;
    apply_rope_norm(q_pe, n_rot, 1, pos, theta, /*rope_dim=*/0);
  }

  // 3. KV: x -> kv_a_mqa(576). Split latent(512) + k_pe(64); norm latent; rope k_pe.
  std::vector<float> kv(latent_row);
  gemv_weight(layer.attn_kv_a_mqa, latent_row, h, normed.data(), kv.data());
  if (is_l0p0) {
    dbg_vec("kv (pre-norm latent part)", kv.data(), kv_rank);
    dbg_vec("kv (k_pe part)", kv.data() + kv_rank, n_rot);
  }
  std::vector<float> cache_row(latent_row);
  rms_norm(cache_row.data(), kv.data(), layer.attn_kv_a_norm.data(), kv_rank, eps, false);
  // k_pe (the n_rot rope dims) is NOT normed; rope then store.
  for (size_t i = 0; i < n_rot; ++i) cache_row[kv_rank + i] = kv[kv_rank + i];
  apply_rope_norm(cache_row.data() + kv_rank, n_rot, 1, pos, theta, /*rope_dim=*/0);
  if (is_l0p0) {
    dbg_vec("cache_row (normed latent)", cache_row.data(), kv_rank);
    dbg_vec("cache_row (roped k_pe)", cache_row.data() + kv_rank, n_rot);
  }

  // 4. Store the 576-dim compressed latent row in the KV cache (kv_heads=1).
  size_t phys = pos % kv_context_;
  size_t base = (l * kv_context_ + phys) * kv_token_size_;
  for (size_t i = 0; i < latent_row; ++i) kv_keys_[base + i] = cache_row[i];

  // 5. Per-head MLA attention over the compressed latent cache (MQA: one shared
  // K/V row per position). Q_head = [absorb(q_nope)->512 ++ q_pe(64)] (576).
  const float* cache_layer = kv_keys_.data() + (l * kv_context_) * kv_token_size_;
  std::vector<float> attn_result(n_heads * mla_val, 0.0f);
  std::vector<float> q_absorbed(kv_rank);     // 512
  std::vector<float> Q(latent_row);           // 576
  std::vector<float> scores(seq_len);
  std::vector<float> ctx_latent(kv_rank);     // 512
  for (size_t hd = 0; hd < n_heads; ++hd) {
    const float* q_head = q.data() + hd * mla_key;
    // absorb wk_b[h]: q_nope(192) -> q_absorbed(512).
    gemv_head(layer.attn_k_b, hd, kv_rank, n_nope, q_head, q_absorbed.data());
    for (size_t i = 0; i < kv_rank; ++i) Q[i] = q_absorbed[i];
    for (size_t i = 0; i < n_rot; ++i) Q[kv_rank + i] = q_head[n_nope + i];

    // scores over cached positions (causal: all [0, seq_len) are valid).
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < seq_len; ++t) {
      const float* row = cache_layer + t * kv_token_size_;
      float dot = 0.0f;
      for (size_t i = 0; i < latent_row; ++i) dot += Q[i] * row[i];
      dot *= kq_scale;
      scores[t] = dot;
      mx = std::max(mx, dot);
    }
    float sum = 0.0f;
    for (size_t t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - mx);
      sum += scores[t];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    // ctx_latent = sum_t a[t] * row[t][0:512]  (V aliases the latent part).
    for (size_t i = 0; i < kv_rank; ++i) ctx_latent[i] = 0.0f;
    for (size_t t = 0; t < seq_len; ++t) {
      float a = scores[t] * inv;
      const float* row = cache_layer + t * kv_token_size_;
      for (size_t i = 0; i < kv_rank; ++i) ctx_latent[i] += a * row[i];
    }
    // decompress via wv_b[h]: ctx_latent(512) -> out(256).
    gemv_head(layer.attn_v_b, hd, mla_val, kv_rank, ctx_latent.data(),
              attn_result.data() + hd * mla_val);
    if (is_l0p0 && hd == 0) {
      dbg_vec("q_absorbed (h0)", q_absorbed.data(), kv_rank);
      dbg_vec("Q[h0] (576)", Q.data(), latent_row);
      char lbl[64]; std::snprintf(lbl, sizeof(lbl), "scores (h0, seq=%zu)", seq_len);
      dbg_vec(lbl, scores.data(), seq_len, seq_len);
      dbg_vec("ctx_latent (h0)", ctx_latent.data(), kv_rank);
      dbg_vec("attn_result (h0)", attn_result.data(), mla_val);
    }
  }
  if (is_l0p0) dbg_vec("attn_result (all heads, first 8)", attn_result.data(), n_heads * mla_val);

  // 6. Output projection: [n_heads*mla_val] -> [hidden].
  gemv_weight(layer.attn_output, h, n_heads * mla_val, attn_result.data(), attn_out);
  if (is_l0p0) dbg_vec("attn_out (post o_proj)", attn_out, h);
}

void LlamaModel::run_layers(size_t pos) {
  if (pos == 0) g_debug = (std::getenv("OXIDIZE_DEBUG") != nullptr);
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

    // MLA (GLM-5.2 glm-dsa / DeepSeek-V2) compressed-attention forward. Computes
    // the attention output into attn_out (RMSNorm + projections done inside),
    // then residual-add and fall through to the shared FFN block below.
    if (layer.is_mla()) {
      mla_attention(layer, l, pos, attn_out.data());
      for (size_t i = 0; i < h; ++i) x_[i] += attn_out[i];
      // ---- FFN block (shared with dense path) ----
      const std::vector<float>& ffn_norm_w = layer.ffn_norm;
      if (ffn_norm_w.empty()) {
        throw std::runtime_error("LlamaModel: MLA layer " + std::to_string(l) +
                                 " missing ffn_norm");
      }
      d_rms_norm(normed.data(), x_.data(), ffn_norm_w.data(), h, eps, plus_one);
      if (pos == 0 && l <= 1) {
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "x after attn+res (l%zu)", l);
        dbg_vec(lbl, x_.data(), h);
        std::snprintf(lbl, sizeof(lbl), "normed ffn (l%zu)", l);
        dbg_vec(lbl, normed.data(), h);
      }
      std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
      if (layer.is_moe()) {
        moe_ffn(layer, normed.data(), ffn_out.data());
      } else {
        // Leading dense layers (0..leading_dense-1): plain SwiGLU FFN.
        d_gemv_weight(layer.ffn_gate, inter, h, normed.data(), gate.data());
        d_gemv_weight(layer.ffn_up, inter, h, normed.data(), up.data());
        d_swiglu(gate.data(), up.data(), gate.data(), inter);
        d_gemv_weight(layer.ffn_down, h, inter, gate.data(), ffn_out.data());
      }
      if (pos == 0 && l <= 1) {
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "ffn_out (l%zu)", l);
        dbg_vec(lbl, ffn_out.data(), h);
      }
      for (size_t i = 0; i < h; ++i) x_[i] += ffn_out[i];
      continue;
    }

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
    // token_slot_index = l*kv_context_ + (pos % kv_context_); F32 storage.
    // (kv_context_ == context_size for normal models.)
    size_t phys = pos % kv_context_;
    size_t base = (l * kv_context_ + phys) * kv_token_size_;
    for (size_t i = 0; i < kv_len; ++i) {
      kv_keys_[base + i] = k[i];
      kv_values_[base + i] = v[i];
    }

    // GQA causal attention over the contiguous [0, seq_len) prefix.
    // Layer prefix start is at position 0 within this layer's slice.
    size_t layer_start = (l * kv_context_) * kv_token_size_;
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
  dbg_vec("final norm x", x_.data(), h);
  dbg_vec("final normed", normed.data(), h);

  Logits logits(vocab, 0.0f);
  const LlamaWeight& head = output_tied_ ? tok_embeddings_ : output_weight_;
  d_gemv_weight(head, vocab, h, normed.data(), logits.data());
  dbg_topk("final logits", logits.data(), vocab, 10);
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
  // Bound by the *allocated* KV context (== context_size for normal models; a
  // smaller cap for huge-context models like GLM-5.2, see ctor).
  if (requested_total > kv_context_) {
    throw std::runtime_error("LlamaModel::forward: context exceeded (requested " +
                             std::to_string(requested_total) + " > kv context " +
                             std::to_string(kv_context_) + ")");
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

std::unique_ptr<Model> load_llama_gguf(const std::string& path, bool want_cuda,
                                       QuantType quantize_to) {
  GgufModel gguf = GgufModel::load(path);
  auto model = std::make_unique<LlamaModel>(std::move(gguf), quantize_to);
  model->set_cuda(want_cuda);
  return model;
}

}  // namespace oxidize
