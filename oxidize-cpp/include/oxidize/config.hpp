#pragma once
// Core value types mirroring oxidize-core/src (DType, ModelArchitecture,
// InferenceConfig, GgufQuantizationType). Kept header-only and dependency-free
// so both the CPU build and the CUDA build share identical definitions.

#include <cstdint>
#include <cstddef>
#include <string>

namespace oxidize {

// Mirror of oxidize-core/src/compute/tensor/types.rs::DType
enum class DType : uint8_t { F32, F16, I8, I16, I32, I64 };

inline size_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::I8:  return 1;
    case DType::I16: return 2;
    case DType::I32: return 4;
    case DType::I64: return 8;
  }
  return 0;
}

// Mirror of oxidize-core/src/format/gguf.rs::GgufQuantizationType (discriminants
// align with llama.cpp ggml_type where applicable; see gguf.cpp::from_ggml_type).
enum class QuantType : uint16_t {
  F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
  Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K,
  IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M,
  I8, I16, I32, I64, F64, BF16, NVFP4,
  Q4_O,  /* oxidize custom: Q4_0 layout, MSE-optimal scale (ggml type 240) */
  Unknown,
};

// Mirror of oxidize-core/src/model/inference.rs::ModelArchitecture
enum class Architecture : uint8_t {
  Llama, Mistral, Mixtral, DeepSeek, Qwen, Gemma, Phi, Falcon,
  Gpt2, GptJ, GptNeoX, MiniMax, Lfm2, Lfm2Moe, GlmDsa,
};

Architecture architecture_from_name(const std::string& name);

// Mirror of oxidize-core/src/model/inference.rs::InferenceConfig
struct InferenceConfig {
  size_t vocab_size = 32000;
  size_t context_size = 4096;
  size_t layer_count = 32;
  size_t hidden_size = 4096;
  size_t intermediate_size = 11008;
  size_t num_attention_heads = 32;
  size_t num_key_value_heads = 32;
  size_t key_value_head_dim = 0;       // 0 => hidden_size / num_attention_heads
  DType kv_cache_dtype = DType::F32;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 10000.0f;
  Architecture architecture = Architecture::Llama;
  size_t sliding_window = 0;
  size_t num_experts = 0;
  size_t num_experts_per_tok = 0;
  size_t expert_intermediate_size = 0;
  size_t alibi_num_heads = 0;
  size_t shortconv_l_cache = 0;
  size_t leading_dense_layers = 0;
  bool   expert_gating_sigmoid = false;
  size_t rope_dim = 0;                 // 0 => full head_dim
  float  rope_theta_swa = 0.0f;
  size_t sliding_window_pattern = 0;
  float  embedding_scale = 1.0f;
  bool   gelu_ffn = false;
  bool   sandwich_norm = false;
  bool   rms_norm_weight_plus_one = false;
  size_t nextn_predict_layers = 0;
  float  expert_weights_scale = 1.0f;
  size_t expert_group_count = 0;
  size_t expert_group_used_count = 0;

  // MLA (GLM-5.2 glm-dsa / DeepSeek-V2 style compressed attention).
  size_t q_lora_rank = 0;        // q down-projection rank (glm-dsa.attention.q_lora_rank)
  size_t kv_lora_rank = 0;       // kv compressed latent dim (glm-dsa.attention.kv_lora_rank)
  size_t mla_key_dim = 0;        // per-head MLA key dim (attention.key_length_mla)
  size_t mla_val_dim = 0;        // per-head MLA value dim (attention.value_length_mla)
  size_t num_shared_experts = 0; // shared (always-on) experts (expert_shared_count)
  bool   expert_weights_norm = false;  // normalize routed top-k expert weights

  size_t head_dim() const {
    return key_value_head_dim != 0 ? key_value_head_dim
                                   : (num_attention_heads ? hidden_size / num_attention_heads : 0);
  }
};

}  // namespace oxidize
