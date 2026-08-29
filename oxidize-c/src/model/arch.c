#include "oxidize/model.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/arena.h"
#include "oxidize/log.h"


OcModelArchitecture oc_model_arch_from_str(const char *s)
{
    if (!s || !*s) return OC_ARCH_UNKNOWN;

    /* Build a normalized copy (lowercase + '-' → '_') in a stack buffer.
     * Architecture strings are short (< 32 chars), so a 64-byte buffer is
     * ample. */
    char norm[64];
    size_t n = 0;
    for (; n < sizeof(norm) - 1 && s[n]; n++) {
        char c = s[n];
        if (c == '-') c = '_';
        else c = (char)tolower((unsigned char)c);
        norm[n] = c;
    }
    norm[n] = '\0';

    /* If the input was longer than our buffer, it can't match any known arch. */
    if (n == sizeof(norm) - 1 && s[n]) return OC_ARCH_UNKNOWN;

    /* Match (Rust from_gguf, in declaration order). */
    if (strcmp(norm, "llama") == 0)                          return OC_ARCH_LLAMA;
    if (strcmp(norm, "mistral") == 0)                         return OC_ARCH_MISTRAL;
    if (strcmp(norm, "mixtral") == 0)                         return OC_ARCH_MIXTRAL;
    if (strcmp(norm, "deepseek") == 0
        || strcmp(norm, "deepseek2") == 0
        || strcmp(norm, "deepseek_v2") == 0
        || strcmp(norm, "deepseek_v3") == 0
        || strcmp(norm, "deepseek_moe") == 0)                return OC_ARCH_DEEPSEEK;
    if (strcmp(norm, "qwen") == 0
        || strcmp(norm, "qwen2") == 0
        || strcmp(norm, "qwen2moe") == 0
        || strcmp(norm, "qwen3") == 0
        || strcmp(norm, "qwen3moe") == 0
        || strcmp(norm, "qwen36") == 0
        || strcmp(norm, "qwen3_6") == 0
        || strcmp(norm, "qwen35") == 0
        || strcmp(norm, "qwen3_5") == 0
        || strcmp(norm, "qwen3_5_text") == 0
        || strcmp(norm, "qwen35_text") == 0
        || strcmp(norm, "qwen3_5_moe") == 0
        || strcmp(norm, "qwen3_5_moe_text") == 0
        || strcmp(norm, "qwen35moe") == 0)                   return OC_ARCH_QWEN;
    if (strcmp(norm, "gemma") == 0
        || strcmp(norm, "gemma2") == 0
        || strcmp(norm, "gemma3") == 0
        || strcmp(norm, "gemma4") == 0)                      return OC_ARCH_GEMMA;
    if (strcmp(norm, "phi") == 0
        || strcmp(norm, "phi3") == 0)                        return OC_ARCH_PHI;
    if (strcmp(norm, "falcon") == 0)                         return OC_ARCH_FALCON;
    if (strcmp(norm, "gpt2") == 0)                           return OC_ARCH_GPT2;
    if (strcmp(norm, "gptj") == 0)                           return OC_ARCH_GPTJ;
    if (strcmp(norm, "gptneox") == 0)                       return OC_ARCH_GPTNEOX;
    if (strcmp(norm, "minimax") == 0
        || strcmp(norm, "minimax_m2") == 0
        || strcmp(norm, "minimax_text_01") == 0)            return OC_ARCH_MINIMAX;
    if (strcmp(norm, "lfm2") == 0)                            return OC_ARCH_LFM2;
    if (strcmp(norm, "lfm2moe") == 0)                        return OC_ARCH_LFM2_MOE;
    if (strcmp(norm, "glm") == 0
        || strcmp(norm, "glm4") == 0
        || strcmp(norm, "glm_moe") == 0
        || strcmp(norm, "glm_moe_dsa") == 0
        || strcmp(norm, "glm_dsa") == 0
        || strcmp(norm, "glmmoe") == 0
        || strcmp(norm, "glmmoedsa") == 0)                  return OC_ARCH_GLM_MOE_DSA;
    if (strcmp(norm, "hunyuan") == 0
        || strcmp(norm, "hunyuan_moe") == 0
        || strcmp(norm, "hunyuanmoe") == 0
        || strcmp(norm, "hy_v3") == 0
        || strcmp(norm, "hyv3") == 0
        || strcmp(norm, "hunyuan_v3") == 0)                 return OC_ARCH_HUNYUAN_MOE;
    if (strcmp(norm, "longcat") == 0
        || strcmp(norm, "longcat2") == 0
        || strcmp(norm, "longcat_2") == 0
        || strcmp(norm, "longcat_flash") == 0
        || strcmp(norm, "longcatflash") == 0)               return OC_ARCH_LONGCAT;
    if (strcmp(norm, "muse_glimmer") == 0
        || strcmp(norm, "museglimmer") == 0
        || strcmp(norm, "muse") == 0)                       return OC_ARCH_MUSE_GLIMMER;

    return OC_ARCH_UNKNOWN;
}

const char *oc_model_arch_name(OcModelArchitecture arch)
{
    switch (arch) {
    case OC_ARCH_LLAMA:        return "llama";
    case OC_ARCH_MISTRAL:      return "mistral";
    case OC_ARCH_MIXTRAL:      return "mixtral";
    case OC_ARCH_DEEPSEEK:     return "deepseek";
    case OC_ARCH_QWEN:         return "qwen";
    case OC_ARCH_GEMMA:        return "gemma";
    case OC_ARCH_PHI:          return "phi";
    case OC_ARCH_FALCON:       return "falcon";
    case OC_ARCH_GPT2:         return "gpt2";
    case OC_ARCH_GPTJ:         return "gptj";
    case OC_ARCH_GPTNEOX:      return "gptneox";
    case OC_ARCH_MINIMAX:      return "minimax";
    case OC_ARCH_LFM2:         return "lfm2";
    case OC_ARCH_LFM2_MOE:     return "lfm2moe";
    case OC_ARCH_GLM_MOE_DSA:  return "glm_moe_dsa";
    case OC_ARCH_HUNYUAN_MOE:  return "hunyuan_moe";
    case OC_ARCH_LONGCAT:      return "longcat";
    case OC_ARCH_MUSE_GLIMMER: return "muse-glimmer";
    default:                   return "unknown";   /* covers OC_ARCH_UNKNOWN + out-of-range */
    }
}

bool oc_model_arch_uses_moe(OcModelArchitecture arch)
{
    switch (arch) {
    case OC_ARCH_MIXTRAL:
    case OC_ARCH_MINIMAX:
    case OC_ARCH_LFM2_MOE:
    case OC_ARCH_DEEPSEEK:
    case OC_ARCH_GLM_MOE_DSA:
    case OC_ARCH_HUNYUAN_MOE:
    case OC_ARCH_LONGCAT:
        return true;
    default:
        return false;
    }
}

bool oc_model_arch_uses_mla(OcModelArchitecture arch)
{
    return arch == OC_ARCH_DEEPSEEK || arch == OC_ARCH_GLM_MOE_DSA
        || arch == OC_ARCH_LONGCAT;
}

bool oc_model_arch_uses_alibi(OcModelArchitecture arch)
{
    switch (arch) {
    case OC_ARCH_FALCON:
    case OC_ARCH_GPT2:
    case OC_ARCH_GPTJ:
    case OC_ARCH_GPTNEOX:
        return true;
    default:
        return false;
    }
}

bool oc_model_arch_uses_sliding_window(OcModelArchitecture arch)
{
    return arch == OC_ARCH_QWEN || arch == OC_ARCH_MISTRAL;
}

bool oc_model_arch_uses_shortconv(OcModelArchitecture arch)
{
    return arch == OC_ARCH_LFM2 || arch == OC_ARCH_LFM2_MOE;
}

bool oc_model_arch_uses_parallel_attn_ffn(OcModelArchitecture arch)
{
    return arch == OC_ARCH_GEMMA || arch == OC_ARCH_PHI;
}


/* Forward decl. */
static const char *map_hf_decoder_name(const char *name, OcArena *arena);
static const char *map_falcon_name(const char *name, OcArena *arena);
static const char *map_gpt2_name(const char *name, OcArena *arena);
static const char *map_gptj_name(const char *name, OcArena *arena);
static const char *map_gpt_neox_name(const char *name, OcArena *arena);

const char *oc_gguf_map_tensor_name(OcModelArchitecture arch, const char *name,
                                     OcArena *arena)
{
    if (!name || !arena) return NULL;

    const char *mapped = NULL;
    switch (arch) {
    case OC_ARCH_LLAMA:
    case OC_ARCH_MISTRAL:
    case OC_ARCH_MIXTRAL:
    case OC_ARCH_QWEN:
    case OC_ARCH_DEEPSEEK:
    case OC_ARCH_GEMMA:
    case OC_ARCH_PHI:
    case OC_ARCH_GLM_MOE_DSA:
    case OC_ARCH_LFM2:
    case OC_ARCH_LFM2_MOE:
    case OC_ARCH_MINIMAX:
    case OC_ARCH_HUNYUAN_MOE:
    case OC_ARCH_LONGCAT:
        mapped = map_hf_decoder_name(name, arena);
        break;
    case OC_ARCH_FALCON:
        mapped = map_falcon_name(name, arena);
        break;
    case OC_ARCH_GPT2:
        mapped = map_gpt2_name(name, arena);
        break;
    case OC_ARCH_GPTJ:
        mapped = map_gptj_name(name, arena);
        break;
    case OC_ARCH_GPTNEOX:
        mapped = map_gpt_neox_name(name, arena);
        break;
    case OC_ARCH_UNKNOWN:
    default:
        mapped = NULL;
        break;
    }

    if (mapped) return mapped;

    /* Fall back to an arena-owned copy of the original name. */
    return oc_arena_dup(arena, name);
}


/* Helper: parse the leading integer from `s` (digits only). Returns true and
 * writes `*out` on success; false if no digits or overflow. The integer is
 * the layer index. */
static bool parse_layer_index(const char *s, uint64_t *out)
{
    if (!s || !*s) return false;
    uint64_t v = 0;
    size_t i = 0;
    while (s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (uint64_t)(s[i] - '0');
        if (v > 0xfffffffffffffffull) return false;   /* overflow guard */
        i++;
    }
    if (i == 0) return false;   /* no digits consumed */
    *out = v;
    return true;
}

/* Match `suffix` against one of the literal suffix patterns; on match return
 * the corresponding mapped suffix string. Returns NULL if no match. */
static const char *match_hf_layer_suffix(const char *suffix)
{
    /* Order matters: longer/more-specific patterns first. */
    if (strcmp(suffix, "input_layernorm.weight") == 0)               return "attn_norm.weight";
    if (strcmp(suffix, "post_attention_layernorm.weight") == 0)     return "ffn_norm.weight";
    if (strcmp(suffix, "self_attn.q_proj.weight") == 0)             return "attn_q.weight";
    if (strcmp(suffix, "self_attn.k_proj.weight") == 0)             return "attn_k.weight";
    if (strcmp(suffix, "self_attn.v_proj.weight") == 0)             return "attn_v.weight";
    if (strcmp(suffix, "self_attn.o_proj.weight") == 0)             return "attn_output.weight";
    /* DeepSeek MLA tensors (VAL-FOUND-011). */
    if (strcmp(suffix, "self_attn.q_a_proj.weight") == 0)           return "attn_q_a.weight";
    if (strcmp(suffix, "self_attn.q_a_layernorm.weight") == 0)      return "attn_q_a_norm.weight";
    if (strcmp(suffix, "self_attn.q_b_proj.weight") == 0)           return "attn_q_b.weight";
    if (strcmp(suffix, "self_attn.kv_a_proj_with_mqa.weight") == 0) return "attn_kv_a_mqa.weight";
    if (strcmp(suffix, "self_attn.kv_a_layernorm.weight") == 0)     return "attn_kv_a_norm.weight";
    if (strcmp(suffix, "self_attn.k_b_proj.weight") == 0)           return "attn_k_b.weight";
    if (strcmp(suffix, "self_attn.v_b_proj.weight") == 0)           return "attn_v_b.weight";
    /* FFN — dense. */
    if (strcmp(suffix, "mlp.up_proj.weight") == 0)                  return "ffn_up.weight";
    if (strcmp(suffix, "mlp.gate_proj.weight") == 0)                return "ffn_gate.weight";
    if (strcmp(suffix, "mlp.down_proj.weight") == 0)               return "ffn_down.weight";
    if (strcmp(suffix, "mlp.gate.weight") == 0)                    return "ffn_gate_inp.weight";
    /* Qwen3-MoE shared expert (VAL-FOUND-010). */
    if (strcmp(suffix, "mlp.shared_expert.gate_proj.weight") == 0) return "ffn_gate_shexp.weight";
    if (strcmp(suffix, "mlp.shared_expert.up_proj.weight") == 0)   return "ffn_up_shexp.weight";
    if (strcmp(suffix, "mlp.shared_expert.down_proj.weight") == 0) return "ffn_down_shexp.weight";
    if (strcmp(suffix, "mlp.shared_expert_gate.weight") == 0)      return "ffn_gate_inp_shexp.weight";
    /* MoE router gate (block_sparse_moe.gate.weight). */
    if (strcmp(suffix, "block_sparse_moe.gate.weight") == 0)       return "ffn_gate_inp.weight";
    return NULL;
}

static const char *map_hf_decoder_name(const char *name, OcArena *arena)
{
    if (!name || !arena) return NULL;

    /* Top-level (non-layer) mappings. */
    if (strcmp(name, "model.embed_tokens.weight") == 0)
        return oc_arena_dup(arena, "tok_embeddings.weight");
    if (strcmp(name, "lm_head.weight") == 0)
        return oc_arena_dup(arena, "output.weight");
    if (strcmp(name, "model.norm.weight") == 0)
        return oc_arena_dup(arena, "norm.weight");

    /* Layer-scoped: model.layers.<N>.<suffix> */
    static const char prefix[] = "model.layers.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (strncmp(name, prefix, prefix_len) != 0) return NULL;
    const char *p = name + prefix_len;

    /* Parse the layer index. */
    uint64_t layer_idx = 0;
    if (!parse_layer_index(p, &layer_idx)) return NULL;
    /* Advance past the digits. */
    while (*p >= '0' && *p <= '9') p++;
    /* The next char MUST be '.' (model.layers.N.<suffix>). */
    if (*p != '.') return NULL;
    p++;   /* skip the '.' */

    /* Now `p` points at the suffix. Check the expert-MoE patterns first
     * (they have their own nested structure). */

    /* Pattern A: block_sparse_moe.experts.<M>.<w1|w2|w3>.weight
     *   → blk.<N>.ffn_<gate|down|up>.<M>.weight
     * This is the Mixtral/Qwen2-MoE convention (VAL-FOUND-009). */
    static const char bsme_prefix[] = "block_sparse_moe.experts.";
    const size_t bsme_len = sizeof(bsme_prefix) - 1;
    if (strncmp(p, bsme_prefix, bsme_len) == 0) {
        const char *rest = p + bsme_len;
        /* Parse expert index. */
        uint64_t expert_idx = 0;
        if (!parse_layer_index(rest, &expert_idx)) return NULL;
        while (*rest >= '0' && *rest <= '9') rest++;
        if (*rest != '.') return NULL;
        rest++;   /* skip '.' */
        /* Match w1/w2/w3.weight. */
        const char *mapped_weight = NULL;
        if (strcmp(rest, "w1.weight") == 0)      mapped_weight = "ffn_gate";
        else if (strcmp(rest, "w2.weight") == 0) mapped_weight = "ffn_down";
        else if (strcmp(rest, "w3.weight") == 0) mapped_weight = "ffn_up";
        else return NULL;
        return oc_arena_printf(arena, "blk.%llu.%s.%llu.weight",
                                (unsigned long long)layer_idx, mapped_weight,
                                (unsigned long long)expert_idx);
    }

    /* Pattern B: mlp.experts.<M>.<gate_proj|up_proj|down_proj>.weight This is the DeepSeek convention (VAL-FOUND-011 Rust test */
    static const char me_prefix[] = "mlp.experts.";
    const size_t me_len = sizeof(me_prefix) - 1;
    if (strncmp(p, me_prefix, me_len) == 0) {
        const char *rest = p + me_len;
        uint64_t expert_idx = 0;
        if (!parse_layer_index(rest, &expert_idx)) return NULL;
        while (*rest >= '0' && *rest <= '9') rest++;
        if (*rest != '.') return NULL;
        rest++;
        const char *mapped_weight = NULL;
        if (strcmp(rest, "gate_proj.weight") == 0)      mapped_weight = "ffn_gate";
        else if (strcmp(rest, "up_proj.weight") == 0)   mapped_weight = "ffn_up";
        else if (strcmp(rest, "down_proj.weight") == 0) mapped_weight = "ffn_down";
        else return NULL;
        return oc_arena_printf(arena, "blk.%llu.%s.%llu.weight",
                                (unsigned long long)layer_idx, mapped_weight,
                                (unsigned long long)expert_idx);
    }

    /* Pattern C: literal suffix match (attn_*, ffn_*, norm, etc.). */
    const char *mapped_suffix = match_hf_layer_suffix(p);
    if (!mapped_suffix) return NULL;
    return oc_arena_printf(arena, "blk.%llu.%s",
                            (unsigned long long)layer_idx, mapped_suffix);
}


static const char *map_falcon_name(const char *name, OcArena *arena)
{
    if (!name || !arena) return NULL;
    if (strcmp(name, "transformer.word_embeddings.weight") == 0)
        return oc_arena_dup(arena, "tok_embeddings.weight");
    if (strcmp(name, "lm_head.weight") == 0)
        return oc_arena_dup(arena, "output.weight");
    if (strcmp(name, "transformer.ln_f.weight") == 0)
        return oc_arena_dup(arena, "norm.weight");
    return NULL;
}

static const char *map_gpt2_name(const char *name, OcArena *arena)
{
    if (!name || !arena) return NULL;
    if (strcmp(name, "transformer.wte.weight") == 0)
        return oc_arena_dup(arena, "tok_embeddings.weight");
    if (strcmp(name, "lm_head.weight") == 0)
        return oc_arena_dup(arena, "output.weight");
    if (strcmp(name, "transformer.ln_f.weight") == 0)
        return oc_arena_dup(arena, "norm.weight");
    return NULL;
}

/* GPTJ shares the same mapping as GPT2 (Rust uses the same table). */
static const char *map_gptj_name(const char *name, OcArena *arena)
{
    return map_gpt2_name(name, arena);
}

static const char *map_gpt_neox_name(const char *name, OcArena *arena)
{
    if (!name || !arena) return NULL;
    if (strcmp(name, "gpt_neox.embed_in.weight") == 0)
        return oc_arena_dup(arena, "tok_embeddings.weight");
    if (strcmp(name, "embed_out.weight") == 0
        || strcmp(name, "lm_head.weight") == 0)
        return oc_arena_dup(arena, "output.weight");
    if (strcmp(name, "gpt_neox.final_layer_norm.weight") == 0)
        return oc_arena_dup(arena, "norm.weight");
    return NULL;
}
