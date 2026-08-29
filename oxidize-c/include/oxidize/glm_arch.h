#ifndef OXIDIZE_GLM_ARCH_H
#define OXIDIZE_GLM_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"   /* OcLlamaSession, OcWeightView, OcLlamaModel */
#include "oxidize/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_GLM_VERSION_UNKNOWN = 0,   /* not a GLM model                         */
    OC_GLM_VERSION_1       = 1,   /* ChatGLM-6B / GLM-130B                   */
    OC_GLM_VERSION_2       = 2,   /* ChatGLM2-6B                              */
    OC_GLM_VERSION_3       = 3,   /* ChatGLM3-6B                              */
    OC_GLM_VERSION_4       = 4,   /* GLM-4 (incl. GLM-4-9B, GLM-4-32B)        */
} OcGlmVersion;

typedef struct OcGlmConfig {
    uint32_t    vocab_size;              /* vocabulary size                    */
    uint32_t    hidden_size;             /* n_embd                              */
    uint32_t    n_layer;                 /* number of transformer layers       */
    uint32_t    num_attention_heads;     /* n_head                              */
    uint32_t    num_kv_heads;            /* n_head_kv (GQA); == n_head if MQA   */
    uint32_t    intermediate_size;       /* n_ff (SwiGLU FFN intermediate)      */
    uint32_t    max_position_embeddings; /* n_ctx                               */
    uint32_t    head_dim;                /* per-head dim (n_embd / n_head)      */
    uint32_t    kv_head_dim;             /* per kv-head dim (usually == head_dim)*/
    uint32_t    rope_dim;                /* RoPE dims (0 → full kv_head_dim)     */
    float       rope_theta;              /* RoPE base frequency                 */
    float       rms_norm_eps;            /* RMSNorm epsilon                     */
    bool        uses_mla;                /* true = MLA (Multi-head Latent Attn) */
    bool        apply_qk_norm;           /* true = RMSNorm on Q/K after proj    */
    bool        uses_interleaved_rope;   /* true = GPT-J interleaved (ChatGLM-6B)*/
    bool        tied_embeddings;         /* true = output.weight == tok_emb     */
    OcGlmVersion glm_version;            /* detected GLM version                */
} OcGlmConfig;

typedef struct OcHunyuanConfig {
    uint32_t    vocab_size;              /* vocabulary size                    */
    uint32_t    hidden_size;             /* n_embd                              */
    uint32_t    n_layer;                 /* number of transformer layers       */
    uint32_t    num_attention_heads;     /* n_head                              */
    uint32_t    num_kv_heads;            /* n_head_kv (GQA)                     */
    uint32_t    intermediate_size;       /* dense FFN size (pre-MoE layers)     */
    uint32_t    max_position_embeddings; /* n_ctx                               */
    uint32_t    head_dim;                /* per-head dim                         */
    uint32_t    kv_head_dim;             /* per kv-head dim                      */
    uint32_t    rope_dim;                /* RoPE dims                            */
    float       rope_theta;              /* RoPE base frequency                  */
    float       rms_norm_eps;            /* RMSNorm epsilon                      */
    /* MoE fields. */
    uint32_t    n_routed_experts;        /* total number of experts              */
    uint32_t    n_active_experts;        /* top-k experts per token              */
    uint32_t    expert_intermediate_size;/* per-expert FFN intermediate size     */
    uint32_t    moe_layer_start;         /* first MoE layer index (0-based)      */
    bool        has_shared_expert;        /* true = shared expert present         */
    uint32_t    shared_expert_intermediate_size; /* shared expert FFN size     */
    /* MLA fields (Hunyuan-Large). */
    bool        uses_mla;                 /* true = MLA attention                */
    uint32_t    mla_q_lora_dim;          /* q_a_proj output dim (latent)        */
    uint32_t    mla_kv_lora_dim;         /* kv_a_proj latent dim (without kv_pe) */
    uint32_t    mla_q_head_dim;          /* per-head q dim (nope + rope)         */
    uint32_t    mla_q_rope_dim;          /* RoPE dim for q_pe                    */
    uint32_t    mla_q_nope_dim;          /* per-head nope dim (mla_q_head_dim - rope_dim) */
    uint32_t    mla_v_head_dim;          /* per-head v dim                       */
    bool        tied_embeddings;         /* true = output.weight == tok_emb      */
} OcHunyuanConfig;

OcError oc_arch_forward_glm(OcLlamaSession *sess, uint32_t token,
                             float *logits_out);

OcError oc_arch_forward_hunyuan(OcLlamaSession *sess, uint32_t token,
                                 float *logits_out);

OcError oc_glm_config_parse(const OcGgufFile *f, const char *arch_str,
                             OcGlmConfig *cfg);

OcError oc_hunyuan_config_parse(const OcGgufFile *f, const char *arch_str,
                                 OcHunyuanConfig *cfg);

/* Detect the GLM version from the architecture string. */
OcGlmVersion oc_glm_version_from_str(const char *s);

/* Initialize a GLM config to sensible defaults (all zeros + default eps,
 * rope_theta, version). Used by tests and as a starting point for parsing. */
void oc_glm_config_defaults(OcGlmConfig *cfg);

/* Initialize a Hunyuan config to sensible defaults. */
void oc_hunyuan_config_defaults(OcHunyuanConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GLM_ARCH_H */
