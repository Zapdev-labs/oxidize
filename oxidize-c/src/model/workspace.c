#define _POSIX_C_SOURCE 200809L
#include "oxidize/workspace.h"

#include <stdlib.h>
#include <string.h>

static float *alloc_zeroed(size_t n)
{
    if (n == 0) n = 1;
    float *p = calloc(n, sizeof(float));
    return p;
}

OcError oc_workspace_for_config(OcWorkspace *ws, const OcInferenceConfig *cfg)
{
    if (!ws || !cfg) return OC_ERR_INVALID_ARG;

    memset(ws, 0, sizeof(*ws));

    size_t h = cfg->hidden_size;
    size_t inter = cfg->intermediate_size;
    size_t expert_inter = cfg->expert_intermediate_size > 0 ? cfg->expert_intermediate_size : inter;
    size_t max_inter = inter > expert_inter ? inter : expert_inter;

    size_t head_dim = cfg->hidden_size / (cfg->num_attention_heads > 0 ? cfg->num_attention_heads : 1);
    size_t kv_head_dim = cfg->key_value_head_dim > 0 ? cfg->key_value_head_dim : head_dim;
    size_t max_kv_len = cfg->num_key_value_heads * kv_head_dim;

    /* MLA compressed attention needs more scratch. */
    size_t mla_head_dim = kv_head_dim > head_dim ? kv_head_dim : head_dim;
    size_t mla_storage = 0;
    /* MLA check: DeepSeek or GLM-MoE-DSA */
    if (cfg->model_type == OC_INF_MODEL_GLM) {
        mla_storage = cfg->num_attention_heads * mla_head_dim;
    }

    size_t max_qkv = h * 3;
    if (max_inter > max_qkv) max_qkv = max_inter;
    if (mla_storage > max_qkv) max_qkv = mla_storage;

    size_t kv_scratch = max_kv_len > mla_storage ? max_kv_len : mla_storage;
    size_t kv_copy_size = (size_t)cfg->context_size * max_kv_len;
    size_t n_experts_per_tok = cfg->num_experts_per_tok > 0 ? cfg->num_experts_per_tok : 1;
    size_t head_dim_max = head_dim;
    if (kv_head_dim > head_dim_max) head_dim_max = kv_head_dim;
    if (192 > head_dim_max) head_dim_max = 192;

    ws->hidden_size = h;
    ws->intermediate_size = max_inter;
    ws->max_qkv = max_qkv;
    ws->kv_scratch = kv_scratch;
    ws->head_dim = head_dim_max;
    ws->kv_copy_size = kv_copy_size;
    ws->vocab_size = cfg->vocab_size;
    ws->n_experts = cfg->num_experts > 0 ? cfg->num_experts : 1;
    ws->n_experts_per_tok = n_experts_per_tok;
    ws->expert_inter = expert_inter;

    /* Allocate all buffers. */
    ws->x = alloc_zeroed(h);
    ws->hidden_a = alloc_zeroed(h);
    ws->hidden_b = alloc_zeroed(h);
    ws->intermediate_a = alloc_zeroed(max_inter);
    ws->intermediate_b = alloc_zeroed(max_inter);
    ws->intermediate_c = alloc_zeroed(max_inter);
    ws->q_full = alloc_zeroed(max_qkv);
    ws->k_vec = alloc_zeroed(kv_scratch);
    ws->v_vec = alloc_zeroed(kv_scratch);
    ws->attn_result = alloc_zeroed(max_qkv);
    ws->flash_q = alloc_zeroed(max_qkv > 64 ? max_qkv : 64);
    ws->head_scratch = alloc_zeroed(head_dim_max);
    ws->kv_keys_copy = alloc_zeroed(kv_copy_size);
    ws->kv_values_copy = alloc_zeroed(kv_copy_size);
    ws->logits = alloc_zeroed(cfg->vocab_size);
    ws->moe_router_logits = alloc_zeroed(ws->n_experts);
    ws->mamba_scratch = alloc_zeroed(h > 576 ? h : 576);
    ws->conv_out = alloc_zeroed(max_qkv);
    ws->shortconv_bcx = alloc_zeroed(h * 3);
    ws->shortconv_bx = alloc_zeroed(h);
    ws->moe_gate_all = alloc_zeroed(n_experts_per_tok * expert_inter);
    ws->moe_up_all = alloc_zeroed(n_experts_per_tok * expert_inter);
    ws->moe_down_all = alloc_zeroed(n_experts_per_tok * h);

    /* Check all allocations succeeded. */
    float *ptrs[] = { ws->x, ws->hidden_a, ws->hidden_b, ws->intermediate_a,
        ws->intermediate_b, ws->intermediate_c, ws->q_full, ws->k_vec, ws->v_vec,
        ws->attn_result, ws->flash_q, ws->head_scratch, ws->kv_keys_copy,
        ws->kv_values_copy, ws->logits, ws->moe_router_logits, ws->mamba_scratch,
        ws->conv_out, ws->shortconv_bcx, ws->shortconv_bx, ws->moe_gate_all,
        ws->moe_up_all, ws->moe_down_all };
    for (size_t i = 0; i < sizeof(ptrs) / sizeof(ptrs[0]); i++) {
        if (!ptrs[i]) {
            oc_workspace_free(ws);
            return OC_ERR_OOM;
        }
    }
    return OC_OK;
}

void oc_workspace_free(OcWorkspace *ws)
{
    if (!ws) return;
    free(ws->x);
    free(ws->hidden_a);
    free(ws->hidden_b);
    free(ws->intermediate_a);
    free(ws->intermediate_b);
    free(ws->intermediate_c);
    free(ws->q_full);
    free(ws->k_vec);
    free(ws->v_vec);
    free(ws->attn_result);
    free(ws->flash_q);
    free(ws->head_scratch);
    free(ws->kv_keys_copy);
    free(ws->kv_values_copy);
    free(ws->logits);
    free(ws->moe_router_logits);
    free(ws->mamba_scratch);
    free(ws->conv_out);
    free(ws->shortconv_bcx);
    free(ws->shortconv_bx);
    free(ws->moe_gate_all);
    free(ws->moe_up_all);
    free(ws->moe_down_all);
    memset(ws, 0, sizeof(*ws));
}

void oc_workspace_zero(OcWorkspace *ws)
{
    if (!ws) return;
    if (ws->x) memset(ws->x, 0, ws->hidden_size * sizeof(float));
    if (ws->hidden_a) memset(ws->hidden_a, 0, ws->hidden_size * sizeof(float));
    if (ws->hidden_b) memset(ws->hidden_b, 0, ws->hidden_size * sizeof(float));
    if (ws->intermediate_a) memset(ws->intermediate_a, 0, ws->intermediate_size * sizeof(float));
    if (ws->intermediate_b) memset(ws->intermediate_b, 0, ws->intermediate_size * sizeof(float));
    if (ws->intermediate_c) memset(ws->intermediate_c, 0, ws->intermediate_size * sizeof(float));
    if (ws->q_full) memset(ws->q_full, 0, ws->max_qkv * sizeof(float));
    if (ws->k_vec) memset(ws->k_vec, 0, ws->kv_scratch * sizeof(float));
    if (ws->v_vec) memset(ws->v_vec, 0, ws->kv_scratch * sizeof(float));
    if (ws->attn_result) memset(ws->attn_result, 0, ws->max_qkv * sizeof(float));
    if (ws->logits) memset(ws->logits, 0, ws->vocab_size * sizeof(float));
    if (ws->moe_router_logits) memset(ws->moe_router_logits, 0, ws->n_experts * sizeof(float));
}

size_t oc_workspace_size_bytes(const OcWorkspace *ws)
{
    if (!ws) return 0;
    size_t total = 0;
    total += ws->hidden_size * 2;  /* x, hidden_a, hidden_b -> 3 */
    total += ws->hidden_size;      /* hidden_b */
    total += ws->intermediate_size * 3;
    total += ws->max_qkv * 3;      /* q_full, attn_result, conv_out */
    total += ws->kv_scratch * 2;   /* k_vec, v_vec */
    total += (ws->max_qkv > 64 ? ws->max_qkv : 64);  /* flash_q */
    total += ws->head_dim;         /* head_scratch */
    total += ws->kv_copy_size * 2; /* kv_keys_copy, kv_values_copy */
    total += ws->vocab_size;       /* logits */
    total += ws->n_experts;        /* moe_router_logits */
    total += (ws->hidden_size > 576 ? ws->hidden_size : 576);  /* mamba_scratch */
    total += ws->hidden_size * 3;  /* shortconv_bcx */
    total += ws->hidden_size;      /* shortconv_bx */
    total += ws->n_experts_per_tok * ws->expert_inter * 2;  /* moe_gate_all, moe_up_all */
    total += ws->n_experts_per_tok * ws->hidden_size;       /* moe_down_all */
    return total * sizeof(float);
}
