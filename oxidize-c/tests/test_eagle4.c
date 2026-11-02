#include <criterion/criterion.h>
#include "oxidize/eagle4.h"
#include "oxidize/gguf.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void tiny(OcEagle4Config *cfg)
{
    oc_eagle4_config_init(cfg);
    cfg->hidden_size = 4;
    cfg->target_hidden_size = 4;
    cfg->n_extract_layers = 1;
    cfg->extract_layers[0] = 0;
    cfg->num_attention_heads = 1;
    cfg->num_key_value_heads = 1;
    cfg->head_dim = 4;
    cfg->intermediate_size = 4;
    cfg->vocab_size = 4;
    cfg->draft_vocab_size = 4;
    cfg->rank = 4;
    cfg->candidate_k = 1;
    cfg->sliding_window = 2;
    cfg->num_hidden_layers = 1;
    cfg->n_experts = 0;
}

static void ident(float *w, size_t n)
{
    size_t i;
    memset(w, 0, n * n * sizeof(float));
    for (i = 0; i < n; i++) w[i * n + i] = 1.0f;
}

Test(eagle4, released_shape_and_policy)
{
    OcEagle4Config cfg;
    OcEagle4Harness h;
    OcEagle4Policy tree, chain, narrow, wide_ctx;
    oc_eagle4_config_init(&cfg);
    oc_eagle4_config_set_released(&cfg);
    cr_assert_eq(cfg.sliding_window, 4096);
    cr_assert_eq(cfg.rank, 128);
    cr_assert_eq(cfg.candidate_k, 128);

    oc_eagle4_harness_init(&h);
    oc_eagle4_policy_for_batch(&h, 1, 128, &tree);
    oc_eagle4_policy_for_batch(&h, 64, 128, &chain);
    oc_eagle4_policy_for_batch(&h, 384, 128, &narrow);
    oc_eagle4_policy_for_batch(&h, 16, 40000, &wide_ctx);
    cr_assert_eq(tree.mode, OC_EAGLE4_MODE_TREE);
    cr_assert_eq(tree.max_branch, 4);
    cr_assert_eq(tree.verify_cap, 16);
    cr_assert_eq(chain.mode, OC_EAGLE4_MODE_CHAIN);
    cr_assert_eq(chain.verify_cap, 4);
    cr_assert_eq(chain.max_branch, 1);
    cr_assert_eq(narrow.verify_cap, 3);
    cr_assert_eq(wide_ctx.verify_cap, 8);
    cr_assert_eq(wide_ctx.max_branch, 1);
}

Test(eagle4, select_level_branches_early_and_stops)
{
    OcEagle4Harness h;
    OcEagle4Policy policy;
    OcEagle4Cand cands[4];
    uint32_t chosen[4];
    uint32_t n, i;
    oc_eagle4_harness_init(&h);
    h.min_gain_per_us = 0.0f;
    oc_eagle4_policy_for_batch(&h, 1, 32, &policy);
    for (i = 0; i < 4; i++) {
        cands[i].parent = -1;
        cands[i].cond_prob = 0.5f / (float)(i + 1);
        cands[i].path_prob = cands[i].cond_prob;
    }
    n = oc_eagle4_select_level(&h, &policy, 1, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 4);

    h.min_gain_per_us = 1.0e9f;
    n = oc_eagle4_select_level(&h, &policy, 1, 2, 2, cands, 4, chosen, 4);
    cr_assert_eq(n, 0);
    n = oc_eagle4_select_level(&h, &policy, 1, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 1);

    oc_eagle4_policy_for_batch(&h, 64, 32, &policy);
    h.min_gain_per_us = 0.0f;
    n = oc_eagle4_select_level(&h, &policy, 64, 1, 0, cands, 4, chosen, 4);
    cr_assert_eq(n, 1);
    cr_assert_eq(policy.verify_cap, 4);
}

Test(eagle4, mask_and_greedy_accept)
{
    OcEagle4Tree tree;
    uint8_t mask[9];
    uint32_t next[3];
    uint32_t out[4];
    uint32_t n;
    memset(&tree, 0, sizeof(tree));
    tree.n_nodes = 3;
    tree.nodes[0].token = 1;
    tree.nodes[0].parent = -1;
    tree.nodes[1].token = 7;
    tree.nodes[1].parent = -1;
    tree.nodes[2].token = 2;
    tree.nodes[2].parent = 0;
    oc_eagle4_tree_mask(&tree, mask);
    cr_assert_eq(mask[0], 1);
    cr_assert_eq(mask[1], 0);
    cr_assert_eq(mask[1 * 3 + 1], 1);
    cr_assert_eq(mask[2 * 3 + 0], 1);
    cr_assert_eq(mask[2 * 3 + 2], 1);
    cr_assert_eq(mask[2 * 3 + 1], 0);

    next[0] = 2;
    next[1] = 0;
    next[2] = 9;
    n = oc_eagle4_accept_greedy(&tree, 1, next, out, 4);
    cr_assert_eq(n, 2);
    cr_assert_eq(out[0], 1);
    cr_assert_eq(out[1], 2);
    n = oc_eagle4_accept_greedy(&tree, 3, next, out, 4);
    cr_assert_eq(n, 0);
}

Test(eagle4, speculative_accept_is_deterministic)
{
    float target[4] = {0, 1, 0, 0};
    float draft[4] = {1, 0, 0, 0};
    float one[4] = {0, 0, 0, 1};
    OcEagle4Step steps[2];
    uint32_t out[4];
    uint32_t rng = 1;
    uint32_t n;
    memset(steps, 0, sizeof(steps));
    steps[0].draft_token = 0;
    steps[0].p = 0.0f;
    steps[0].q = 1.0f;
    steps[0].target_dist = target;
    steps[0].draft_dist = draft;
    steps[0].vocab = 4;
    n = oc_eagle4_speculative_accept(steps, 1, &rng, out, 4);
    cr_assert_eq(n, 1);
    cr_assert_eq(out[0], 1);

    steps[0].p = 1.0f;
    steps[0].q = 1.0f;
    steps[0].draft_token = 4;
    steps[0].target_dist = one;
    steps[0].draft_dist = one;
    steps[1] = steps[0];
    steps[1].draft_token = 5;
    n = oc_eagle4_speculative_accept(steps, 2, &rng, out, 4);
    cr_assert_eq(n, 3);
    cr_assert_eq(out[0], 4);
    cr_assert_eq(out[1], 5);
    cr_assert_eq(out[2], 3);
}

Test(eagle4, low_rank_head_touches_k_rows_and_window)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float *logits;
    uint64_t before;
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 2;
    cfg.sliding_window = 2;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++) m.tok_embeddings.data[i * 4 + i] = 1.0f;
    logits = (float *)calloc(4, sizeof(float));
    cr_assert_not_null(logits);
    oc_eagle4_reset_stats(&m);
    cr_assert_eq(oc_eagle4_forward(&m, 1, logits, 4), OC_OK);
    cr_assert_eq(m.stats.exact_rows, 2);
    cr_assert_eq(m.stats.score_rows, 4);
    cr_assert(logits[1] > logits[0]);

    for (i = 0; i < 3; i++) cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    before = m.stats.attn_keys;
    cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    cr_assert_eq(m.stats.attn_keys - before, 2);
    free(logits);
    oc_eagle4_model_free(&m);
}

Test(eagle4, d2t_scatters_exact_logit)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float logits[4];
    static uint32_t mapv[4] = {0, 3, 1, 2};
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 1;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++) m.tok_embeddings.data[i * 4 + i] = 1.0f;
    m.d2t = mapv;
    m.n_d2t = 4;
    cr_assert_eq(oc_eagle4_forward(&m, 1, logits, 4), OC_OK);
    cr_assert(logits[3] > logits[0]);
    cr_assert(logits[1] < logits[3]);
    m.d2t = NULL;
    oc_eagle4_model_free(&m);
}

Test(eagle4, propose_chain_then_commit)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    OcEagle4Harness h;
    OcEagle4Tree tree;
    uint32_t i;
    tiny(&cfg);
    cfg.candidate_k = 1;
    cfg.sliding_window = 16;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_down, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_score, 4, 4), OC_OK);
    ident(m.lm_head.data, 4);
    ident(m.lm_down.data, 4);
    ident(m.lm_score.data, 4);
    for (i = 0; i < 4; i++)
        m.tok_embeddings.data[i * 4 + ((i + 1) % 4)] = 1.0f;
    oc_eagle4_harness_init(&h);
    h.max_depth = 4;
    h.max_branch = 1;
    h.verify_cap = 4;
    h.min_gain_per_us = 0.0f;
    h.tree_batch_max = 4;
    cr_assert_eq(oc_eagle4_propose(&m, &h, 0, 1, 8, &tree), OC_OK);
    cr_assert_eq(tree.n_nodes, 4);
    cr_assert_eq(tree.nodes[0].token, 1);
    cr_assert_eq(tree.nodes[1].token, 2);
    cr_assert_eq(tree.nodes[2].token, 3);
    cr_assert_eq(tree.nodes[3].token, 0);
    cr_assert_eq(oc_eagle4_kv_len(&m), 1);
    cr_assert_eq(oc_eagle4_commit(&m, (uint32_t[]){1, 2, 3, 0}, 4), OC_OK);
    cr_assert_eq(oc_eagle4_kv_len(&m), 5);
    oc_eagle4_model_free(&m);
}

Test(eagle4, moe_expert_is_selected)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    tiny(&cfg);
    cfg.hidden_size = 2;
    cfg.head_dim = 2;
    cfg.intermediate_size = 2;
    cfg.vocab_size = 2;
    cfg.draft_vocab_size = 2;
    cfg.n_experts = 2;
    cfg.top_k_experts = 1;
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_f32(&m, &m.layers[0].router, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_gate[0], 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_up[0], 2, 2), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.layers[0].expert_down[0], 2, 2), OC_OK);
    m.tok_embeddings.data[0] = 1.0f;
    m.layers[0].router[0] = 10.0f;
    m.layers[0].expert_gate[0].data[0] = 4.0f;
    m.layers[0].expert_up[0].data[0] = 1.0f;
    m.layers[0].expert_down[0].data[0] = 1.0f;
    m.layers[0].expert_down[0].data[3] = 1.0f;
    cr_assert_eq(oc_eagle4_forward(&m, 0, NULL, 0), OC_OK);
    cr_assert(m.g_embeddings[0] > 4.0f);
    oc_eagle4_model_free(&m);
}

Test(eagle4, train_step_lowers_loss_and_refuses_gguf)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    float features[4] = {0.2f, 0, 0, 0};
    float before = 0, after = 0;
    uint32_t pred = 99;
    int step;
    char path[64];
    tiny(&cfg);
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    m.tok_embeddings.data[0] = 1.0f;
    cr_assert_eq(oc_eagle4_eval(&m, features, 4, 0, 2, &before, NULL), OC_OK);
    for (step = 0; step < 40; step++)
        cr_assert_eq(oc_eagle4_train_step(&m, features, 4, 0, 2, 0.4f, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&m, features, 4, 0, 2, &after, &pred), OC_OK);
    cr_assert(after < before);
    cr_assert_eq(pred, 2);
    cr_assert(oc_eagle4_path_is_gguf("draft.gguf"));
    cr_assert(oc_eagle4_path_is_gguf("Draft.GGUF"));
    cr_assert(!oc_eagle4_path_is_gguf("draft.safetensors"));
    snprintf(path, sizeof(path), "/tmp/oc-eagle4-%d.gguf", (int)getpid());
    cr_assert_eq(oc_eagle4_train_safetensors(&m, path, 0.1f, 1, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_eagle4_load_safetensors(&m, path), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_eagle4_write_examples(path, features, 1, 4,
                                         (uint32_t[]){0}, (uint32_t[]){2}),
                 OC_ERR_INVALID_ARG);
    oc_eagle4_model_free(&m);
}

Test(eagle4, safetensors_roundtrip_trains_and_exports_gguf)
{
    OcEagle4Config cfg;
    OcEagle4Draft m, loaded;
    float features[4] = {1, 0, 0, 0};
    uint32_t tokens[1] = {0};
    uint32_t targets[1] = {1};
    float before, after;
    char st_path[64], ex_path[64], gg_path[64];
    OcGgufFile gg;
    const OcGgufTensorInfo *ti;
    int pid = (int)getpid();
    tiny(&cfg);
    cr_assert_eq(oc_eagle4_model_init(&m, &cfg), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.tok_embeddings, 4, 4), OC_OK);
    cr_assert_eq(oc_eagle4_alloc_weight(&m, &m.lm_head, 4, 4), OC_OK);
    m.tok_embeddings.data[0] = 0.5f;
    m.tok_embeddings.data[5] = 0.25f;
    snprintf(st_path, sizeof(st_path), "/tmp/oc-eagle4-%d.safetensors", pid);
    snprintf(ex_path, sizeof(ex_path), "/tmp/oc-eagle4-ex-%d.safetensors", pid);
    snprintf(gg_path, sizeof(gg_path), "/tmp/oc-eagle4-out-%d.gguf", pid);
    cr_assert_eq(oc_eagle4_save_safetensors(&m, st_path), OC_OK);
    memset(&loaded, 0, sizeof(loaded));
    cr_assert_eq(oc_eagle4_load_safetensors(&loaded, st_path), OC_OK);
    cr_assert_float_eq(loaded.tok_embeddings.data[0], 0.5f, 1e-6);
    cr_assert_float_eq(loaded.tok_embeddings.data[5], 0.25f, 1e-6);
    cr_assert_eq(loaded.config.hidden_size, 4);
    cr_assert_eq(oc_eagle4_write_examples(ex_path, features, 1, 4, tokens, targets), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&loaded, features, 4, 0, 1, &before, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_train_safetensors(&loaded, ex_path, 0.4f, 30, NULL), OC_OK);
    cr_assert_eq(oc_eagle4_eval(&loaded, features, 4, 0, 1, &after, NULL), OC_OK);
    cr_assert(after < before);
    cr_assert_eq(oc_eagle4_export_gguf(&loaded, gg_path), OC_OK);
    memset(&gg, 0, sizeof(gg));
    cr_assert_eq(oc_gguf_open(gg_path, &gg), OC_OK);
    ti = oc_gguf_tensor_get(&gg, "eagle4.embed");
    cr_assert_not_null(ti);
    cr_assert_eq(ti->ggml_type, 0);
    cr_assert_eq(memcmp(gg.backing_buf + ti->absolute_offset,
                        loaded.tok_embeddings.data, 16), 0);
    cr_assert_not_null(oc_gguf_metadata_get(&gg, "eagle4.format"));
    oc_gguf_free(&gg);
    oc_eagle4_model_free(&loaded);
    oc_eagle4_model_free(&m);
    unlink(st_path);
    unlink(ex_path);
    unlink(gg_path);
}
