/*
 * eagle4.h — EAGLE-4 speculative decoding for the C engine.
 *
 * Independent implementation of the published EAGLE-4 drafter: one decoder
 * layer over fused target hidden states, sliding-window attention, a rank-128
 * candidate head, and a cost-model harness that widens the tree at batch 1
 * and narrows to a chain as the batch grows. No Eagle AI source is included.
 *
 * Training and checkpoints are SafeTensors. GGUF is an export of a draft that
 * was already trained from SafeTensors; it is not a training input. Weights
 * are not shipped here.
 */
#ifndef OXIDIZE_EAGLE4_H
#define OXIDIZE_EAGLE4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_EAGLE4_MAX_LAYERS 4u
#define OC_EAGLE4_MAX_EXTRACT 8u
#define OC_EAGLE4_MAX_EXPERTS 8u
#define OC_EAGLE4_MAX_NODES 64u
#define OC_EAGLE4_TOP_KEEP 8u
#define OC_EAGLE4_RELEASED_WINDOW 4096u
#define OC_EAGLE4_RELEASED_RANK 128u
#define OC_EAGLE4_RELEASED_CANDIDATES 128u
#define OC_EAGLE4_RELEASED_DEPTH 12u
#define OC_EAGLE4_RELEASED_BRANCH 4u
#define OC_EAGLE4_RELEASED_VERIFY 16u

typedef struct OcEagle4Weight {
    float *data;
    size_t rows;
    size_t cols;
} OcEagle4Weight;

typedef struct OcEagle4Config {
    size_t hidden_size;
    size_t target_hidden_size;
    size_t n_extract_layers;
    size_t extract_layers[OC_EAGLE4_MAX_EXTRACT];
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t head_dim;
    size_t intermediate_size;
    size_t vocab_size;
    size_t draft_vocab_size;
    size_t rank;
    size_t candidate_k;
    size_t sliding_window;
    size_t num_hidden_layers;
    size_t n_experts;
    size_t top_k_experts;
    float rms_norm_eps;
    float rope_theta;
    bool use_rope;
    bool norm_before_residual;
} OcEagle4Config;

typedef struct OcEagle4Layer {
    float *attn_norm;
    OcEagle4Weight q_proj;
    OcEagle4Weight k_proj;
    OcEagle4Weight v_proj;
    OcEagle4Weight o_proj;
    float *ffn_norm;
    OcEagle4Weight mlp_gate;
    OcEagle4Weight mlp_up;
    OcEagle4Weight mlp_down;
    float *router;
    OcEagle4Weight expert_gate[OC_EAGLE4_MAX_EXPERTS];
    OcEagle4Weight expert_up[OC_EAGLE4_MAX_EXPERTS];
    OcEagle4Weight expert_down[OC_EAGLE4_MAX_EXPERTS];
} OcEagle4Layer;

typedef struct OcEagle4Stats {
    uint64_t exact_rows;
    uint64_t score_rows;
    uint64_t attn_keys;
} OcEagle4Stats;

typedef struct OcEagle4Draft {
    OcEagle4Config config;
    OcEagle4Weight fc;
    float *fc_norm[OC_EAGLE4_MAX_EXTRACT];
    float *fc_post_norm;
    OcEagle4Weight eh_proj;
    OcEagle4Weight tok_embeddings;
    float *embed_norm;
    float *input_norm;
    float *output_norm;
    OcEagle4Weight lm_head;
    OcEagle4Weight lm_down;
    OcEagle4Weight lm_score;
    uint32_t *d2t;
    size_t n_d2t;
    OcEagle4Layer layers[OC_EAGLE4_MAX_LAYERS];
    bool full_lm_head;
    bool loaded;
    OcEagle4Stats stats;
    float *g_embeddings;
    size_t position;
    size_t kv_seq0;
    /* Owned by the model. Not part of the checkpoint. */
    void *impl;
} OcEagle4Draft;

typedef enum OcEagle4Mode {
    OC_EAGLE4_MODE_TREE = 0,
    OC_EAGLE4_MODE_CHAIN = 1
} OcEagle4Mode;

typedef struct OcEagle4Harness {
    float draft_step_us;
    float verify_per_row_us;
    float min_gain_per_us;
    uint32_t max_depth;
    uint32_t max_branch;
    uint32_t verify_cap;
    uint32_t tree_batch_max;
    uint32_t chain_width_batch;
    uint32_t narrow_batch;
} OcEagle4Harness;

typedef struct OcEagle4Policy {
    OcEagle4Mode mode;
    uint32_t max_depth;
    uint32_t max_branch;
    uint32_t verify_cap;
} OcEagle4Policy;

typedef struct OcEagle4Cand {
    int32_t parent;
    float cond_prob;
    float path_prob;
} OcEagle4Cand;

typedef struct OcEagle4Node {
    uint32_t token;
    int32_t parent;
    float cond_prob;
    float path_prob;
    uint32_t depth;
} OcEagle4Node;

typedef struct OcEagle4Tree {
    OcEagle4Node nodes[OC_EAGLE4_MAX_NODES];
    uint32_t n_nodes;
} OcEagle4Tree;

typedef struct OcEagle4Step {
    uint32_t draft_token;
    float p;
    float q;
    const float *target_dist;
    const float *draft_dist;
    uint32_t vocab;
} OcEagle4Step;

typedef struct OcEagle4TrainResult {
    float loss;
    uint32_t pred;
    uint32_t target;
    uint32_t steps;
} OcEagle4TrainResult;

typedef struct OcEagle4BenchReport {
    double head_full_us;
    double head_lowrank_us;
    double attn_full_us;
    double attn_window_us;
    double head_speedup;
    double attn_speedup;
    uint32_t batch1_cap;
    uint32_t batch1_branch;
    uint32_t batch64_cap;
    uint32_t batch384_cap;
    uint32_t batch1_selected;
} OcEagle4BenchReport;

void oc_eagle4_config_init(OcEagle4Config *cfg);
void oc_eagle4_config_set_released(OcEagle4Config *cfg);
void oc_eagle4_harness_init(OcEagle4Harness *h);

OcError oc_eagle4_model_init(OcEagle4Draft *m, const OcEagle4Config *cfg);
void oc_eagle4_model_free(OcEagle4Draft *m);
OcError oc_eagle4_alloc_f32(OcEagle4Draft *m, float **dst, size_t n);
OcError oc_eagle4_alloc_weight(OcEagle4Draft *m, OcEagle4Weight *w,
                               size_t rows, size_t cols);
void oc_eagle4_reset_cache(OcEagle4Draft *m);
void oc_eagle4_reset_stats(OcEagle4Draft *m);
size_t oc_eagle4_kv_len(const OcEagle4Draft *m);

OcError oc_eagle4_encode_features(OcEagle4Draft *m, const float *features,
                                  size_t n_features);
OcError oc_eagle4_forward(OcEagle4Draft *m, uint32_t token, float *out_logits,
                          size_t logits_len);

void oc_eagle4_policy_for_batch(const OcEagle4Harness *h, uint32_t batch,
                                size_t context_len, OcEagle4Policy *out);
uint32_t oc_eagle4_select_level(const OcEagle4Harness *h,
                                const OcEagle4Policy *policy, uint32_t batch,
                                uint32_t depth, uint32_t already,
                                const OcEagle4Cand *cands, uint32_t n_cands,
                                uint32_t *chosen, uint32_t max_chosen);

OcError oc_eagle4_propose(OcEagle4Draft *m, const OcEagle4Harness *h,
                          uint32_t prefix_token, uint32_t batch,
                          size_t context_len, OcEagle4Tree *out);
OcError oc_eagle4_commit(OcEagle4Draft *m, const uint32_t *tokens, size_t n);
void oc_eagle4_tree_mask(const OcEagle4Tree *tree, uint8_t *mask);
uint32_t oc_eagle4_accept_greedy(const OcEagle4Tree *tree, uint32_t first_target,
                                 const uint32_t *next_target, uint32_t *out,
                                 uint32_t max_out);
uint32_t oc_eagle4_speculative_accept(const OcEagle4Step *steps, uint32_t n_steps,
                                      uint32_t *rng, uint32_t *out,
                                      uint32_t max_out);

/* SafeTensors is the training and checkpoint format. A path ending in .gguf
 * is rejected. export_gguf writes a package from an in-memory draft; it does
 * not read GGUF and it is not called by the trainer. */
bool oc_eagle4_path_is_gguf(const char *path);
OcError oc_eagle4_save_safetensors(const OcEagle4Draft *m, const char *path);
OcError oc_eagle4_load_safetensors(OcEagle4Draft *m, const char *path);
OcError oc_eagle4_write_examples(const char *path, const float *features,
                                 size_t n, size_t width, const uint32_t *tokens,
                                 const uint32_t *targets);
OcError oc_eagle4_export_gguf(const OcEagle4Draft *m, const char *path);

OcError oc_eagle4_eval(OcEagle4Draft *m, const float *features, size_t n_features,
                       uint32_t token, uint32_t target, float *out_loss,
                       uint32_t *out_pred);
OcError oc_eagle4_train_step(OcEagle4Draft *m, const float *features,
                             size_t n_features, uint32_t token, uint32_t target,
                             float lr, OcEagle4TrainResult *out);
OcError oc_eagle4_train_safetensors(OcEagle4Draft *m, const char *examples_path,
                                    float lr, uint32_t epochs,
                                    OcEagle4TrainResult *out);

OcError oc_eagle4_bench(int iters, OcEagle4BenchReport *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_EAGLE4_H */
