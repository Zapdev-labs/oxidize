/* oxidize-c: plain-C runtime + tools surface (GGUF parse, quant/dequant,
 * tokenizer, model forward, generation, server, pruning, finetune, optional CUDA).
 * Errors exit(1) with a message for CLI-first tooling. */
#ifndef OC_H
#define OC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ---- quant types (subset of oxidize-cpp QuantType we can run) ---- */
typedef enum {
  OC_F32, OC_F16, OC_BF16,
  OC_Q4_0, OC_Q4_1, OC_Q5_0, OC_Q5_1, OC_Q8_0,
  OC_Q2_K, OC_Q3_K, OC_Q4_K, OC_Q5_K, OC_Q6_K,
  OC_IQ4_XS,
  OC_IQ2_XXS, OC_IQ2_XS, OC_IQ2_S, OC_IQ3_XXS, OC_IQ3_S, OC_IQ4_NL,
  OC_AL5,       /* custom: Q4_0-sized (18B) symmetric 4-bit, MSE-optimal scale */
  OC_AL8,       /* custom: Q8_0-sized (34B) 8-bit with refined scale */
  OC_AL6,       /* custom: Q5_0-sized (22B) ~6-bit with refined scale */
  OC_AL5_XS,    /* custom: 3-bit symmetric (14B/32w) */
  OC_UNKNOWN,
} oc_quant;

#define QK 32          /* small-block element count */
#define QK_K 256       /* K-quant superblock element count */
#define QK4_NL 32      /* IQ4_NL block element count */

size_t oc_block_values(oc_quant q);
size_t oc_block_bytes(oc_quant q);
size_t oc_row_bytes(oc_quant q, size_t cols);
oc_quant oc_from_ggml_type(uint32_t t);
uint32_t oc_to_ggml_type(oc_quant q);
const char *oc_quant_name(oc_quant q);
oc_quant oc_quant_parse(const char *name); /* OC_UNKNOWN if unrecognized */

float oc_f16_to_f32(const uint8_t *le2);
uint16_t oc_f32_to_f16(float f);
void oc_dequant_row(oc_quant q, const uint8_t *src, float *dst, size_t n);
/* Quantize one row of n f32 values into dst. Supported targets: F32, F16,
 * Q8_0, Q4_0 (the set the prune/finetune tools re-emit). n % block == 0. */
bool oc_quantize_row(oc_quant q, const float *src, uint8_t *dst, size_t n);

/* ---- activation int8 blocks (for fused integer gemv) ----
 * x is quantized once per matvec into blocks of 32: scale d and s = d*sum(q).
 * s lets K-quant mins fold in without a second pass. */
typedef struct { float d, s; int8_t q[QK]; } oc_q8blk;
void oc_quantize_act(const float *x, oc_q8blk *out, size_t n); /* n%32==0 */

/* ---- kernels ---- */
void oc_rms_norm(float *out, const float *x, const float *w, size_t n, float eps);
/* NeoX split-half partial rope with optional YaRN (yarn_factor > 1 enables).
 * ff = optional per-dim frequency divisors [rope_dim/2] (ggml freq_factors). */
void oc_rope(float *vec, size_t head_dim, size_t n_heads, size_t pos,
             float theta, size_t rope_dim, float yarn_factor, float yarn_orig_ctx,
             const float *ff);
void oc_swiglu(float *gate, const float *up, size_t n);   /* gate = silu(gate)*up */
void oc_geglu(float *gate, const float *up, size_t n);    /* gate = gelu_tanh(gate)*up */
void oc_softmax(float *x, size_t n);
float oc_dot_f32(const float *a, const float *b, size_t n);
/* y = W*x. W either f32 (wf) or quantized block rows (wq). xq = pre-quantized
 * activation blocks (may be NULL for f32/f16/bf16 weights). */
typedef struct {
  bool quantized;
  oc_quant quant;
  const uint8_t *data;   /* quantized rows (mmap-backed) */
  float *f32;            /* dense rows (owned) */
  size_t rows, cols;
} oc_weight;
void oc_gemv(const oc_weight *w, size_t rows, size_t cols, const float *x,
             const oc_q8blk *xq, float *y);
/* batched: inputs [batch x cols] (+ per-row quantized blocks), outputs [batch x rows] */
void oc_gemm(const oc_weight *w, size_t rows, size_t cols, const float *in,
             const oc_q8blk *inq, float *out, size_t batch);
/* scale = score multiplier (pass 0 for the default 1/sqrt(head_dim)) */
void oc_attention(float *out, const float *q, const float *k_cache,
                  const float *v_cache, size_t seq_len, size_t n_heads,
                  size_t kv_heads, size_t head_dim, float scale);
/* int8 KV cache: k8/v8 are per-(pos,head) symmetric int8; ks/vs the scales
 * [pos*kv_heads + head]. Query stays f32. */
void oc_attention_q8(float *out, const float *q, const int8_t *k8,
                     const float *ks, const int8_t *v8, const float *vs,
                     size_t seq_len, size_t n_heads, size_t kv_heads,
                     size_t head_dim, float scale);
/* Quantize a K/V row [n_kv*hd] into int8 with one scale per head. */
void oc_quantize_kv(const float *x, int8_t *q, float *scale, size_t n_kv,
                    size_t hd);

/* ---- GGUF ---- */
typedef struct {
  char *key;
  int kind;              /* 0 scalar-num, 1 string, 2 array */
  double num;            /* scalar numeric (widened) */
  char *str;             /* scalar string */
  /* array payload */
  size_t count;
  bool is_str;
  char **strs;
  double *nums;
} oc_meta;

typedef struct {
  char *name;
  uint64_t dims[4];
  uint32_t n_dims;
  oc_quant quant;
  uint32_t ggml_type;    /* raw GGUF type id (kept for writer pass-through) */
  uint64_t offset;       /* absolute into mmap */
} oc_tensor_info;

typedef struct {
  void *map; size_t map_size;
  const uint8_t *base;
  oc_meta *meta; size_t n_meta;
  oc_tensor_info *tensors; size_t n_tensors;
  /* writer support: raw KV section is copied verbatim into pruned outputs */
  uint32_t version;
  size_t kv_off, kv_end; /* byte range of the metadata KV section */
  uint64_t align;        /* general.alignment (default 32) */
} oc_gguf;

oc_gguf *oc_gguf_load(const char *path);
void oc_gguf_free(oc_gguf *g);
const oc_meta *oc_meta_get(const oc_gguf *g, const char *key);
bool oc_meta_u32(const oc_gguf *g, const char *key, uint32_t *out);
bool oc_meta_f32(const oc_gguf *g, const char *key, float *out);
const char *oc_meta_str(const oc_gguf *g, const char *key); /* NULL if absent */
const oc_tensor_info *oc_find_tensor(const oc_gguf *g, const char *name);

/* ---- model ---- */
typedef struct {
  size_t vocab_size, context_size, layer_count, hidden_size, intermediate_size;
  size_t n_heads, kv_heads, head_dim, rope_dim, sliding_window;
  float rms_eps, rope_theta;
  float yarn_factor, yarn_orig_ctx;  /* rope scaling (0 = off) */
  size_t max_ctx;          /* CLI-capped KV context (0 = context_size) */
  int kv_int8;             /* 1 = store KV cache as int8 (per-head scale) */
  size_t n_expert;         /* MoE: total experts (0 = dense) */
  size_t n_expert_used;    /* MoE: top-k experts per token */
  size_t expert_ff;        /* MoE: per-expert intermediate size */
  int expert_weights_norm; /* MoE: renormalize the top-k routing weights */
} oc_config;

typedef struct {
  float *attn_norm, *ffn_norm;
  oc_weight wq, wk, wv, wo, gate, up, down;
  /* MoE: router + expert-stacked FFN weights (per-expert 2D slices). */
  bool is_moe;
  oc_weight router, e_gate, e_up, e_down;
  float *q_bias, *k_bias, *v_bias;         /* NULL when absent */
  size_t q_bias_n, k_bias_n, v_bias_n;
  float *q_norm, *k_norm;                  /* per-head (Qwen3), NULL when absent */
  int kv_slot;                             /* index into KV cache (-1 for GDN) */

  /* Per-layer attention geometry. Set for every attention layer at load;
   * non-gemma archs get the global config values. */
  size_t hd, n_kv, n_rot;                  /* head_dim, kv heads, rope dims */
  float theta;                             /* rope base */
  const float *rope_ff;                    /* freq divisors [n_rot/2] (borrowed) */
  float attn_scale;                        /* 0 = default 1/sqrt(hd) */
  bool v_from_k;                           /* gemma4 k==v: V = k-proj output */
  bool v_rms;                              /* unweighted per-head RMS on V */
  float *attn_post_norm, *ffn_post_norm;   /* gemma post-norms, NULL else */
  float out_scale_v;                       /* per-layer output scalar (1.0) */
  float *kv_ck, *kv_cv;                    /* private KV cache (gemma; owned) */
  int8_t *kv_ck8, *kv_cv8;                 /* int8 variant (cfg.kv_int8) */
  float *kv_cks, *kv_cvs;                  /* per-(pos,head) scales for int8 */
  size_t kv_cap;                           /* positions in private cache */

  /* Gated-DeltaNet (qwen3.5 linear-attention) layer, detected by ssm_a. */
  bool is_gdn;
  oc_weight qkv, ssm_alpha, ssm_beta, ssm_out, gdn_gate;
  float *ssm_a, *ssm_dt_bias;              /* [n_v_heads] */
  float *ssm_conv1d;                       /* [4 * qkv_out], tap-major */
  float *ssm_norm;                         /* [head_v_dim] */
  size_t qkv_out, value_dim, key_dim;
  size_t n_v_heads, n_k_heads, head_k, head_v;
  /* recurrent state (owned) */
  float *state;                            /* [n_v_heads * head_k * head_v] */
  float *conv_ring;                        /* [4 * qkv_out] ring */
  int ring_head, ring_len;
} oc_layer;

/* MTP/nextn draft block (qwen3.5): one extra attention layer + fuse proj. */
typedef struct {
  oc_layer layer;                          /* attention layer weights */
  oc_weight eh_proj;                       /* [h, 2h] */
  float *enorm, *hnorm, *head_norm;
  float *kv_k, *kv_v;                      /* small dedicated KV [draft_max] */
  size_t draft_max;
} oc_mtp;

typedef struct {
  oc_gguf *g;
  oc_config cfg;
  oc_weight tok_emb;      /* row-gathered, kept quantized */
  float *final_norm;
  oc_weight lm_head;      /* may alias tok_emb (tied) */
  bool tied;
  oc_layer *layers;
  size_t n_kv_layers;     /* attention layers only */
  float *kv_k, *kv_v;     /* [kv_slot][pos][kv_heads*head_dim] */
  int8_t *kv_k8, *kv_v8;  /* int8 variant (cfg.kv_int8); same indexing */
  float *kv_ks, *kv_vs;   /* per-(slot,pos,head) scales for int8 */
  size_t kv_stride;       /* kv_heads*head_dim */
  size_t kv_ctx;          /* allocated positions per layer */
  float *x;               /* hidden state [hidden] */
  oc_mtp *mtp;            /* NULL if the GGUF has no nextn block */
  bool gpu_active;        /* resident CUDA forward built (OC_CUDA builds only) */
  /* gemma4 */
  bool gemma;             /* GELU FFN, post-norms, per-layer KV, emb scaling */
  float emb_scale;        /* sqrt(hidden) for gemma, else 1.0 */
  float logit_softcap;    /* 0 = off */
  float *rope_freqs;      /* shared full-attn freq divisors (owned) */
} oc_model;

/* max_ctx: 0 = model default. kv_int8: 1 = int8 KV cache (4x smaller). */
oc_model *oc_model_load(const char *path, size_t max_ctx, int kv_int8);
void oc_model_free(oc_model *m);
/* Run tokens[0..n) starting at absolute position start_pos; returns logits for
 * the last token in caller-provided buf [vocab]. */
void oc_forward(oc_model *m, const uint32_t *tokens, size_t n, size_t start_pos,
                float *logits);
/* Same, but writes logits for EVERY position into logits_all [n * vocab]
 * (batched lm_head — used by speculative verify). */
void oc_forward_all(oc_model *m, const uint32_t *tokens, size_t n,
                    size_t start_pos, float *logits_all);
/* Training forward: writes the post-final-norm hidden state for every
 * position into normed_all [n * hidden] AND logits into logits_all
 * [n * vocab] (finetune needs the hidden rows for the LoRA head). */
void oc_forward_train(oc_model *m, const uint32_t *tokens, size_t n,
                      size_t start_pos, float *normed_all, float *logits_all);
/* Reset recurrent state (SSM states + conv rings) for a fresh sequence. */
void oc_reset_state(oc_model *m);
/* Snapshot/restore recurrent state (for speculative rollback). Buffer size
 * from oc_state_bytes(). */
size_t oc_state_bytes(const oc_model *m);
void oc_state_save(const oc_model *m, uint8_t *buf);
void oc_state_load(oc_model *m, const uint8_t *buf);
/* MTP: draft up to k greedy tokens continuing after `start_token`, anchored on
 * the model's current last hidden state. Returns number drafted. */
size_t oc_mtp_draft(oc_model *m, uint32_t start_token, size_t start_pos, size_t k,
                    uint32_t *out);

/* ---- tokenizer ---- */
typedef struct oc_tokenizer oc_tokenizer;
oc_tokenizer *oc_tokenizer_load(const oc_gguf *g); /* NULL if unsupported */
void oc_tokenizer_free(oc_tokenizer *t);
/* returns malloc'd id array, count in *n_out */
uint32_t *oc_tokenize(const oc_tokenizer *t, const char *text, bool add_bos, size_t *n_out);
/* decoded fragment for one token; writes into caller buf, returns bytes */
size_t oc_detokenize(const oc_tokenizer *t, uint32_t id, char *buf, size_t cap);
bool oc_is_eog(const oc_tokenizer *t, uint32_t id);

/* ---- Resident CUDA forward (only defined in -DOC_CUDA builds) ----
 * oc_cuda_build uploads FP16 weights + norms + KV/SSM state to the GPU and sets
 * m->gpu_active. oc_cuda_forward runs the ENTIRE per-token forward on-device:
 * `embed_host` is the (host, dequantized+scaled) embedding row for one token at
 * absolute position `pos`; logits are copied back only when want_logits. */
int oc_cuda_build(oc_model *m);            /* 0 ok, -1 no GPU / disabled */
void oc_cuda_forward(oc_model *m, const float *embed_host, size_t pos,
                     int want_logits, float *logits_host);
void oc_cuda_reset(oc_model *m);           /* zero device SSM state + conv rings */

/* ---- tools (prune.c / finetune.c): CLI subcommand entry points ---- */
int oc_prune_main(int argc, char **argv);
int oc_finetune_main(int argc, char **argv);

/* prune primitives (exposed for test_oc) */
void oc_magnitude_mask(const float *w, size_t rows, size_t cols, float sparsity,
                       bool *mask);
void oc_wanda_mask(const float *w, const float *col_norms, size_t rows,
                   size_t cols, float sparsity, bool *mask);
/* structured N:M on top of a base mask (keep n of every m per row) */
void oc_nm_mask(const float *scores, size_t rows, size_t cols, size_t n,
                size_t m, bool *mask);

/* LoRA adapter (exposed for test_oc): out += scale * B(Ax) */
typedef struct {
  size_t in_dim, out_dim, rank;
  float scale;
  float *a, *b;              /* a[rank*in], b[out*rank] */
  float *grad_a, *grad_b;
  float *m_a, *v_a, *m_b, *v_b; /* AdamW state */
} oc_lora;
oc_lora *oc_lora_new(size_t in_dim, size_t out_dim, size_t rank, float alpha,
                     uint64_t seed);
void oc_lora_free(oc_lora *l);
void oc_lora_forward(const oc_lora *l, const float *xs, float *outs, size_t count);
void oc_lora_backward(oc_lora *l, const float *xs, const float *grad_outs,
                      size_t count);
void oc_lora_step(oc_lora *l, float lr, float weight_decay, size_t step);
void oc_lora_zero_grad(oc_lora *l);
/* logits -> grad_scale*(softmax-onehot) in place; returns summed loss,
 * token count in *n_out */
float oc_ce_grad(float *logits, const uint32_t *targets, size_t count,
                 size_t vocab, float grad_scale, size_t *n_out);

/* die with message */
void oc_die(const char *fmt, ...);

#endif
