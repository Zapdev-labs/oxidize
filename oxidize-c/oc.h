/* oxidize-c: plain-C port of the oxidize-cpp dense Llama-family CPU inference
 * path (GGUF parse, quant dequant + fused int8 gemv, tokenizer, model, CLI).
 *
 * Scope (ponytail): dense Llama/Mistral/Qwen/TinyLlama decode only. No MoE,
 * MLA, CUDA, split GGUF, training, or server. Errors exit(1) with a message
 * (single-purpose CLI process). */
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
  OC_UNKNOWN,
} oc_quant;

#define QK 32          /* small-block element count */
#define QK_K 256       /* K-quant superblock element count */

size_t oc_block_values(oc_quant q);
size_t oc_block_bytes(oc_quant q);
size_t oc_row_bytes(oc_quant q, size_t cols);
oc_quant oc_from_ggml_type(uint32_t t);
const char *oc_quant_name(oc_quant q);

float oc_f16_to_f32(const uint8_t *le2);
void oc_dequant_row(oc_quant q, const uint8_t *src, float *dst, size_t n);

/* ---- activation int8 blocks (for fused integer gemv) ----
 * x is quantized once per matvec into blocks of 32: scale d and s = d*sum(q).
 * s lets K-quant mins fold in without a second pass. */
typedef struct { float d, s; int8_t q[QK]; } oc_q8blk;
void oc_quantize_act(const float *x, oc_q8blk *out, size_t n); /* n%32==0 */

/* ---- kernels ---- */
void oc_rms_norm(float *out, const float *x, const float *w, size_t n, float eps);
/* NeoX split-half partial rope with optional YaRN (yarn_factor > 1 enables). */
void oc_rope(float *vec, size_t head_dim, size_t n_heads, size_t pos,
             float theta, size_t rope_dim, float yarn_factor, float yarn_orig_ctx);
void oc_swiglu(float *gate, const float *up, size_t n);   /* gate = silu(gate)*up */
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
void oc_attention(float *out, const float *q, const float *k_cache,
                  const float *v_cache, size_t seq_len, size_t n_heads,
                  size_t kv_heads, size_t head_dim);

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
  uint64_t offset;       /* absolute into mmap */
} oc_tensor_info;

typedef struct {
  void *map; size_t map_size;
  const uint8_t *base;
  oc_meta *meta; size_t n_meta;
  oc_tensor_info *tensors; size_t n_tensors;
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
} oc_config;

typedef struct {
  float *attn_norm, *ffn_norm;
  oc_weight wq, wk, wv, wo, gate, up, down;
  float *q_bias, *k_bias, *v_bias;         /* NULL when absent */
  size_t q_bias_n, k_bias_n, v_bias_n;
  float *q_norm, *k_norm;                  /* per-head (Qwen3), NULL when absent */
  int kv_slot;                             /* index into KV cache (-1 for GDN) */

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
  size_t kv_stride;       /* kv_heads*head_dim */
  size_t kv_ctx;          /* allocated positions per layer */
  float *x;               /* hidden state [hidden] */
  oc_mtp *mtp;            /* NULL if the GGUF has no nextn block */
  bool gpu_active;        /* resident CUDA forward built (OC_CUDA builds only) */
} oc_model;

oc_model *oc_model_load(const char *path, size_t max_ctx); /* 0 = model default */
void oc_model_free(oc_model *m);
/* Run tokens[0..n) starting at absolute position start_pos; returns logits for
 * the last token in caller-provided buf [vocab]. */
void oc_forward(oc_model *m, const uint32_t *tokens, size_t n, size_t start_pos,
                float *logits);
/* Same, but writes logits for EVERY position into logits_all [n * vocab]
 * (batched lm_head — used by speculative verify). */
void oc_forward_all(oc_model *m, const uint32_t *tokens, size_t n,
                    size_t start_pos, float *logits_all);
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

/* die with message */
void oc_die(const char *fmt, ...);

#endif
