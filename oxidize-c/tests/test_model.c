/* Batched prefill == sequential decode.
 *
 * This is the acceptance test for oc_matmul and the two *_forward_batch paths,
 * and it is the ONLY thing standing between us and a silently wrong causal
 * mask. A batch that lets token i attend to token i+1 does not crash, does not
 * NaN, and produces confident, fluent, subtly wrong text. The reference
 * implementation this port came from has exactly that bug in its own prefill
 * kernel.
 *
 * Both models are built through the real gguf_parse + *_load path (an in-memory
 * GGUF, no file), so the geometry the test exercises is the geometry the loader
 * infers, not a hand-filled struct that can drift from it.
 *
 * The gemma4 fixture is deliberately hostile: a 4-slot sliding window with a
 * 17-token prompt, so the SWA ring wraps four times mid-batch and every token
 * needs a different window. That is the case where "write all the K/V, then
 * attend" silently eats the oldest position of every earlier token.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gguf.h"
#include "../src/model.h"
#include "../src/model_deepseek.h"
#include "../src/model_gemma4.h"
#include "../src/model_llama.h"
#include "../src/model_qwen36.h"
#include "../src/quant.h"
#include "../src/tensor.h"
#include "gguf_build.h"
#include "tests.h"

/* ---- comparison ------------------------------------------------------------ */

static void same_tol(const char* what, size_t n_tok, const float* got,
                     const float* want, size_t n, float tol) {
  for (size_t i = 0; i < n; ++i) {
    if (fabsf(got[i] - want[i]) <= tol * (1.0f + fabsf(want[i]))) continue;
    fprintf(stderr,
            "FAIL %s: batched prefill of %zu tokens disagrees with %zu sequential "
            "forwards at logit %zu: got %.9g want %.9g (tol %.1e)\n",
            what, n_tok, n_tok, i, (double)got[i], (double)want[i], (double)tol);
    exit(1);
  }
}
static void same(const char* what, size_t n_tok, const float* got, const float* want,
                 size_t n) {
  same_tol(what, n_tok, got, want, n, 5e-4f);
}

/* Greedy argmax, first-max-wins — identical tie-break to the model's internal
 * argmax_f32, so the plain-greedy reference and the speculative path agree. */
static int32_t amax(const float* v, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return (int32_t)best;
}

/* Prefix lengths to prefill in one batch and check against sequential decode.
 * 17 > ctx window (4) and > batch (4 or 16): the ring wraps inside the batch and
 * the batch itself is split into chunks. */
static const size_t SPLITS[] = {1, 2, 3, 5, 12, 17};
#define N_SPLITS (sizeof SPLITS / sizeof SPLITS[0])
#define N_TOK 17

/* ---- gemma4 ---------------------------------------------------------------- */

static uint8_t* gemma4_fixture(size_t* len) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = 64, NL = 3, NH = 4, HD = 16, KVH = 2, FF = 128, V = 32;
  rs = 7u;
  kv_str(&m, "general.architecture", "gemma4");
  kv_u32(&m, "gemma4.embedding_length", H);
  kv_u32(&m, "gemma4.block_count", NL);
  kv_u32(&m, "gemma4.attention.head_count", NH);
  kv_u32(&m, "gemma4.attention.head_count_kv", KVH);
  kv_u32(&m, "gemma4.feed_forward_length", FF);
  kv_u32(&m, "gemma4.context_length", 64);
  kv_u32(&m, "gemma4.attention.sliding_window", 4); /* tiny: the ring wraps */
  kv_u32(&m, "gemma4.attention.sliding_window_pattern", 3); /* l=0,1 SWA; l=2 global */
  kv_f32(&m, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
  kv_f32(&m, "gemma4.final_logit_softcapping", 30.0f);
  kv_f32(&m, "gemma4.attention.scale", 1.0f);
  kv_f32(&m, "gemma4.rope.freq_base", 1e6f);
  kv_f32(&m, "gemma4.rope.freq_base_swa", 1e4f);

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);
  tsr(&m, "rope_freqs.weight", 0, HD / 2, 1.0f, 0.2f); /* global-layer divisors */

  static char names[3][16][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NAME("attn_q.weight"), NH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_k.weight"), KVH * HD, H, 0.0f, 0.15f);
    /* Global layer omits attn_v (K and V share the projection) — exercises the
     * V = K branch of both forwards. */
    if (l != NL - 1) tsr(&m, NAME("attn_v.weight"), KVH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_output.weight"), H, NH * HD, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_gate.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_up.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_down.weight"), H, FF, 0.0f, 0.15f);
    tsr(&m, NAME("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("attn_q_norm.weight"), 0, HD, 1.0f, 0.1f);
    tsr(&m, NAME("attn_k_norm.weight"), 0, HD, 1.0f, 0.1f);
    tsr(&m, NAME("post_attention_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("post_ffw_norm.weight"), 0, H, 1.0f, 0.1f);
    if (l == 1) tsr(&m, NAME("layer_output_scale.weight"), 0, 1, 0.9f, 0.0f);
#undef NAME
  }
  return build(&m, len);
}

static void gemma4_reset(Gemma4Model* m) {
  for (size_t l = 0; l < m->n_layers; ++l) {
    Gemma4Layer* L = &m->layers[l];
    size_t kn = L->cache_cap * L->n_kv_heads * L->head_dim;
    size_t vn = L->cache_cap * L->n_kv_heads * L->v_head_dim;
    if (L->k_cache) memset(L->k_cache, 0, kn * sizeof(float));
    if (L->v_cache) memset(L->v_cache, 0, vn * sizeof(float));
    if (L->k_qcache) { /* byte size depends on the codec (q4 packs 2/byte) */
      size_t e = oc_kv_elem_bytes(m->kv_type);
      memset(L->k_qcache, 0, m->kv_type == OC_KV_Q4 ? kn / 2 : kn * e);
      memset(L->v_qcache, 0, m->kv_type == OC_KV_Q4 ? vn / 2 : vn * e);
    }
    if (L->k_qmeta) memset(L->k_qmeta, 0, L->cache_cap * L->n_kv_heads * 2 * sizeof(float));
    if (L->v_qmeta) memset(L->v_qmeta, 0, L->cache_cap * L->n_kv_heads * 2 * sizeof(float));
  }
  m->kv_len = 0;
}

static void gemma4_case(const uint8_t* blob, size_t len, OcKvType kt, size_t batch,
                        float tol) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Gemma4Model m;
  /* The bool param forces q4; f16/q8/f32 flow through the process-wide type. */
  oc_kv_set_type(kt);
  if (gemma4_load(&m, &g, 0, kt == OC_KV_Q4, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL gemma4_load: %s\n", err);
    exit(1);
  }
  CHECK(m.kv_type == kt); /* the fixture head dims are eligible for every type */
  m.batch = batch;
  CHECK(m.vocab == 32); /* the seq[] reference buffer */

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 7 + 3) % m.vocab);

  /* Reference: one token at a time, logits after every prefix. */
  static float seq[N_TOK][32];
  gemma4_reset(&m);
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lg = gemma4_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(seq[i], lg, m.vocab * sizeof(float));
  }

  char what[64];
  snprintf(what, sizeof what, "gemma4 kv=%s batch=%zu", oc_kv_type_name(kt), batch);
  for (size_t s = 0; s < N_SPLITS; ++s) {
    size_t n = SPLITS[s];
    gemma4_reset(&m);
    float* lg = gemma4_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same_tol(what, n, lg, seq[n - 1], m.vocab, tol);
  }

  /* Resumed batch: prefill part sequentially, then batch the rest. The ring is
   * already populated and wrapped, so the batch must read it for t < pos0. */
  gemma4_reset(&m);
  for (size_t i = 0; i < 6; ++i) CHECK(gemma4_forward(&m, ids[i], i, false) == NULL);
  float* lg = gemma4_forward_batch(&m, ids + 6, N_TOK - 6, 6, true);
  CHECK(lg != NULL);
  same_tol(what, N_TOK - 6, lg, seq[N_TOK - 1], m.vocab, tol);

  gemma4_free(&m);
}

/* ---- qwen36 ---------------------------------------------------------------- */

static uint8_t* qwen36_fixture(size_t* len) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = 64, NL = 4, NH = 4, KVH = 2, HD = 16, FF = 128, V = 32;
  const size_t DCONV = 4, DSTATE = 8, NK = 2, NV = 4, INNER = 32;
  const size_t HVD = INNER / NV, KEYD = DSTATE * NK, VALD = HVD * NV;
  const size_t CONVD = KEYD * 2 + VALD;
  rs = 99u;
  kv_str(&m, "general.architecture", "qwen35");
  kv_u32(&m, "qwen35.embedding_length", H);
  kv_u32(&m, "qwen35.block_count", NL);
  kv_u32(&m, "qwen35.nextn_predict_layers", 0);
  kv_u32(&m, "qwen35.attention.head_count", NH);
  kv_u32(&m, "qwen35.attention.head_count_kv", KVH);
  kv_u32(&m, "qwen35.attention.key_length", HD);
  kv_u32(&m, "qwen35.feed_forward_length", FF);
  kv_u32(&m, "qwen35.context_length", 64);
  kv_f32(&m, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
  kv_f32(&m, "qwen35.rope.freq_base", 1e7f);
  kv_u32(&m, "qwen35.rope.dimension_count", 8);
  kv_u32(&m, "qwen35.ssm.conv_kernel", DCONV);
  kv_u32(&m, "qwen35.ssm.state_size", DSTATE);
  kv_u32(&m, "qwen35.ssm.group_count", NK);
  kv_u32(&m, "qwen35.ssm.time_step_rank", NV);
  kv_u32(&m, "qwen35.ssm.inner_size", INNER);
  kv_u32(&m, "qwen35.full_attention_interval", 4); /* l=0,1,2 linear; l=3 full */

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);

  static char names[4][16][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NAME("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("post_attention_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_gate.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_up.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_down.weight"), H, FF, 0.0f, 0.15f);
    if ((l + 1) % 4 != 0) { /* gated DeltaNet */
      tsr(&m, NAME("attn_qkv.weight"), CONVD, H, 0.0f, 0.15f);
      tsr(&m, NAME("attn_gate.weight"), VALD, H, 0.0f, 0.15f);
      tsr(&m, NAME("ssm_out.weight"), H, VALD, 0.0f, 0.15f);
      tsr(&m, NAME("ssm_beta.weight"), NV, H, 0.0f, 0.15f);
      tsr(&m, NAME("ssm_alpha.weight"), NV, H, 0.0f, 0.15f);
      tsr(&m, NAME("ssm_dt.bias"), 0, NV, 0.0f, 0.3f);
      tsr(&m, NAME("ssm_a"), 0, NV, -1.5f, 0.5f); /* must stay negative: decay */
      tsr(&m, NAME("ssm_norm.weight"), 0, HVD, 1.0f, 0.1f);
      tsr(&m, NAME("ssm_conv1d.weight"), CONVD, DCONV, 0.0f, 0.4f);
    } else { /* gated full attention */
      tsr(&m, NAME("attn_q.weight"), NH * HD * 2, H, 0.0f, 0.15f);
      tsr(&m, NAME("attn_k.weight"), KVH * HD, H, 0.0f, 0.15f);
      tsr(&m, NAME("attn_v.weight"), KVH * HD, H, 0.0f, 0.15f);
      tsr(&m, NAME("attn_output.weight"), H, NH * HD, 0.0f, 0.15f);
      tsr(&m, NAME("attn_q_norm.weight"), 0, HD, 1.0f, 0.1f);
      tsr(&m, NAME("attn_k_norm.weight"), 0, HD, 1.0f, 0.1f);
    }
#undef NAME
  }
  return build(&m, len);
}

static void qwen36_reset(Qwen36Model* m) {
  for (size_t l = 0; l < m->n_layers; ++l) {
    Qwen36Layer* L = &m->layers[l];
    if (L->conv_state)
      memset(L->conv_state, 0, (m->d_conv - 1) * m->conv_dim * sizeof(float));
    if (L->S)
      memset(L->S, 0, m->n_v_heads * m->head_v_dim * m->d_state * sizeof(float));
    if (L->k_cache)
      memset(L->k_cache, 0, m->ctx * m->n_kv_heads * m->head_dim * sizeof(float));
    if (L->v_cache)
      memset(L->v_cache, 0, m->ctx * m->n_kv_heads * m->head_dim * sizeof(float));
  }
}

static void qwen36_case(const uint8_t* blob, size_t len, size_t batch) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Qwen36Model m;
  if (qwen36_load(&m, &g, 0, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL qwen36_load: %s\n", err);
    exit(1);
  }
  m.batch = batch;

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 5 + 1) % m.vocab);

  static float seq[N_TOK][32];
  qwen36_reset(&m);
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lg = qwen36_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(seq[i], lg, m.vocab * sizeof(float));
  }

  char what[48];
  snprintf(what, sizeof what, "qwen36 batch=%zu", batch);
  for (size_t s = 0; s < N_SPLITS; ++s) {
    size_t n = SPLITS[s];
    qwen36_reset(&m);
    float* lg = qwen36_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same(what, n, lg, seq[n - 1], m.vocab);
  }

  /* Resumed batch: the conv window and the delta state S carry over from the
   * sequential part, so the scan must pick them up exactly where it left off. */
  qwen36_reset(&m);
  for (size_t i = 0; i < 6; ++i) CHECK(qwen36_forward(&m, ids[i], i, false) == NULL);
  float* lg = qwen36_forward_batch(&m, ids + 6, N_TOK - 6, 6, true);
  CHECK(lg != NULL);
  same(what, N_TOK - 6, lg, seq[N_TOK - 1], m.vocab);

  qwen36_free(&m);
}

/* Fixture WITH a NextN/MTP draft block: block_count=5, nextn=1 (so n_layers=4:
 * layers 0-2 gated-DeltaNet, layer 3 gated full attention) and block 4 is the
 * MTP head — the nextn.* fusion (eh_proj/enorm/hnorm) + shared head plus a
 * gated full-attention layer. Same geometry as qwen36_fixture. */
static uint8_t* qwen36_mtp_fixture(size_t* len) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = 64, NL = 5, NH = 4, KVH = 2, HD = 16, FF = 128, V = 32;
  const size_t DCONV = 4, DSTATE = 8, NK = 2, NV = 4, INNER = 32;
  const size_t HVD = INNER / NV, KEYD = DSTATE * NK, VALD = HVD * NV;
  const size_t CONVD = KEYD * 2 + VALD;
  rs = 77u;
  kv_str(&m, "general.architecture", "qwen35");
  kv_u32(&m, "qwen35.embedding_length", H);
  kv_u32(&m, "qwen35.block_count", NL);         /* 5 blocks... */
  kv_u32(&m, "qwen35.nextn_predict_layers", 1); /* ...last is the MTP head */
  kv_u32(&m, "qwen35.attention.head_count", NH);
  kv_u32(&m, "qwen35.attention.head_count_kv", KVH);
  kv_u32(&m, "qwen35.attention.key_length", HD);
  kv_u32(&m, "qwen35.feed_forward_length", FF);
  kv_u32(&m, "qwen35.context_length", 64);
  kv_f32(&m, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
  kv_f32(&m, "qwen35.rope.freq_base", 1e7f);
  kv_u32(&m, "qwen35.rope.dimension_count", 8);
  kv_u32(&m, "qwen35.ssm.conv_kernel", DCONV);
  kv_u32(&m, "qwen35.ssm.state_size", DSTATE);
  kv_u32(&m, "qwen35.ssm.group_count", NK);
  kv_u32(&m, "qwen35.ssm.time_step_rank", NV);
  kv_u32(&m, "qwen35.ssm.inner_size", INNER);
  kv_u32(&m, "qwen35.full_attention_interval", 4);

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);

  static char names[5][20][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NM(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NM("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NM("post_attention_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NM("ffn_gate.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NM("ffn_up.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NM("ffn_down.weight"), H, FF, 0.0f, 0.15f);
    int is_linear = (l < NL - 1) && ((l + 1) % 4 != 0);
    if (is_linear) {
      tsr(&m, NM("attn_qkv.weight"), CONVD, H, 0.0f, 0.15f);
      tsr(&m, NM("attn_gate.weight"), VALD, H, 0.0f, 0.15f);
      tsr(&m, NM("ssm_out.weight"), H, VALD, 0.0f, 0.15f);
      tsr(&m, NM("ssm_beta.weight"), NV, H, 0.0f, 0.15f);
      tsr(&m, NM("ssm_alpha.weight"), NV, H, 0.0f, 0.15f);
      tsr(&m, NM("ssm_dt.bias"), 0, NV, 0.0f, 0.3f);
      tsr(&m, NM("ssm_a"), 0, NV, -1.5f, 0.5f);
      tsr(&m, NM("ssm_norm.weight"), 0, HVD, 1.0f, 0.1f);
      tsr(&m, NM("ssm_conv1d.weight"), CONVD, DCONV, 0.0f, 0.4f);
    } else { /* full attention: main layer 3 AND the MTP block 4 */
      tsr(&m, NM("attn_q.weight"), NH * HD * 2, H, 0.0f, 0.15f);
      tsr(&m, NM("attn_k.weight"), KVH * HD, H, 0.0f, 0.15f);
      tsr(&m, NM("attn_v.weight"), KVH * HD, H, 0.0f, 0.15f);
      tsr(&m, NM("attn_output.weight"), H, NH * HD, 0.0f, 0.15f);
      tsr(&m, NM("attn_q_norm.weight"), 0, HD, 1.0f, 0.1f);
      tsr(&m, NM("attn_k_norm.weight"), 0, HD, 1.0f, 0.1f);
    }
    if (l == NL - 1) { /* nextn fusion + shared head on the MTP block */
      tsr(&m, NM("nextn.eh_proj.weight"), H, 2 * H, 0.0f, 0.1f);
      tsr(&m, NM("nextn.enorm.weight"), 0, H, 1.0f, 0.1f);
      tsr(&m, NM("nextn.hnorm.weight"), 0, H, 1.0f, 0.1f);
      tsr(&m, NM("nextn.shared_head_norm.weight"), 0, H, 1.0f, 0.1f);
      tsr(&m, NM("nextn.shared_head_head.weight"), V, H, 0.0f, 0.2f);
    }
#undef NM
  }
  return build(&m, len);
}

/* Speculative decode with the native MTP draft head must be bit-exact to plain
 * greedy decode, for any draft length and any acceptance pattern. This is the
 * P16 acceptance test: the target verification makes the emitted tokens the
 * target's own argmax, so the draft only changes SPEED, never the output — and
 * the DeltaNet recurrent state has to snapshot/restore exactly across a
 * partially-accepted step or the two paths silently diverge. */
static void qwen36_spec_case(void) {
  size_t len = 0;
  uint8_t* blob = qwen36_mtp_fixture(&len);
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Qwen36Model m;
  if (qwen36_load(&m, &g, 0, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL qwen36_load (mtp): %s\n", err);
    exit(1);
  }
  CHECK(m.has_mtp);       /* the nextn/MTP block was loaded, not discarded */
  CHECK(m.n_layers == 4); /* block_count 5 - nextn 1 */

  const size_t NP = 5, NG = 14;
  int32_t prompt[5];
  for (size_t i = 0; i < NP; ++i) prompt[i] = (int32_t)((i * 5 + 1) % m.vocab);
  size_t pos;
  float* lg = NULL;

  /* plain greedy reference: target argmax at each step */
  int32_t plain[14];
  qwen36_reset(&m);
  for (size_t i = 0; i < NP; ++i) {
    lg = qwen36_forward(&m, prompt[i], i, true);
    CHECK(lg);
  }
  pos = NP;
  for (size_t i = 0; i < NG; ++i) {
    plain[i] = amax(lg, m.vocab);
    lg = qwen36_forward(&m, plain[i], pos++, true);
    CHECK(lg);
  }

  /* speculative greedy for draft-tokens 1..4: identical token ids */
  for (size_t k = 1; k <= 4; ++k) {
    qwen36_reset(&m);
    for (size_t i = 0; i < NP; ++i) lg = qwen36_forward(&m, prompt[i], i, true);
    pos = NP;
    int32_t seed = prompt[NP - 1];
    int32_t got[14];
    size_t ng = 0, steps = 0, acc_total = 0;
    while (ng < NG) {
      int32_t draft[8], out[8];
      qwen36_mtp_draft(&m, seed, draft, k);
      size_t acc = 0;
      size_t l = qwen36_spec_step(&m, draft, k, pos, out, &acc);
      CHECK(l >= 1 && l <= k + 1 && acc <= k);
      for (size_t j = 0; j < l && ng < NG; ++j) got[ng++] = out[j];
      pos += l;
      seed = out[l - 1];
      steps++;
      acc_total += acc;
    }
    for (size_t i = 0; i < NG; ++i)
      if (got[i] != plain[i]) {
        fprintf(stderr, "FAIL qwen36 spec k=%zu: token %zu got %d want %d\n", k, i,
                got[i], plain[i]);
        exit(1);
      }
    /* Honest measurement: the fixture has RANDOM weights, so the draft head is
     * near chance (vocab 32) and acceptance is low — the point of the test is
     * bit-exactness, not speed. A real trained MTP head is where the accepted/
     * step climbs and the wall-clock speedup appears. */
    fprintf(stderr, "  spec k=%zu: %zu steps, %.2f draft accepted/step "
            "(%zu tokens over %zu target batches)\n",
            k, steps, steps ? (double)acc_total / (double)steps : 0.0, ng, steps);
  }

  /* Hand-crafted acceptance: force full / partial / zero accept with known
   * drafts and prove the committed state stays exactly on the plain trajectory.
   * The partial/zero steps run the throwaway verify batch (mutating the DeltaNet
   * state), then roll it back — if the snapshot/restore is wrong the tail below
   * diverges from plain[]. */
  qwen36_reset(&m);
  for (size_t i = 0; i < NP; ++i) lg = qwen36_forward(&m, prompt[i], i, true);
  pos = NP;
  size_t gi = 0, acc;
  int32_t out[8];
  { /* full accept: 2 correct drafts -> 2 accepted + 1 bonus continuation */
    int32_t d[2] = {plain[0], plain[1]};
    size_t l = qwen36_spec_step(&m, d, 2, pos, out, &acc);
    CHECK(acc == 2 && l == 3);
    for (size_t j = 0; j < l; ++j) CHECK(out[j] == plain[gi + j]);
    gi += l;
    pos += l;
  }
  { /* partial accept: first draft right, second wrong -> 1 accepted + correction */
    int32_t d[3] = {plain[gi], (int32_t)((plain[gi + 1] + 1) % (int)m.vocab), 3};
    size_t l = qwen36_spec_step(&m, d, 3, pos, out, &acc);
    CHECK(acc == 1 && l == 2);
    CHECK(out[0] == plain[gi] && out[1] == plain[gi + 1]);
    gi += l;
    pos += l;
  }
  { /* SPEC-EQUIV PROBE: accept 2 of 3 then reject -> 2 accepted + correction.
       Verify advances DeltaNet state through a WRONG 3rd draft; restore must undo
       exactly that or the tail diverges from plain. */
    int32_t d[3] = {plain[gi], plain[gi + 1],
                    (int32_t)((plain[gi + 2] + 1) % (int)m.vocab)};
    size_t l = qwen36_spec_step(&m, d, 3, pos, out, &acc);
    CHECK(acc == 2 && l == 3);
    CHECK(out[0] == plain[gi] && out[1] == plain[gi + 1] &&
          out[2] == plain[gi + 2]);
    gi += l;
    pos += l;
  }
  { /* zero accept: first draft wrong -> just the target's correction */
    int32_t d[3] = {(int32_t)((plain[gi] + 1) % (int)m.vocab), 0, 1};
    size_t l = qwen36_spec_step(&m, d, 3, pos, out, &acc);
    CHECK(acc == 0 && l == 1);
    CHECK(out[0] == plain[gi]);
    gi += l;
    pos += l;
  }
  while (gi < NG) { /* the state survived every rollback: tail matches plain */
    CHECK(amax(m.logits, m.vocab) == plain[gi]);
    lg = qwen36_forward(&m, plain[gi], pos++, true);
    gi++;
  }

  /* Direct DeltaNet snapshot/restore: forward, snapshot, forward one, restore,
   * re-forward the same token -> bit-identical logits. A positional KV rewind
   * alone could not do this (the recurrent state has no per-position history). */
  qwen36_reset(&m);
  for (size_t i = 0; i < NP; ++i) lg = qwen36_forward(&m, prompt[i], i, true);
  qwen36_state_snapshot(&m);
  static float ref[32];
  lg = qwen36_forward(&m, plain[0], NP, true);
  memcpy(ref, lg, m.vocab * sizeof(float));
  qwen36_state_restore(&m);
  lg = qwen36_forward(&m, plain[0], NP, true);
  same_tol("qwen36 deltanet snapshot/restore", 1, lg, ref, m.vocab, 1e-6f);

  /* Over-long-draft guard: k > mtp.cap (== batch_cap) must fill only the mini-KV
   * it owns and set the surplus draft[] slots to -1, never write past the cache.
   * Exercised at the REAL batch_cap (default 32) so the standard suite covers the
   * clamp, not just a tiny OC_BATCH; asan reports a regression as a heap overflow. */
  {
    size_t cap = m.batch_cap, kk = cap + 3;
    int32_t dd[520]; /* batch_cap is clamped to <= 512 at load, so cap+3 fits */
    CHECK(kk <= sizeof dd / sizeof *dd);
    for (size_t i = 0; i < kk; ++i) dd[i] = -7;
    qwen36_mtp_draft(&m, plain[0], dd, kk);
    for (size_t i = 0; i < cap; ++i) CHECK(dd[i] >= 0 && dd[i] < (int32_t)m.vocab);
    for (size_t i = cap; i < kk; ++i) CHECK(dd[i] == -1);
  }

  qwen36_free(&m);
  free(blob);
}

/* ---- llama (generic dense) ------------------------------------------------- */

/* Which optional pieces the fixture carries — the loader dispatches on the
 * tensors present, so these are the axes worth exercising. */
typedef enum {
  LLAMA_PLAIN,      /* untied output head, no bias, no qk-norm (llama/mistral) */
  LLAMA_BIAS_TIED,  /* tied embeddings + q/k/v bias (Qwen2-0.5B shape) */
  LLAMA_QKNORM,     /* untied + per-head q/k RMSNorm + independent, partially
                       rotated head_dim (Qwen3 shape) */
} LlamaVariant;

/* GGUF prefixes model metadata with the arch string, so the loader reads
 * "{arch}.embedding_length" etc. — the fixture must key the same way. */
static void kv_u32p(GgufB* m, const char* arch, const char* suffix, uint32_t v) {
  char key[160];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  kv_u32(m, key, v);
}
static void kv_f32p(GgufB* m, const char* arch, const char* suffix, float v) {
  char key[160];
  snprintf(key, sizeof key, "%s.%s", arch, suffix);
  kv_f32(m, key, v);
}

static uint8_t* llama_fixture(size_t* len, LlamaVariant var, const char* arch) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = 64, NL = 3, NH = 4, KVH = 2, FF = 128, V = 32;
  /* Qwen3 gives head_dim its own key_length (here 24 != H/NH == 16) and rotates
   * only rope_dim of it; the other variants keep the square head_dim. */
  const size_t HD = var == LLAMA_QKNORM ? 24 : 16;
  rs = 4321u;
  kv_str(&m, "general.architecture", arch);
  kv_u32p(&m, arch, "embedding_length", H);
  kv_u32p(&m, arch, "block_count", NL);
  kv_u32p(&m, arch, "attention.head_count", NH);
  kv_u32p(&m, arch, "attention.head_count_kv", KVH);
  kv_u32p(&m, arch, "feed_forward_length", FF);
  kv_u32p(&m, arch, "context_length", 64);
  kv_f32p(&m, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
  kv_f32p(&m, arch, "rope.freq_base", 1e4f);
  if (var == LLAMA_QKNORM) {
    kv_u32p(&m, arch, "attention.key_length", HD);
    kv_u32p(&m, arch, "rope.dimension_count", 16); /* partial: 16 < 24 */
  } else {
    kv_u32p(&m, arch, "rope.dimension_count", HD); /* full rotary */
  }

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  if (var != LLAMA_BIAS_TIED) tsr(&m, "output.weight", V, H, 0.0f, 0.2f); /* untied */
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);

  static char names[3][20][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NAME("attn_q.weight"), NH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_k.weight"), KVH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_v.weight"), KVH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_output.weight"), H, NH * HD, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_gate.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_up.weight"), FF, H, 0.0f, 0.15f);
    tsr(&m, NAME("ffn_down.weight"), H, FF, 0.0f, 0.15f);
    tsr(&m, NAME("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_norm.weight"), 0, H, 1.0f, 0.1f);
    if (var == LLAMA_BIAS_TIED) {
      tsr(&m, NAME("attn_q.bias"), 0, NH * HD, 0.0f, 0.3f);
      tsr(&m, NAME("attn_k.bias"), 0, KVH * HD, 0.0f, 0.3f);
      tsr(&m, NAME("attn_v.bias"), 0, KVH * HD, 0.0f, 0.3f);
    }
    if (var == LLAMA_QKNORM) {
      tsr(&m, NAME("attn_q_norm.weight"), 0, HD, 1.0f, 0.1f);
      tsr(&m, NAME("attn_k_norm.weight"), 0, HD, 1.0f, 0.1f);
    }
#undef NAME
  }
  return build(&m, len);
}

static void llama_reset(LlamaModel* m) {
  size_t kv = m->ctx * m->n_kv_heads * m->head_dim;
  size_t kb = kv * oc_kv_elem_bytes(m->kv_type);
  size_t sb = m->ctx * m->n_kv_heads * sizeof(float);
  for (size_t l = 0; l < m->n_layers; ++l) {
    memset(m->layers[l].k_cache, 0, kb);
    memset(m->layers[l].v_cache, 0, kb);
    if (m->layers[l].k_scale) memset(m->layers[l].k_scale, 0, sb);
    if (m->layers[l].v_scale) memset(m->layers[l].v_scale, 0, sb);
  }
  m->kv_len = 0;
}

static void llama_case(const uint8_t* blob, size_t len, LlamaVariant var, size_t batch,
                       OcKvType kt, float tol) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  LlamaModel m;
  oc_kv_set_type(kt);
  if (llama_load(&m, &g, 0, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL llama_load: %s\n", err);
    exit(1);
  }
  CHECK(m.kv_type == kt);
  m.batch = batch;
  CHECK(m.vocab == 32);

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 5 + 2) % m.vocab);

  static float seq[N_TOK][32];
  llama_reset(&m);
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lg = llama_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(seq[i], lg, m.vocab * sizeof(float));
  }

  char what[64];
  snprintf(what, sizeof what, "llama var=%d kv=%s batch=%zu", (int)var,
           oc_kv_type_name(kt), batch);
  for (size_t s = 0; s < N_SPLITS; ++s) {
    size_t n = SPLITS[s];
    llama_reset(&m);
    float* lg = llama_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same_tol(what, n, lg, seq[n - 1], m.vocab, tol);
  }

  /* Resumed batch: prefill part sequentially, then batch the rest reading the
   * already-populated KV cache for t < pos0. */
  llama_reset(&m);
  for (size_t i = 0; i < 6; ++i) CHECK(llama_forward(&m, ids[i], i, false) == NULL);
  float* lg = llama_forward_batch(&m, ids + 6, N_TOK - 6, 6, true);
  CHECK(lg != NULL);
  same_tol(what, N_TOK - 6, lg, seq[N_TOK - 1], m.vocab, tol);

  llama_free(&m);
}

/* The dispatch routes by arch string: known dense arches -> llama family;
 * unknown -> a clean error rather than a silent fall-through. */
static void llama_dispatch_smoke(void) {
  const char* ok[] = {"llama", "mistral", "qwen2", "qwen3", "yi", "phi3", NULL};
  for (size_t i = 0; ok[i]; ++i) {
    size_t len = 0;
    uint8_t* blob = llama_fixture(&len, LLAMA_PLAIN, ok[i]);
    char err[256] = {0};
    GgufFile g;
    CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
    Model M;
    if (model_load(&M, &g, 0, false, err, sizeof err) != 0) {
      fprintf(stderr, "FAIL dispatch arch=%s: %s\n", ok[i], err);
      exit(1);
    }
    CHECK(M.family == MODEL_LLAMA);
    /* The rope-mode routing: llama/mistral/yi ship llama.cpp-permuted q/k and
     * MUST use ggml NORMAL rope; qwen2/qwen3/phi3 use NeoX. Applying NeoX to a
     * permuted-q/k llama GGUF is fluent garbage, not a crash. */
    bool want_norm = strcmp(ok[i], "llama") == 0 ||
                     strcmp(ok[i], "mistral") == 0 || strcmp(ok[i], "yi") == 0;
    CHECK(((LlamaModel*)M.handle)->rope_norm == want_norm);
    model_free(&M);
    free(blob);
  }
  /* An arch nobody routes must fail loudly, not fall through to a loader. */
  size_t len = 0;
  uint8_t* blob = llama_fixture(&len, LLAMA_PLAIN, "bert");
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Model M;
  CHECK(model_load(&M, &g, 0, false, err, sizeof err) != 0);
  CHECK(strstr(err, "unsupported architecture") != NULL);
  gguf_close(&g);
  free(blob);
}

/* RoPE-mode correctness — the kernel half of the llama.cpp-permuted-q/k fix.
 * Ground truth is ggml's two rope conventions, not oxidize self-consistency:
 *   (1) oc_rope_normal must match an independent adjacent-pair (NORMAL) ref.
 *   (2) llama.cpp permutes q/k so that NORMAL on the permuted weights equals
 *       NeoX on the natural layout: oc_rope_normal(permute(u)) reordered ==
 *       oc_rope(u). This is exactly why llama/mistral/yi need NORMAL, not NeoX. */
static void check_rope_modes(void) {
  enum { D = 8 };
  const float theta = 1e4f;
  const size_t pos = 3;
  const float u[D] = {0.5f, -1.2f, 0.3f, 0.9f, -0.7f, 1.1f, -0.4f, 0.2f};

  /* (1) oc_rope_normal vs a hand-written adjacent-pair reference. */
  float got[D], ref[D];
  memcpy(got, u, sizeof got);
  oc_rope_normal(got, D, 1, pos, theta, 0);
  for (int i = 0; i < D / 2; ++i) {
    float freq = powf(theta, -2.0f * (float)i / (float)D);
    float a = (float)pos * freq, c = cosf(a), s = sinf(a);
    ref[2 * i] = u[2 * i] * c - u[2 * i + 1] * s;
    ref[2 * i + 1] = u[2 * i] * s + u[2 * i + 1] * c;
  }
  for (int i = 0; i < D; ++i) CHECK(fabsf(got[i] - ref[i]) < 1e-5f);

  /* (2) permute identity: perm[2i]=u[i], perm[2i+1]=u[D/2+i] (llama.cpp permute);
   * NORMAL(perm) in interleaved layout must equal NeoX(u) in split-half layout. */
  float perm[D], neox[D];
  for (int i = 0; i < D / 2; ++i) {
    perm[2 * i] = u[i];
    perm[2 * i + 1] = u[D / 2 + i];
  }
  oc_rope_normal(perm, D, 1, pos, theta, 0);
  memcpy(neox, u, sizeof neox);
  oc_rope(neox, D, 1, pos, theta, 0, NULL);
  for (int i = 0; i < D / 2; ++i) {
    CHECK(fabsf(perm[2 * i] - neox[i]) < 1e-5f);
    CHECK(fabsf(perm[2 * i + 1] - neox[D / 2 + i]) < 1e-5f);
  }
  printf("ok rope modes: oc_rope_normal == ggml-NORMAL ref; "
         "NORMAL(permute)==NeoX (llama/mistral/yi permuted-q/k path)\n");
}

/* ---- MoE (Mixtral / Qwen-MoE / DeepSeek / OLMoE / gpt-oss family) ----------
 *
 * The routed FFN is the one part the batched==sequential test above CANNOT
 * validate on its own: both llama_forward and llama_forward_batch call the same
 * llama_moe_ffn per token, so a wrong softmax, a wrong top-k or a wrong expert
 * byte-stride is identical in both and cancels in the comparison. So this block
 * adds an INDEPENDENT reference — hand-rolled routing + a from-first-principles
 * expert stride — and asserts the model's FFN equals it. The expert stacks are
 * Q8_0, not f32, so a byte-stride computed with the wrong row size reads a
 * different expert (or misaligned bytes) and the reference diverges loudly. */

/* Geometry shared by the fixture and the reference (hidden 64, 3 MoE layers,
 * 4 experts top-2, expert_inter 128 != hidden so a hidden/inter swap shows,
 * shared expert width 96 != both). */
#define MOE_H 64
#define MOE_NL 3
#define MOE_NE 4
#define MOE_K 2
#define MOE_EI 128
#define MOE_SI 96

static uint8_t* moe_fixture(size_t* len, bool shared) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = MOE_H, NL = MOE_NL, NH = 4, KVH = 2, HD = 16, V = 32;
  const size_t NE = MOE_NE, K = MOE_K, EI = MOE_EI, SI = MOE_SI;
  const char* arch = "qwen3moe"; /* NeoX rope; reached via llama_load directly */
  rs = 2024u;
  kv_str(&m, "general.architecture", arch);
  kv_u32p(&m, arch, "embedding_length", H);
  kv_u32p(&m, arch, "block_count", NL);
  kv_u32p(&m, arch, "attention.head_count", NH);
  kv_u32p(&m, arch, "attention.head_count_kv", KVH);
  kv_u32p(&m, arch, "feed_forward_length", EI); /* all-MoE: only sizes dense scratch */
  kv_u32p(&m, arch, "context_length", 64);
  kv_f32p(&m, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
  kv_f32p(&m, arch, "rope.freq_base", 1e4f);
  kv_u32p(&m, arch, "rope.dimension_count", HD); /* full NeoX rotary */
  kv_u32p(&m, arch, "expert_count", NE);
  kv_u32p(&m, arch, "expert_used_count", K);
  kv_u32p(&m, arch, "expert_weights_norm", 1); /* norm_topk on; the check flips it */
  kv_f32p(&m, arch, "expert_weights_scale", 1.0f);
  if (shared) kv_u32p(&m, arch, "expert_shared_count", 1);

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output.weight", V, H, 0.0f, 0.2f); /* untied */
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);

  static char names[MOE_NL][24][56];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[56] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 56, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NAME("attn_q.weight"), NH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_k.weight"), KVH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_v.weight"), KVH * HD, H, 0.0f, 0.15f);
    tsr(&m, NAME("attn_output.weight"), H, NH * HD, 0.0f, 0.15f);
    tsr(&m, NAME("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_norm.weight"), 0, H, 1.0f, 0.1f);
    /* Router f32 [H, NE] (spread wide so the top-k boundary is unambiguous);
     * gate/up expert stacks [H, EI, NE]; down [EI, H, NE]; all Q8_0. */
    tsr(&m, NAME("ffn_gate_inp.weight"), NE, H, 0.0f, 0.5f);
    tsr_q8(&m, NAME("ffn_gate_exps.weight"), NE, EI, H, 0.15f);
    tsr_q8(&m, NAME("ffn_up_exps.weight"), NE, EI, H, 0.15f);
    tsr_q8(&m, NAME("ffn_down_exps.weight"), NE, H, EI, 0.15f);
    if (shared) {
      /* Shared expert (Qwen2-MoE/DeepSeek): a 2-D SwiGLU triple, plus the
       * optional sigmoid gate ffn_gate_inp_shexp (exercises that branch). */
      tsr_q8(&m, NAME("ffn_gate_shexp.weight"), 0, SI, H, 0.15f);
      tsr_q8(&m, NAME("ffn_up_shexp.weight"), 0, SI, H, 0.15f);
      tsr_q8(&m, NAME("ffn_down_shexp.weight"), 0, H, SI, 0.15f);
      tsr(&m, NAME("ffn_gate_inp_shexp.weight"), 0, H, 0.0f, 0.5f);
    }
#undef NAME
  }
  return build(&m, len);
}

/* Batched-prefill == sequential-decode for the MoE stack: top-k routing on every
 * token, a chunk boundary at batch=4 (n=5,12,17 span it), and a resumed batch. */
static void moe_case(const uint8_t* blob, size_t len, bool shared, size_t batch) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  LlamaModel m;
  oc_kv_set_type(OC_KV_F32); /* the MoE reference bar is f32-exact */
  if (llama_load(&m, &g, 0, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL moe llama_load: %s\n", err);
    exit(1);
  }
  m.batch = batch;
  CHECK(m.vocab == 32);
  CHECK(m.has_moe && m.n_experts == MOE_NE && m.n_experts_used == MOE_K &&
        m.expert_inter == MOE_EI && m.n_layers == MOE_NL);
  const LlamaLayer* last = &m.layers[m.n_layers - 1];
  CHECK(last->is_moe && (last->ffn_gate_shexp != NULL) == shared);
  CHECK(shared ? m.shexp_inter == MOE_SI : m.shexp_inter == 0);

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 3 + 4) % m.vocab);

  static float seq[N_TOK][32];
  llama_reset(&m);
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lg = llama_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(seq[i], lg, m.vocab * sizeof(float));
  }

  char what[64];
  snprintf(what, sizeof what, "moe shared=%d batch=%zu", (int)shared, batch);
  for (size_t s = 0; s < N_SPLITS; ++s) {
    size_t n = SPLITS[s];
    llama_reset(&m);
    float* lg = llama_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same(what, n, lg, seq[n - 1], m.vocab);
  }
  llama_reset(&m);
  for (size_t i = 0; i < 6; ++i) CHECK(llama_forward(&m, ids[i], i, false) == NULL);
  float* lg = llama_forward_batch(&m, ids + 6, N_TOK - 6, 6, true);
  CHECK(lg != NULL);
  same(what, N_TOK - 6, lg, seq[N_TOK - 1], m.vocab);

  llama_free(&m);
}

/* y[r] = dot(dequant(expert e's row r over `cols` values), x), r in [0, rows).
 * Expert e's 2-D slice begins at e*rows*row_bytes(type, cols) and row r follows
 * at +r*row_bytes — the ggml stacked-expert layout, derived here independently
 * so that a model computing the stride differently reads different bytes. */
static void ref_expert_matvec(const GgufTensorInfo* T, size_t e, size_t rows,
                              size_t cols, const float* x, float* y) {
  size_t rb = oc_row_bytes(T->ggml_type, cols);
  CHECK(rb > 0 && cols <= MOE_EI);
  const uint8_t* base = T->data + (size_t)e * rows * rb;
  float row[MOE_EI];
  for (size_t r = 0; r < rows; ++r) {
    CHECK(oc_dequant_row(T->ggml_type, base + r * rb, row, cols) == 0);
    y[r] = oc_dot_f32(row, x, cols);
  }
}

/* Independent MoE FFN reference: hand-rolled softmax over ALL experts -> top-k
 * -> renorm(+scale) -> weighted SwiGLU sum, plus the optional (sigmoid-gated)
 * shared expert. Writes ref[H]; returns the k-th-selected minus best-unselected
 * probability gap so the caller can reject a near-tie (where the model's and the
 * reference's ~1e-6-different logits could pick a different expert). */
static float moe_ref(const LlamaModel* m, const LlamaLayer* L, const float* x,
                     bool norm_topk, float scale, float* ref) {
  const size_t H = m->hidden, NE = m->n_experts, K = m->n_experts_used, EI = m->expert_inter;
  CHECK(NE <= 16 && K <= 16 && EI <= MOE_EI && H <= MOE_H);

  const float* rw = (const float*)L->ffn_gate_inp->data; /* f32 [H, NE], row e = e*H */
  float logit[16], prob[16];
  for (size_t e = 0; e < NE; ++e) logit[e] = oc_dot_f32(rw + e * H, x, H);
  float mx = logit[0];
  for (size_t e = 1; e < NE; ++e) if (logit[e] > mx) mx = logit[e];
  float s = 0.0f;
  for (size_t e = 0; e < NE; ++e) { prob[e] = expf(logit[e] - mx); s += prob[e]; }
  for (size_t e = 0; e < NE; ++e) prob[e] /= s;

  int sel[16];
  float w[16];
  bool used[16] = {false};
  float wsum = 0.0f;
  for (size_t j = 0; j < K; ++j) {
    int best = -1;
    float bv = -1.0f;
    for (size_t e = 0; e < NE; ++e)
      if (!used[e] && prob[e] > bv) { bv = prob[e]; best = (int)e; }
    used[best] = true;
    sel[j] = best;
    w[j] = prob[best];
    wsum += prob[best];
  }
  float top_unsel = -1.0f;
  for (size_t e = 0; e < NE; ++e)
    if (!used[e] && prob[e] > top_unsel) top_unsel = prob[e];
  float gap = w[K - 1] - top_unsel;

  float norm = (norm_topk && wsum > 0.0f) ? wsum : 1.0f;
  for (size_t j = 0; j < K; ++j) w[j] = scale * w[j] / norm;

  for (size_t i = 0; i < H; ++i) ref[i] = 0.0f;
  for (size_t j = 0; j < K; ++j) {
    size_t e = (size_t)sel[j];
    float g[MOE_EI], u[MOE_EI], d[MOE_H];
    ref_expert_matvec(L->ffn_gate_exps, e, EI, H, x, g);
    ref_expert_matvec(L->ffn_up_exps, e, EI, H, x, u);
    for (size_t r = 0; r < EI; ++r) g[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
    ref_expert_matvec(L->ffn_down_exps, e, H, EI, g, d);
    for (size_t i = 0; i < H; ++i) ref[i] += w[j] * d[i];
  }
  if (L->ffn_gate_shexp) {
    float sg = 1.0f;
    if (L->ffn_gate_inp_shexp)
      sg = 1.0f / (1.0f + expf(-oc_dot_f32(L->ffn_gate_inp_shexp, x, H)));
    const size_t SI = m->shexp_inter;
    float g[MOE_EI], u[MOE_EI], d[MOE_H];
    ref_expert_matvec(L->ffn_gate_shexp, 0, SI, H, x, g);
    ref_expert_matvec(L->ffn_up_shexp, 0, SI, H, x, u);
    for (size_t r = 0; r < SI; ++r) g[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
    ref_expert_matvec(L->ffn_down_shexp, 0, H, SI, g, d);
    for (size_t i = 0; i < H; ++i) ref[i] += sg * d[i];
  }
  return gap;
}

/* Router in isolation: llama_moe_route on fixed logits must be the FULL softmax
 * over all experts, then the top-k highest, then renorm(+scale) — the three
 * silent failure modes (softmax after top-k / wrong k / no renorm). */
static void moe_route_unit(void) {
  const float logits[MOE_NE] = {2.0f, 1.0f, 3.0f, 0.5f}; /* argmax order: 2,0,1,3 */
  for (int tv = 0; tv < 2; ++tv) {
    for (int sv = 0; sv < 2; ++sv) {
      bool norm = tv == 1;
      float scale = sv == 1 ? 1.7f : 1.0f;
      float p[MOE_NE], s = 0.0f, mx = 3.0f;
      for (int e = 0; e < MOE_NE; ++e) { p[e] = expf(logits[e] - mx); s += p[e]; }
      for (int e = 0; e < MOE_NE; ++e) p[e] /= s;
      int esel[MOE_K] = {2, 0};
      float ew[MOE_K] = {p[2], p[0]};
      float nrm = norm ? ew[0] + ew[1] : 1.0f;
      for (int j = 0; j < MOE_K; ++j) ew[j] = scale * ew[j] / nrm;

      float probs[MOE_NE], w[MOE_K];
      int sel[MOE_K];
      llama_moe_route(logits, MOE_NE, MOE_K, norm, scale, probs, sel, w);
      CHECK(sel[0] == esel[0] && sel[1] == esel[1]);
      for (int j = 0; j < MOE_K; ++j)
        CHECK(fabsf(w[j] - ew[j]) <= 1e-6f * (1.0f + fabsf(ew[j])));
      for (int e = 0; e < MOE_NE; ++e) /* probs are the full softmax, all experts */
        CHECK(fabsf(probs[e] - p[e]) <= 1e-6f * (1.0f + p[e]));
    }
  }
}

/* Full FFN vs the independent reference, over norm_topk on/off and scale 1.0/1.7
 * (flipped on the live model fields, which llama_moe_ffn reads each call). */
static void moe_ffn_check(const uint8_t* blob, size_t len, bool shared) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  LlamaModel m;
  oc_kv_set_type(OC_KV_F32);
  CHECK(llama_load(&m, &g, 0, err, sizeof err) == 0);
  CHECK(m.has_moe);
  const LlamaLayer* L = &m.layers[m.n_layers - 1];
  CHECK(L->is_moe && (L->ffn_gate_shexp != NULL) == shared);

  float x[MOE_H];
  for (size_t i = 0; i < m.hidden; ++i)
    x[i] = 0.6f * sinf(0.7f * (float)i + 1.3f) + 0.2f; /* any hidden vector */

  float got[MOE_H], ref[MOE_H];
  for (int tv = 0; tv < 2; ++tv) {
    for (int sv = 0; sv < 2; ++sv) {
      m.norm_topk_prob = tv == 1;
      m.expert_weights_scale = sv == 1 ? 1.7f : 1.0f;
      float gap = moe_ref(&m, L, x, m.norm_topk_prob, m.expert_weights_scale, ref);
      CHECK(gap > 1e-3f); /* unambiguous top-k: model and reference agree on selection */
      llama_moe_ffn(&m, L, x, got);
      for (size_t i = 0; i < m.hidden; ++i) {
        if (fabsf(got[i] - ref[i]) <= 2e-3f * (1.0f + fabsf(ref[i]))) continue;
        fprintf(stderr,
                "FAIL moe ffn ref shared=%d norm=%d scale=%.1f idx %zu: "
                "got %.9g want %.9g\n",
                (int)shared, tv, (double)m.expert_weights_scale, i,
                (double)got[i], (double)ref[i]);
        exit(1);
      }
    }
  }
  llama_free(&m);
}

/* Dispatch status for the MoE arch strings. The llama LOADER is arch-agnostic
 * (it dispatches on which tensors a layer carries, not the arch string), so
 * llama_load already builds every MoE model and the checks above drive them
 * through it directly. But model.c's model_load — owned by another phase — does
 * NOT list these arch strings, so a real GGUF with e.g. general.architecture=
 * qwen3moe is rejected at the top level. This pins that gap: when model.c gains
 * the strings, these reject-assertions fail and this note is updated to assert
 * MODEL_LLAMA + has_moe. (model.c is not edited here — reported for next phase.) */
static void moe_dispatch_note(void) {
  /* Softmax-gated MoE archs now route to the llama path (Mixtral is arch
   * "llama" already). A plain fixture loads as dense — MoE is detected by
   * tensor presence — so acceptance is model_load == 0. */
  const char* ok[] = {"qwen2moe", "qwen3moe", "olmoe", NULL};
  for (size_t i = 0; ok[i]; ++i) {
    size_t len = 0;
    uint8_t* blob = llama_fixture(&len, LLAMA_PLAIN, ok[i]);
    char err[256] = {0};
    GgufFile g;
    CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
    Model M;
    CHECK(model_load(&M, &g, 0, false, err, sizeof err) == 0);
    model_free(&M);
    gguf_close(&g);
    free(blob);
  }
  /* gpt-oss stays rejected: attention sinks (a per-head bias in the softmax
   * denominator) are not implemented, so a clean error beats silently-wrong
   * tokens. deepseek2 now routes to the MLA family (a plain-llama fixture keyed
   * deepseek2 would route there and then fail to LOAD for lack of MLA tensors,
   * which is a different, correct kind of loud failure — the real deepseek2
   * dispatch is exercised by deepseek_dispatch_case with a proper MLA fixture). */
  const char* rej[] = {"gpt-oss", NULL};
  for (size_t i = 0; rej[i]; ++i) {
    size_t len = 0;
    uint8_t* blob = llama_fixture(&len, LLAMA_PLAIN, rej[i]);
    char err[256] = {0};
    GgufFile g;
    CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
    Model M;
    CHECK(model_load(&M, &g, 0, false, err, sizeof err) != 0);
    CHECK(strstr(err, "unsupported architecture") != NULL);
    gguf_close(&g);
    free(blob);
  }
}

/* ---- DeepSeek-V2/V3: MLA + group-routed MoE --------------------------------
 *
 * MLA fails SILENTLY. A wrong latent up-projection stride, a nope/rope key split
 * off by qk_rope, or a dropped softmax scale all produce finite, plausible
 * numbers — and batched==sequential cannot see ANY of them, because both forward
 * paths call the same mla_project_pos and the same mla_attn. So the bar here is
 * an INDEPENDENT end-to-end reference (ds_ref_logits): it re-derives the whole
 * model the "direct" way — materialize the full per-head K/V from the latent
 * itself, plain O(T^2) attention with its own scale, its own adjacent-pair RoPE,
 * its own group router, its own ggml expert strides — and asserts the model's
 * logits equal it. Same idea as moe_ref, one level up. */

#define DS_H 64
#define DS_NL 3   /* layer 0 leading-dense, layers 1-2 MoE */
#define DS_NH 2
#define DS_KVL 32 /* kv_lora_rank */
#define DS_QL 32  /* q_lora_rank (compressed q path) */
#define DS_QN 16  /* qk_nope_head_dim */
#define DS_QR 16  /* qk_rope_head_dim */
#define DS_QH (DS_QN + DS_QR)
#define DS_VD 16 /* v_head_dim != qk_head_dim, so a v/qk mix-up shows */
#define DS_NE 8
#define DS_K 2
#define DS_NG 4  /* 2 experts per group... */
#define DS_TKG 2 /* ...top-2 groups kept => 4 candidates, top-2 of them routed:
                  * both the group cut AND the within-group top-k are load-bearing */
#define DS_EI 96 /* expert_inter != hidden != shexp_inter */
#define DS_SI 64
#define DS_FF 96 /* dense (leading) layer inter */
#define DS_V 32

/* q_lora == 0 (DeepSeek-V2-Lite) takes the OTHER q path: a plain attn_q instead
 * of the attn_q_a/attn_q_a_norm/attn_q_b LoRA triple, and the loader must infer
 * qk_head_dim from attn_q's rows over hidden, not over q_lora. */
static uint8_t* deepseek_fixture(size_t* len, const char* arch, bool q_compressed) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  rs = 31337u;
  kv_str(&m, "general.architecture", arch);
  kv_u32p(&m, arch, "embedding_length", DS_H);
  kv_u32p(&m, arch, "block_count", DS_NL);
  kv_u32p(&m, arch, "attention.head_count", DS_NH);
  kv_u32p(&m, arch, "feed_forward_length", DS_FF);
  kv_u32p(&m, arch, "context_length", 64);
  kv_f32p(&m, arch, "attention.layer_norm_rms_epsilon", 1e-6f);
  kv_f32p(&m, arch, "rope.freq_base", 1e4f);
  /* MLA geometry */
  kv_u32p(&m, arch, "attention.kv_lora_rank", DS_KVL);
  if (q_compressed) kv_u32p(&m, arch, "attention.q_lora_rank", DS_QL);
  kv_u32p(&m, arch, "attention.key_length", DS_QH);
  kv_u32p(&m, arch, "rope.dimension_count", DS_QR);
  kv_u32p(&m, arch, "attention.value_length", DS_VD);
  /* DeepSeek MoE: sigmoid gating + V3 bias + group-limited routing */
  kv_u32p(&m, arch, "expert_count", DS_NE);
  kv_u32p(&m, arch, "expert_used_count", DS_K);
  kv_u32p(&m, arch, "expert_shared_count", 1);
  kv_u32p(&m, arch, "expert_feed_forward_length", DS_EI);
  kv_u32p(&m, arch, "expert_group_count", DS_NG);
  kv_u32p(&m, arch, "expert_group_used_count", DS_TKG);
  kv_u32p(&m, arch, "expert_weights_norm", 1);
  kv_u32p(&m, arch, "expert_gating_func", 2); /* 2 == sigmoid (V3) */
  kv_f32p(&m, arch, "expert_weights_scale", 1.5f);
  kv_u32p(&m, arch, "leading_dense_block_count", 1);

  tsr(&m, "token_embd.weight", DS_V, DS_H, 0.0f, 0.2f);
  tsr(&m, "output.weight", DS_V, DS_H, 0.0f, 0.2f); /* untied */
  tsr(&m, "output_norm.weight", 0, DS_H, 1.0f, 0.1f);

  static char names[DS_NL][20][56];
  for (size_t l = 0; l < DS_NL; ++l) {
    char(*nm)[56] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 56, "blk.%zu." suffix, l), nm[i++])
    tsr(&m, NAME("attn_norm.weight"), 0, DS_H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_norm.weight"), 0, DS_H, 1.0f, 0.1f);
    /* Every big MLA projection is Q8_0: its row stride (34 B / 32 vals) matches
     * neither the f32 stride nor the value count, so a byte-stride slip reads
     * garbage instead of quietly landing on the next row. */
    if (q_compressed) {
      tsr_q8(&m, NAME("attn_q_a.weight"), 0, DS_QL, DS_H, 0.15f);
      tsr(&m, NAME("attn_q_a_norm.weight"), 0, DS_QL, 1.0f, 0.1f);
      tsr_q8(&m, NAME("attn_q_b.weight"), 0, DS_NH * DS_QH, DS_QL, 0.15f);
    } else { /* V2-Lite: one plain q projection straight off the hidden state */
      tsr_q8(&m, NAME("attn_q.weight"), 0, DS_NH * DS_QH, DS_H, 0.15f);
    }
    tsr_q8(&m, NAME("attn_kv_a_mqa.weight"), 0, DS_KVL + DS_QR, DS_H, 0.15f);
    tsr(&m, NAME("attn_kv_a_norm.weight"), 0, DS_KVL, 1.0f, 0.1f);
    tsr_q8(&m, NAME("attn_kv_b.weight"), 0, DS_NH * (DS_QN + DS_VD), DS_KVL, 0.15f);
    tsr_q8(&m, NAME("attn_output.weight"), 0, DS_H, DS_NH * DS_VD, 0.15f);
    if (l == 0) { /* leading dense block */
      tsr_q8(&m, NAME("ffn_gate.weight"), 0, DS_FF, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_up.weight"), 0, DS_FF, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_down.weight"), 0, DS_H, DS_FF, 0.15f);
    } else {
      tsr(&m, NAME("ffn_gate_inp.weight"), DS_NE, DS_H, 0.0f, 0.5f);
      tsr(&m, NAME("exp_probs_b.bias"), 0, DS_NE, 0.0f, 0.4f); /* V3 bias */
      tsr_q8(&m, NAME("ffn_gate_exps.weight"), DS_NE, DS_EI, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_up_exps.weight"), DS_NE, DS_EI, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_down_exps.weight"), DS_NE, DS_H, DS_EI, 0.15f);
      tsr_q8(&m, NAME("ffn_gate_shexp.weight"), 0, DS_SI, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_up_shexp.weight"), 0, DS_SI, DS_H, 0.15f);
      tsr_q8(&m, NAME("ffn_down_shexp.weight"), 0, DS_H, DS_SI, 0.15f);
    }
#undef NAME
  }
  return build(&m, len);
}

static void deepseek_reset(DeepseekModel* m) {
  for (size_t l = 0; l < m->n_layers; ++l) {
    memset(m->layers[l].kv_lat_cache, 0, m->ctx * m->kv_lora * sizeof(float));
    memset(m->layers[l].k_pe_cache, 0, m->ctx * m->qk_rope * sizeof(float));
  }
  m->kv_len = 0;
}

/* ---- the independent reference --------------------------------------------- */

static void ds_rms(float* o, const float* x, const float* w, size_t n, float eps) {
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) s += x[i] * x[i];
  float inv = 1.0f / sqrtf(s / (float)n + eps);
  for (size_t i = 0; i < n; ++i) o[i] = x[i] * inv * w[i];
}

/* Adjacent-pair (ggml NORMAL) RoPE, written from the definition — NOT a call
 * into oc_rope_normal. If model_deepseek ever switched q_pe/k_pe to split-half
 * NeoX rope this diverges immediately. */
static void ds_rope(float* p, size_t d, size_t pos, float theta) {
  for (size_t i = 0; i < d / 2; ++i) {
    float freq = powf(theta, -2.0f * (float)i / (float)d);
    float a = (float)pos * freq, c = cosf(a), s = sinf(a);
    float x = p[2 * i], y = p[2 * i + 1];
    p[2 * i] = x * c - y * s;
    p[2 * i + 1] = x * s + y * c;
  }
}

/* Hand reference for the DeepSeek group-limited router. Written straight from
 * the spec in model_deepseek.h; shares no code with deepseek_moe_route. */
static void ds_route_ref(const DeepseekModel* m, const float* logit, const float* bias,
                         int* sel, float* w) {
  const size_t NE = m->n_experts, K = m->n_experts_used;
  float prob[DS_NE], score[DS_NE];
  CHECK(NE <= DS_NE && K <= DS_K);

  if (m->gating_sigmoid) {
    for (size_t e = 0; e < NE; ++e) prob[e] = 1.0f / (1.0f + expf(-logit[e]));
  } else {
    float mx = logit[0], s = 0.0f;
    for (size_t e = 1; e < NE; ++e) if (logit[e] > mx) mx = logit[e];
    for (size_t e = 0; e < NE; ++e) { prob[e] = expf(logit[e] - mx); s += prob[e]; }
    for (size_t e = 0; e < NE; ++e) prob[e] /= s;
  }
  for (size_t e = 0; e < NE; ++e) score[e] = prob[e] + (bias ? bias[e] : 0.0f);

  /* Group cut: group score = sum of its top-2 selection scores; keep top tkg. */
  if (m->n_group > 1 && m->topk_group < m->n_group) {
    const size_t NG = m->n_group, EPG = NE / NG, TKG = m->topk_group;
    float gs[DS_NG];
    bool gkeep[DS_NG] = {false};
    for (size_t g = 0; g < NG; ++g) {
      float b1 = -INFINITY, b2 = -INFINITY;
      for (size_t e = g * EPG; e < (g + 1) * EPG; ++e) {
        if (score[e] > b1) { b2 = b1; b1 = score[e]; }
        else if (score[e] > b2) { b2 = score[e]; }
      }
      gs[g] = EPG >= 2 ? b1 + b2 : b1;
    }
    for (size_t s = 0; s < TKG; ++s) {
      size_t best = 0;
      float bv = -INFINITY;
      for (size_t g = 0; g < NG; ++g)
        if (!gkeep[g] && gs[g] > bv) { bv = gs[g]; best = g; }
      gkeep[best] = true;
    }
    for (size_t e = 0; e < NE; ++e)
      if (!gkeep[e / EPG]) score[e] = -INFINITY;
  }

  bool taken[DS_NE] = {false};
  float wsum = 0.0f;
  for (size_t s = 0; s < K; ++s) {
    size_t best = 0;
    float bv = -INFINITY;
    for (size_t e = 0; e < NE; ++e)
      if (!taken[e] && score[e] > bv) { bv = score[e]; best = e; }
    taken[best] = true;
    sel[s] = (int)best;
    w[s] = prob[best]; /* the ORIGINAL prob, never the bias-corrected score */
    wsum += prob[best];
  }
  float denom = (m->norm_topk_prob && wsum > 0.0f) ? wsum : 1.0f;
  float scale = m->routed_scale > 0.0f ? m->routed_scale : 1.0f;
  for (size_t s = 0; s < K; ++s) w[s] = scale * w[s] / denom;
}

/* Independent DeepSeek FFN for one token (routed experts + shared expert), using
 * the from-first-principles ggml expert stride in ref_expert_matvec. */
static void ds_ref_ffn(const DeepseekModel* m, const DeepseekLayer* L, const float* x,
                       float* out) {
  const size_t H = m->hidden;
  for (size_t i = 0; i < H; ++i) out[i] = 0.0f;

  if (!L->is_moe) {
    float g[DS_FF], u[DS_FF], d[DS_H];
    ref_expert_matvec(L->ffn_gate, 0, m->inter, H, x, g);
    ref_expert_matvec(L->ffn_up, 0, m->inter, H, x, u);
    for (size_t r = 0; r < m->inter; ++r) g[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
    ref_expert_matvec(L->ffn_down, 0, H, m->inter, g, d);
    for (size_t i = 0; i < H; ++i) out[i] = d[i];
    return;
  }

  float logit[DS_NE], w[DS_K];
  int sel[DS_K];
  ref_expert_matvec(L->ffn_gate_inp, 0, m->n_experts, H, x, logit);
  ds_route_ref(m, logit, L->ffn_exp_probs_b, sel, w);

  for (size_t j = 0; j < m->n_experts_used; ++j) {
    float g[DS_EI], u[DS_EI], d[DS_H];
    size_t e = (size_t)sel[j];
    ref_expert_matvec(L->ffn_gate_exps, e, m->expert_inter, H, x, g);
    ref_expert_matvec(L->ffn_up_exps, e, m->expert_inter, H, x, u);
    for (size_t r = 0; r < m->expert_inter; ++r)
      g[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
    ref_expert_matvec(L->ffn_down_exps, e, H, m->expert_inter, g, d);
    for (size_t i = 0; i < H; ++i) out[i] += w[j] * d[i];
  }
  if (L->ffn_gate_shexp) { /* always-on, weight 1 */
    float g[DS_SI], u[DS_SI], d[DS_H];
    ref_expert_matvec(L->ffn_gate_shexp, 0, m->shexp_inter, H, x, g);
    ref_expert_matvec(L->ffn_up_shexp, 0, m->shexp_inter, H, x, u);
    for (size_t r = 0; r < m->shexp_inter; ++r)
      g[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
    ref_expert_matvec(L->ffn_down_shexp, 0, H, m->shexp_inter, g, d);
    for (size_t i = 0; i < H; ++i) out[i] += d[i];
  }
}

/* The whole model, the DIRECT way: no latent cache trick, no online softmax, no
 * shared kv_b_recon buffer. Per position we build the FULL per-head K and V from
 * the latent right here, concatenate the decoupled RoPE key onto k, and run a
 * plain scale-softmax-weighted-sum attention. Everything (the nope/rope split,
 * the scale 1/sqrt(qk_head_dim), the expert strides, the router) is re-derived,
 * so an error in model_deepseek.c that is consistent across its own decode and
 * batch paths still shows up here. Writes logits for the LAST token. */
static void ds_ref_logits(const DeepseekModel* m, const int32_t* ids, size_t n,
                          float* logits) {
  const size_t H = m->hidden, nH = m->n_head, qn = m->qk_nope, qr = m->qk_rope;
  const size_t qh = m->qk_head_dim, vd = m->v_head_dim, KVL = m->kv_lora;
  const size_t hd_kv = qn + vd;
  const float eps = m->eps, theta = m->rope_theta;
  const float scale = 1.0f / sqrtf((float)qh); /* mscale == 1: no YaRN KVs here */
  CHECK(n <= N_TOK && H == DS_H && nH == DS_NH);

  static float x[N_TOK][DS_H];
  static float lat[DS_NL][N_TOK][DS_KVL];   /* the cached latent, our own copy */
  static float kpe[DS_NL][N_TOK][DS_QR];    /* the decoupled RoPE key */
  static float qb[N_TOK][DS_NH * DS_QH];    /* RoPE'd query */
  static float att[N_TOK][DS_NH * DS_VD];

  size_t erow = oc_row_bytes(m->tok_embd->ggml_type, H);
  for (size_t i = 0; i < n; ++i) {
    size_t tk = (size_t)ids[i] < m->vocab ? (size_t)ids[i] : m->vocab - 1;
    CHECK(oc_dequant_row(m->tok_embd->ggml_type, m->tok_embd->data + tk * erow, x[i], H) == 0);
  }

  for (size_t l = 0; l < m->n_layers; ++l) {
    const DeepseekLayer* L = &m->layers[l];

    for (size_t i = 0; i < n; ++i) { /* per-position projections */
      float nrm[DS_H];
      ds_rms(nrm, x[i], L->attn_norm, H, eps);

      if (m->q_compressed) {
        float cq[DS_QL], cqn[DS_QL];
        ref_expert_matvec(L->mla_q_a, 0, m->q_lora, H, nrm, cq);
        ds_rms(cqn, cq, L->mla_q_a_norm, m->q_lora, eps);
        ref_expert_matvec(L->mla_q_b, 0, nH * qh, m->q_lora, cqn, qb[i]);
      } else {
        ref_expert_matvec(L->mla_q_b, 0, nH * qh, H, nrm, qb[i]);
      }
      for (size_t h = 0; h < nH; ++h) /* only the pe tail of each head rotates */
        ds_rope(qb[i] + h * qh + qn, qr, i, theta);

      float kva[DS_KVL + DS_QR];
      ref_expert_matvec(L->mla_kv_a_mqa, 0, KVL + qr, H, nrm, kva);
      ds_rms(lat[l][i], kva, L->mla_kv_a_norm, KVL, eps);
      memcpy(kpe[l][i], kva + KVL, qr * sizeof(float));
      ds_rope(kpe[l][i], qr, i, theta);
    }

    for (size_t i = 0; i < n; ++i) { /* plain causal attention on the FULL K/V */
      float kv[DS_NH * (DS_QN + DS_VD)]; /* reconstructed per-head [k_nope | v] */
      float sc[N_TOK][DS_NH];
      for (size_t t = 0; t <= i; ++t) {
        ref_expert_matvec(L->mla_kv_b, 0, nH * hd_kv, KVL, lat[l][t], kv);
        for (size_t h = 0; h < nH; ++h) {
          const float* q_nope = qb[i] + h * qh;
          const float* q_pe = q_nope + qn;
          const float* k_nope = kv + h * hd_kv;
          sc[t][h] = (oc_dot_f32(q_nope, k_nope, qn) +
                      oc_dot_f32(q_pe, kpe[l][t], qr)) * scale;
        }
      }
      for (size_t h = 0; h < nH; ++h) {
        float mx = -INFINITY, sum = 0.0f;
        for (size_t t = 0; t <= i; ++t)
          if (sc[t][h] > mx) mx = sc[t][h];
        for (size_t t = 0; t <= i; ++t) sum += expf(sc[t][h] - mx);
        float* oh = att[i] + h * vd;
        for (size_t d = 0; d < vd; ++d) oh[d] = 0.0f;
        for (size_t t = 0; t <= i; ++t) {
          ref_expert_matvec(L->mla_kv_b, 0, nH * hd_kv, KVL, lat[l][t], kv);
          float p = expf(sc[t][h] - mx) / sum;
          const float* v = kv + h * hd_kv + qn;
          for (size_t d = 0; d < vd; ++d) oh[d] += p * v[d];
        }
      }
    }

    for (size_t i = 0; i < n; ++i) {
      float proj[DS_H], nrm[DS_H], ffn[DS_H];
      ref_expert_matvec(L->attn_out, 0, H, nH * vd, att[i], proj);
      for (size_t d = 0; d < H; ++d) x[i][d] += proj[d];
      ds_rms(nrm, x[i], L->ffn_norm, H, eps);
      ds_ref_ffn(m, L, nrm, ffn);
      for (size_t d = 0; d < H; ++d) x[i][d] += ffn[d];
    }
  }

  float nrm[DS_H];
  ds_rms(nrm, x[n - 1], m->out_norm, H, eps);
  ref_expert_matvec(m->out_w, 0, m->vocab, H, nrm, logits);
}

/* ---- the cases -------------------------------------------------------------- */

static DeepseekModel* ds_open(const uint8_t* blob, size_t len, bool q_compressed,
                              DeepseekModel* m) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  if (deepseek_load(m, &g, 0, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL deepseek_load: %s\n", err);
    exit(1);
  }
  /* The loader must have inferred exactly the geometry the fixture encodes —
   * everything downstream (strides, splits, scale) is derived from these. */
  CHECK(m->hidden == DS_H && m->n_layers == DS_NL && m->n_head == DS_NH);
  CHECK(m->kv_lora == DS_KVL && m->q_compressed == q_compressed);
  CHECK(m->q_lora == (q_compressed ? DS_QL : 0));
  CHECK(m->qk_nope == DS_QN && m->qk_rope == DS_QR && m->qk_head_dim == DS_QH);
  CHECK(m->v_head_dim == DS_VD && m->vocab == DS_V);
  CHECK(fabsf(m->softmax_scale - 1.0f / sqrtf((float)DS_QH)) < 1e-6f);
  CHECK(m->has_moe && m->n_experts == DS_NE && m->n_experts_used == DS_K);
  CHECK(m->n_group == DS_NG && m->topk_group == DS_TKG);
  CHECK(m->expert_inter == DS_EI && m->shexp_inter == DS_SI);
  CHECK(m->gating_sigmoid && m->norm_topk_prob);
  CHECK(!m->layers[0].is_moe && m->layers[0].ffn_gate);   /* leading dense */
  CHECK(m->layers[1].is_moe && m->layers[1].ffn_exp_probs_b);
  CHECK(m->layers[DS_NL - 1].ffn_gate_shexp);             /* shared expert */
  return m;
}

/* Batched prefill == sequential decode, over the standard splits + a resumed
 * batch (latent cache already populated for t < pos0). */
static void deepseek_case(const uint8_t* blob, size_t len, bool qc, size_t batch) {
  DeepseekModel m;
  ds_open(blob, len, qc, &m);
  m.batch = batch;

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 11 + 5) % m.vocab);

  static float seq[N_TOK][DS_V];
  deepseek_reset(&m);
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lg = deepseek_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(seq[i], lg, m.vocab * sizeof(float));
  }

  char what[48];
  snprintf(what, sizeof what, "deepseek qlora=%d batch=%zu", (int)qc, batch);
  for (size_t s = 0; s < N_SPLITS; ++s) {
    size_t n = SPLITS[s];
    deepseek_reset(&m);
    float* lg = deepseek_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same(what, n, lg, seq[n - 1], m.vocab);
  }

  deepseek_reset(&m);
  for (size_t i = 0; i < 6; ++i) CHECK(deepseek_forward(&m, ids[i], i, false) == NULL);
  float* lg = deepseek_forward_batch(&m, ids + 6, N_TOK - 6, 6, true);
  CHECK(lg != NULL);
  same(what, N_TOK - 6, lg, seq[N_TOK - 1], m.vocab);

  /* Rewind: drop the tail, regenerate the same tokens, get the same logits. */
  deepseek_reset(&m);
  for (size_t i = 0; i < 8; ++i) CHECK(deepseek_forward(&m, ids[i], i, true) != NULL);
  CHECK(m.kv_len == 8);
  deepseek_kv_rewind(&m, 3);
  CHECK(m.kv_len == 3);
  deepseek_kv_rewind(&m, 99); /* past the end: no-op */
  CHECK(m.kv_len == 3);
  for (size_t i = 3; i < 8; ++i) {
    float* r = deepseek_forward(&m, ids[i], i, true);
    CHECK(r != NULL);
    same_tol("deepseek rewind", i, r, seq[i], m.vocab, 1e-5f);
  }

  deepseek_free(&m);
}

/* THE MLA CHECK: model logits == the independent direct-attention reference,
 * over a whole 17-token prompt, through both the decode and the batched path.
 * This is what catches a wrong kv_b up-projection, a swapped nope/rope key
 * split, or a dropped/wrong softmax scale — none of which batched==sequential
 * can see. */
static void deepseek_mla_ref_case(const uint8_t* blob, size_t len, bool qc) {
  DeepseekModel m;
  ds_open(blob, len, qc, &m);

  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 11 + 5) % m.vocab);

  static float ref[DS_V];
  const size_t ns[] = {1, 2, 5, 17};
  for (size_t s = 0; s < sizeof ns / sizeof ns[0]; ++s) {
    size_t n = ns[s];
    ds_ref_logits(&m, ids, n, ref);

    deepseek_reset(&m);
    float* lg = NULL;
    for (size_t i = 0; i < n; ++i) lg = deepseek_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    same_tol("deepseek MLA decode vs direct reference", n, lg, ref, m.vocab, 3e-3f);

    deepseek_reset(&m);
    lg = deepseek_forward_batch(&m, ids, n, 0, true);
    CHECK(lg != NULL);
    same_tol("deepseek MLA batch vs direct reference", n, lg, ref, m.vocab, 3e-3f);
  }
  deepseek_free(&m);
}

/* THE ROUTER CHECK: deepseek_moe_route on fixed logits vs the hand reference,
 * across sigmoid/softmax gating, bias on/off, norm_topk on/off, routed_scale,
 * and topk_group == n_group (no cut) vs < n_group (cut active). Plus one fully
 * hand-computed expectation, so the two implementations cannot agree by both
 * being wrong in the same way. */
static void deepseek_route_case(const uint8_t* blob, size_t len) {
  DeepseekModel m;
  ds_open(blob, len, true, &m);

  /* Groups of 2: {0,1} {2,3} {4,5} {6,7}. Group scores (sigmoid, no bias):
   * g0=s(3.0)+s(-1.0), g1=s(2.0)+s(0.5), g2=s(2.5)+s(2.4), g3=s(-2)+s(-3).
   * g2 (~1.84) > g0 (~1.22) > g1 (~1.5)? -> compute, don't guess: the hand
   * expectation below is asserted explicitly for exactly this vector. */
  const float logit[DS_NE] = {3.0f, -1.0f, 2.0f, 0.5f, 2.5f, 2.4f, -2.0f, -3.0f};
  const float bias[DS_NE] = {0.0f, 0.0f, 0.30f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  float prob[DS_NE], selscore[DS_NE], grp[DS_NG];
  int grpsel[DS_NG], sel[DS_K];
  float w[DS_K];
  int rsel[DS_K];
  float rw[DS_K];

  for (int gi = 0; gi < 2; ++gi) {         /* sigmoid / softmax gating */
    for (int bi = 0; bi < 2; ++bi) {       /* V3 bias on / off */
      for (int ni = 0; ni < 2; ++ni) {     /* norm_topk on / off */
        for (int si = 0; si < 2; ++si) {   /* routed_scale 1.0 / 1.5 */
          for (int ti = 0; ti < 2; ++ti) { /* group cut active / inactive */
            m.gating_sigmoid = gi == 0;
            m.norm_topk_prob = ni == 1;
            m.routed_scale = si == 1 ? 1.5f : 1.0f;
            m.topk_group = ti == 0 ? DS_TKG : DS_NG; /* DS_NG == no cut */
            const float* b = bi == 1 ? bias : NULL;

            deepseek_moe_route(&m, logit, b, prob, selscore, grp, grpsel, sel, w);
            ds_route_ref(&m, logit, b, rsel, rw);
            for (size_t j = 0; j < DS_K; ++j) {
              if (sel[j] == rsel[j] &&
                  fabsf(w[j] - rw[j]) <= 1e-5f * (1.0f + fabsf(rw[j])))
                continue;
              fprintf(stderr,
                      "FAIL deepseek route (gating=%s bias=%d norm=%d scale=%.1f "
                      "topk_group=%zu) slot %zu: got expert %d w %.9g, "
                      "want expert %d w %.9g\n",
                      gi == 0 ? "sigmoid" : "softmax", bi, ni, (double)m.routed_scale,
                      m.topk_group, j, sel[j], (double)w[j], rsel[j], (double)rw[j]);
              exit(1);
            }
          }
        }
      }
    }
  }

  /* Fully hand-computed anchor: sigmoid gating, bias +0.30 on expert 2, groups
   * of 2, top-2 groups, top-2 experts, norm_topk on, scale 1.5.
   *   prob  = sigmoid(logit) = [.9526 .2689 .8808 .6225 .9241 .9168 .1192 .0474]
   *   score = prob + bias    = [.9526 .2689 1.1808 .6225 .9241 .9168 .1192 .0474]
   *   group sums (top-2 each, groups of 2 => the whole group):
   *     g0 = .9526+.2689 = 1.2215
   *     g1 = 1.1808+.6225 = 1.8033   <- kept
   *     g2 = .9241+.9168 = 1.8409    <- kept
   *     g3 = .1192+.0474 = 0.1666
   *   candidates {2,3,4,5}; top-2 by score: 2 (1.1808) then 4 (.9241)
   *   weights from the UNBIASED prob: .880797 and .924142; norm_topk -> /1.804939;
   *   scale 1.5 -> [.731989, .768011]
   * If the bias leaked into the WEIGHT (a classic V3 bug) w[0] would be .980654. */
  m.gating_sigmoid = true;
  m.norm_topk_prob = true;
  m.routed_scale = 1.5f;
  m.topk_group = DS_TKG;
  deepseek_moe_route(&m, logit, bias, prob, selscore, grp, grpsel, sel, w);
  CHECK(sel[0] == 2 && sel[1] == 4);
  CHECK(fabsf(w[0] - 0.731989f) < 1e-5f);
  CHECK(fabsf(w[1] - 0.768011f) < 1e-5f);

  deepseek_free(&m);
}

/* model_load must route every deepseek arch spelling to the MLA family (and
 * actually LOAD the MLA tensors), not to the dense llama path. */
static void deepseek_dispatch_case(void) {
  const char* archs[] = {"deepseek2", "deepseek", "deepseek_v2",
                         "deepseek_v3", "deepseek-v2", "deepseek-v3", NULL};
  for (size_t i = 0; archs[i]; ++i) {
    size_t len = 0;
    uint8_t* blob = deepseek_fixture(&len, archs[i], i % 2 == 0);
    char err[256] = {0};
    GgufFile g;
    CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
    Model M;
    if (model_load(&M, &g, 0, false, err, sizeof err) != 0) {
      fprintf(stderr, "FAIL deepseek dispatch arch=%s: %s\n", archs[i], err);
      exit(1);
    }
    CHECK(M.family == MODEL_DEEPSEEK);
    CHECK(((DeepseekModel*)M.handle)->has_moe);
    CHECK(M.forward(M.handle, 3, 0, true) != NULL); /* the MLA path actually runs */
    model_free(&M);
    free(blob);
  }
}

/* ---- KV rewind -------------------------------------------------------------
 * Generate N tokens sequentially (recording each position's logits), drop the
 * tail with *_kv_rewind, then regenerate the dropped positions with the SAME
 * tokens and assert the logits reappear. This is the contract speculative
 * decoding and chat editing depend on: after a rewind the cache reproduces
 * exactly what a fresh run to that length would have held. Regeneration is
 * bit-exact (same inputs, same code, deterministic codecs), so the tolerance is
 * tight for every precision — an f16/q8 store that were not deterministic, or a
 * kv_len that did not move, would break it. */
static void llama_rewind_case(const uint8_t* blob, size_t len, OcKvType kt) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  LlamaModel m;
  oc_kv_set_type(kt);
  CHECK(llama_load(&m, &g, 0, err, sizeof err) == 0);
  CHECK(m.kv_type == kt);
  const size_t N = 8, K = 3;
  int32_t ids[8];
  for (size_t i = 0; i < N; ++i) ids[i] = (int32_t)((i * 5 + 2) % m.vocab);

  llama_reset(&m);
  static float ref[8][32];
  for (size_t i = 0; i < N; ++i) {
    float* lg = llama_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(ref[i], lg, m.vocab * sizeof(float));
  }
  CHECK(m.kv_len == N);
  llama_kv_rewind(&m, K);
  CHECK(m.kv_len == K);
  llama_kv_rewind(&m, N + 5); /* rewinding past the end is a no-op */
  CHECK(m.kv_len == K);

  char what[48];
  snprintf(what, sizeof what, "llama rewind kv=%s", oc_kv_type_name(kt));
  for (size_t i = K; i < N; ++i) {
    float* lg = llama_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    same_tol(what, i, lg, ref[i], m.vocab, 1e-5f);
  }
  CHECK(m.kv_len == N);
  llama_free(&m);
}

/* Gemma4 rewind, kept inside the SWA window (N <= cache_cap) so no ring slot is
 * evicted — a rewind further back than cache_cap cannot be served and is
 * documented as out of contract. */
static void gemma4_rewind_case(const uint8_t* blob, size_t len, OcKvType kt) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Gemma4Model m;
  oc_kv_set_type(kt);
  CHECK(gemma4_load(&m, &g, 0, kt == OC_KV_Q4, err, sizeof err) == 0);
  const size_t N = 4, K = 1; /* SWA window is 4: these never evict */
  int32_t ids[4];
  for (size_t i = 0; i < N; ++i) ids[i] = (int32_t)((i * 7 + 3) % m.vocab);

  gemma4_reset(&m);
  static float ref[4][32];
  for (size_t i = 0; i < N; ++i) {
    float* lg = gemma4_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(ref[i], lg, m.vocab * sizeof(float));
  }
  CHECK(m.kv_len == N);
  CHECK(gemma4_kv_rewind(&m, K)); /* within the window: reproducible, returns true */
  CHECK(m.kv_len == K);

  char what[48];
  snprintf(what, sizeof what, "gemma4 rewind kv=%s", oc_kv_type_name(kt));
  for (size_t i = K; i < N; ++i) {
    float* lg = gemma4_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    same_tol(what, i, lg, ref[i], m.vocab, 1e-5f);
  }
  CHECK(m.kv_len == N);
  gemma4_free(&m);
}

/* Rewind PAST the SWA window. Once generation has wrapped the 4-slot ring
 * (N > window), the tail a deep rewind would re-read has been evicted, so it is
 * NOT reproducible even at a rewind distance < window. gemma4_kv_rewind must
 * refuse it (return false, kv_len untouched) instead of silently serving stale
 * slots; only dropping the last token (pos == kv_len-1) stays exact. */
static void gemma4_rewind_evict_case(const uint8_t* blob, size_t len) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  Gemma4Model m;
  oc_kv_set_type(OC_KV_F32); /* pure ring/positional bug, no quant precision */
  CHECK(gemma4_load(&m, &g, 0, false, err, sizeof err) == 0);
  const size_t N = 8; /* window is 4: positions 0..3 are evicted by 4..7 */
  int32_t ids[8];
  for (size_t i = 0; i < N; ++i) ids[i] = (int32_t)((i * 7 + 3) % m.vocab);

  gemma4_reset(&m);
  static float ref[8][32];
  for (size_t i = 0; i < N; ++i) {
    float* lg = gemma4_forward(&m, ids[i], i, true);
    CHECK(lg != NULL);
    memcpy(ref[i], lg, m.vocab * sizeof(float));
  }
  CHECK(m.kv_len == N);

  /* Deep rewind (distance 2 < window 4, but the window tail is already gone):
   * refused, cache left intact. Before the fix this silently served stale KV. */
  CHECK(!gemma4_kv_rewind(&m, N - 2));
  CHECK(m.kv_len == N);
  /* Dropping only the last token is always reproducible. */
  CHECK(gemma4_kv_rewind(&m, N - 1));
  CHECK(m.kv_len == N - 1);
  float* lg = gemma4_forward(&m, ids[N - 1], N - 1, true);
  CHECK(lg != NULL);
  same_tol("gemma4 rewind-evict last-token", N - 1, lg, ref[N - 1], m.vocab, 1e-5f);
  CHECK(m.kv_len == N);
  gemma4_free(&m);
}

void test_forward_batch(void) {
  check_rope_modes();
  size_t len = 0;
  uint8_t* blob = gemma4_fixture(&len);
  /* batched==sequential under every KV precision, at batch 16 (one chunk) and 4
   * (internal chunking mid-window). f32/f16/q4 are near-exact — both paths round
   * K/V through the identical codec; q8 gets a looser bar since a GEMM-vs-GEMV
   * projection difference can land either side of an int8 code boundary. */
  for (size_t bi = 0; bi < 2; ++bi) {
    size_t b = bi == 0 ? 16 : 4;
    gemma4_case(blob, len, OC_KV_F32, b, 5e-4f);
    gemma4_case(blob, len, OC_KV_F16, b, 1e-3f);
    gemma4_case(blob, len, OC_KV_Q8, b, 1e-2f);
    gemma4_case(blob, len, OC_KV_Q4, b, 5e-4f); /* rotated int4 rotoquant */
  }
  gemma4_rewind_case(blob, len, OC_KV_F32);
  gemma4_rewind_case(blob, len, OC_KV_F16);
  gemma4_rewind_case(blob, len, OC_KV_Q8);
  gemma4_rewind_evict_case(blob, len);
  free(blob);

  blob = qwen36_fixture(&len);
  qwen36_case(blob, len, 16);
  qwen36_case(blob, len, 4);
  free(blob);

  qwen36_spec_case(); /* P16: MTP-draft speculative decode == plain greedy */

  const LlamaVariant vars[] = {LLAMA_PLAIN, LLAMA_BIAS_TIED, LLAMA_QKNORM};
  for (size_t vi = 0; vi < sizeof vars / sizeof vars[0]; ++vi) {
    blob = llama_fixture(&len, vars[vi], "llama");
    for (size_t bi = 0; bi < 2; ++bi) {
      size_t b = bi == 0 ? 16 : 4; /* one chunk, then internal chunking */
      llama_case(blob, len, vars[vi], b, OC_KV_F32, 5e-4f);
      llama_case(blob, len, vars[vi], b, OC_KV_F16, 1e-3f);
      llama_case(blob, len, vars[vi], b, OC_KV_Q8, 1e-2f);
    }
    if (vi == 0) { /* rewind across precisions on one variant is enough */
      llama_rewind_case(blob, len, OC_KV_F32);
      llama_rewind_case(blob, len, OC_KV_F16);
      llama_rewind_case(blob, len, OC_KV_Q8);
    }
    free(blob);
  }

  /* llama has no rotoquant (the FHT is gemma4-only), so --kv-type q4 must fall
   * back to f16 — loudly, never a silent wrong path. */
  blob = llama_fixture(&len, LLAMA_PLAIN, "llama");
  {
    char err[256] = {0};
    GgufFile g;
    CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
    LlamaModel m;
    oc_kv_set_type(OC_KV_Q4);
    CHECK(llama_load(&m, &g, 0, err, sizeof err) == 0);
    CHECK(m.kv_type == OC_KV_F16);
    llama_free(&m);
  }
  oc_kv_set_type(OC_KV_F32);
  free(blob);
  llama_dispatch_smoke();

  /* MoE: router unit test, then batched==sequential + FFN-vs-reference for both
   * the plain and shared-expert stacks, then the (currently absent) dispatch. */
  moe_route_unit();
  for (int sh = 0; sh < 2; ++sh) {
    bool shared = sh == 1;
    blob = moe_fixture(&len, shared);
    moe_case(blob, len, shared, 16); /* one chunk */
    moe_case(blob, len, shared, 4);  /* chunk boundary mid-prompt */
    moe_ffn_check(blob, len, shared);
    free(blob);
  }
  moe_dispatch_note();

  /* DeepSeek: MLA + group-routed MoE. batched==sequential proves the causal
   * mask; the direct-attention reference proves the MLA numerics; the router
   * reference proves the group cut / bias / gating. */
  for (int qc = 0; qc < 2; ++qc) { /* V2/V3 compressed q, then V2-Lite plain q */
    blob = deepseek_fixture(&len, "deepseek2", qc == 1);
    deepseek_case(blob, len, qc == 1, 16); /* one chunk */
    deepseek_case(blob, len, qc == 1, 4);  /* internal chunking mid-prompt */
    deepseek_mla_ref_case(blob, len, qc == 1);
    if (qc == 1) deepseek_route_case(blob, len);
    free(blob);
  }
  deepseek_dispatch_case();
  printf("ok deepseek MLA: logits == independent direct-attention reference "
         "(full per-head K/V materialized from the latent, own nope/rope split, "
         "own 1/sqrt(qk_head_dim) scale, own adjacent-pair rope; decode + batch, "
         "prefixes 1/2/5/17); compressed-q (V2/V3) AND plain-q (V2-Lite) paths; "
         "batched==sequential over splits 1..17 + resumed batch + kv_rewind; "
         "q8_0 MLA projections\n");
  printf("ok deepseek router: group-limited routing == hand reference over "
         "sigmoid/softmax gating x V3 bias x norm_topk x routed_scale x group-cut "
         "on/off, plus a hand-computed anchor (bias steers SELECTION only, never "
         "the weight); leading-dense + shared expert; deepseek2/v2/v3 arch "
         "spellings dispatch to the MLA family\n");

  printf("ok moe: router softmax+top-k+renorm, ffn==independent reference "
         "(q8_0 experts, norm_topk on/off, scale, shared expert on/off), "
         "batched==sequential over splits 1..17\n");

  printf("ok batched prefill == sequential decode (gemma4 swa-ring, qwen36 "
         "deltanet scan, llama plain/qkv-bias-tied/qk-norm + dispatch; "
         "%zu token prompt, splits at 1..17)\n", (size_t)N_TOK);
  printf("ok qwen36 MTP speculative decode == plain greedy (bit-exact token ids, "
         "draft-tokens 1..4; full/partial/zero accept; DeltaNet state "
         "snapshot/restore across the throwaway verify batch)\n");
  printf("ok kv-cache precision: batched==sequential for gemma4 + llama under "
         "f32/f16/q8/q4 (f16 tol 1e-3, q8 tol 1e-2); *_kv_rewind regenerates "
         "bit-exact after rewind (f32/f16/q8); gemma4 rewind past the SWA window "
         "is refused (returns false, cache intact) not silently served stale\n");
  oc_kv_set_type(OC_KV_F32); /* leave the process default clean for other suites */
}
