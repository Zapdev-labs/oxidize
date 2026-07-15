/* CUDA correctness gate: the CPU forward IS the specification.
 *
 * Builds a tiny synthetic gemma4 GGUF in memory (the same builder the CPU
 * suites use), loads it twice — once for gemma4_forward, once for the CUDA
 * backend — feeds both the same token stream, and asserts the logits agree.
 * Nothing about a CUDA kernel is trusted until it has been through here.
 *
 * It is deliberately hostile in the same way tests/test_model.c is: a 8-slot
 * sliding window with a 12-token stream (the SWA ring wraps), an SWA/global
 * layer mix (so rope_freqs, the K=V-projection layer and the per-layer rope
 * theta all get exercised), a layer_output_scale != 1, and head dims that are
 * powers of two so the rotoquant KV path is eligible.
 *
 * Both models use the F16 KV cache, because that is what the GPU cache is; the
 * comparison is then apples to apples and the tolerance can stay tight enough
 * to catch a wrong kernel rather than hide it.
 *
 * THE CPU REFERENCE IS THE EXACT ONE (scalar / AVX2), not whatever the host
 * CPU would pick. On an AVX-512-VNNI host oc_matvec routes the K-quants through
 * oc_dot_row_q8 — an int8-ACTIVATION approximation, not a reference. It is
 * orders of magnitude looser than the f32 paths on this fixture (q4_k: 1.3e-3
 * under vnni vs 2.9e-7 under avx2; q6_k, whose per-block dynamic range is the
 * widest, reaches 0.33), which would make a GPU-vs-CPU tolerance either
 * meaningless or a lie about the GPU. Set OC_ISA=avx512 to see that for
 * yourself — the env var still wins, this only picks the default.
 *
 * Coverage: every weight type check_type() admits, all-GPU and at three -ngl
 * splits, plus the rotoquant (--kv-quant) path.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/cuda/gemma4_cuda.h"
#include "../src/cuda/llama_cuda.h"
#include "../src/gguf.h"
#include "../src/model_gemma4.h"
#include "../src/model_llama.h"
#include "../src/quant.h"
#include "../src/tensor.h"
#include "gguf_build.h"

/* Geometry: cols are multiples of 256 so the K-quants (256 values/block) fit,
 * head dims are powers of two so --kv-quant is eligible. */
#define H 256
#define NL 3
#define NH 4
#define HD 64
#define KVH 2
#define FF 512
#define V 256
#define N_TOK 12

static uint8_t* fixture(size_t* len, uint32_t wtype, uint32_t etype) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  rs = 7u;
  kv_str(&m, "general.architecture", "gemma4");
  kv_u32(&m, "gemma4.embedding_length", H);
  kv_u32(&m, "gemma4.block_count", NL);
  kv_u32(&m, "gemma4.attention.head_count", NH);
  kv_u32(&m, "gemma4.attention.head_count_kv", KVH);
  kv_u32(&m, "gemma4.feed_forward_length", FF);
  kv_u32(&m, "gemma4.context_length", 64);
  kv_u32(&m, "gemma4.attention.sliding_window", 8); /* the ring wraps in 12 */
  kv_u32(&m, "gemma4.attention.sliding_window_pattern", 3); /* l=0,1 SWA; l=2 global */
  kv_f32(&m, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
  kv_f32(&m, "gemma4.final_logit_softcapping", 30.0f);
  kv_f32(&m, "gemma4.attention.scale", 1.0f);
  kv_f32(&m, "gemma4.rope.freq_base", 1e6f);
  kv_f32(&m, "gemma4.rope.freq_base_swa", 1e4f);

  tsr_any(&m, "token_embd.weight", etype, V, H, 0.2f);
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);
  tsr(&m, "rope_freqs.weight", 0, HD / 2, 1.0f, 0.2f); /* global-layer divisors */

  static char names[NL][16][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    tsr_any(&m, NAME("attn_q.weight"), wtype, NH * HD, H, 0.15f);
    tsr_any(&m, NAME("attn_k.weight"), wtype, KVH * HD, H, 0.15f);
    /* The global layer omits attn_v (K and V share the projection). */
    if (l != NL - 1) tsr_any(&m, NAME("attn_v.weight"), wtype, KVH * HD, H, 0.15f);
    tsr_any(&m, NAME("attn_output.weight"), wtype, H, NH * HD, 0.15f);
    tsr_any(&m, NAME("ffn_gate.weight"), wtype, FF, H, 0.15f);
    tsr_any(&m, NAME("ffn_up.weight"), wtype, FF, H, 0.15f);
    tsr_any(&m, NAME("ffn_down.weight"), wtype, H, FF, 0.15f);
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

static int32_t amax(const float* v, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return (int32_t)best;
}

static int fails = 0;

/* Run N_TOK tokens through the CPU forward and through the CUDA backend and
 * compare every logit at every position. Returns max |diff|. */
static void compare(const char* what, uint32_t wtype, uint32_t etype,
                    bool kv_quant, int ngl, float tol) {
  size_t len = 0;
  uint8_t* blob = fixture(&len, wtype, etype);
  char err[256] = {0};
  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 37 + 3) % V);

  /* the GPU's KV cache is f16 (or the int4 rotoquant); make the CPU's match */
  oc_kv_set_type(kv_quant ? OC_KV_Q4 : OC_KV_F16);

  GgufFile ga, gb;
  CHECK(gguf_parse(&ga, blob, len, err, sizeof err) == 0);
  CHECK(gguf_parse(&gb, blob, len, err, sizeof err) == 0);
  Gemma4Model cpu, gpu;
  CHECK(gemma4_load(&cpu, &ga, 0, kv_quant, err, sizeof err) == 0);
  CHECK(gemma4_load(&gpu, &gb, 0, kv_quant, err, sizeof err) == 0);

  Gemma4Cuda* c = NULL;
  if (gemma4_cuda_init(&c, &gpu, 1, ngl, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL %s: gemma4_cuda_init: %s\n", what, err);
    exit(1);
  }

  float ref[V];
  float worst = 0.0f;
  size_t worst_at = 0, worst_pos = 0;
  int tok_mismatch = 0;
  for (size_t i = 0; i < N_TOK; ++i) {
    float* a = gemma4_forward(&cpu, ids[i], i, true);
    CHECK(a != NULL);
    memcpy(ref, a, V * sizeof(float));

    int failed = 0;
    float* b = gemma4_cuda_step(c, &gpu, ids[i], i, true, &failed);
    if (failed || !b) {
      fprintf(stderr, "FAIL %s: cuda step failed at pos %zu\n", what, i);
      exit(1);
    }
    for (size_t j = 0; j < V; ++j) {
      CHECK(isfinite(ref[j]) && isfinite(b[j])); /* a NaN on both sides is not a pass */
      float d = fabsf(b[j] - ref[j]) / (1.0f + fabsf(ref[j]));
      if (d > worst) { worst = d; worst_at = j; worst_pos = i; }
    }
    if (amax(ref, V) != amax(b, V)) tok_mismatch++;
  }

  printf("%-34s ngl=%-2d max_rel_diff=%.3e (logit %zu @ pos %zu) argmax_mismatch=%d/%d  %s\n",
         what, ngl, (double)worst, worst_at, worst_pos, tok_mismatch, N_TOK,
         worst <= tol && tok_mismatch == 0 ? "OK" : "FAIL");
  if (worst > tol || tok_mismatch) fails++;

  gemma4_cuda_free(c);
  gemma4_free(&cpu);
  gemma4_free(&gpu);
  free(blob);
}

/* ---- llama-family fixture + gate ------------------------------------------
 * A synthetic dense llama model exercising every knob the GPU forward must
 * honor: arch (selects rope_norm — "llama"/"mistral"/"yi" => ggml NORMAL
 * adjacent-pair rope, everything else => NeoX split-half), optional q/k/v/o
 * biases, optional per-head q/k RMSNorm, tied vs untied (output.weight) head.
 * Geometry reuses the gemma4 constants (H/NH/HD/KVH/FF/V) so every weight column
 * is a multiple of 256 and the K-quants fit. */
static uint8_t* llama_fixture(size_t* len, uint32_t wtype, uint32_t etype,
                              const char* arch, bool bias, bool qknorm,
                              bool untied) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  rs = 11u;
  char key[96];
#define KV_U32(suffix, val) (snprintf(key, sizeof key, "%s." suffix, arch), \
                             kv_u32(&m, key, (val)))
#define KV_F32(suffix, val) (snprintf(key, sizeof key, "%s." suffix, arch), \
                             kv_f32(&m, key, (val)))
  kv_str(&m, "general.architecture", arch);
  KV_U32("embedding_length", H);
  KV_U32("block_count", NL);
  KV_U32("attention.head_count", NH);
  KV_U32("attention.head_count_kv", KVH);
  KV_U32("feed_forward_length", FF);
  KV_U32("context_length", 64);
  KV_F32("attention.layer_norm_rms_epsilon", 1e-5f);
  KV_F32("rope.freq_base", 1e4f);
#undef KV_U32
#undef KV_F32

  tsr_any(&m, "token_embd.weight", etype, V, H, 0.2f);
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);
  if (untied) tsr_any(&m, "output.weight", etype, V, H, 0.2f);

  static char nm[NL][24][56];
  for (size_t l = 0; l < NL; ++l) {
    int i = 0;
#define NAME(suffix) (snprintf(nm[l][i], 56, "blk.%zu." suffix, l), nm[l][i++])
    tsr_any(&m, NAME("attn_q.weight"), wtype, NH * HD, H, 0.15f);
    tsr_any(&m, NAME("attn_k.weight"), wtype, KVH * HD, H, 0.15f);
    tsr_any(&m, NAME("attn_v.weight"), wtype, KVH * HD, H, 0.15f);
    tsr_any(&m, NAME("attn_output.weight"), wtype, H, NH * HD, 0.15f);
    tsr_any(&m, NAME("ffn_gate.weight"), wtype, FF, H, 0.15f);
    tsr_any(&m, NAME("ffn_up.weight"), wtype, FF, H, 0.15f);
    tsr_any(&m, NAME("ffn_down.weight"), wtype, H, FF, 0.15f);
    tsr(&m, NAME("attn_norm.weight"), 0, H, 1.0f, 0.1f);
    tsr(&m, NAME("ffn_norm.weight"), 0, H, 1.0f, 0.1f);
    if (bias) {
      tsr(&m, NAME("attn_q.bias"), 0, NH * HD, 0.0f, 0.1f);
      tsr(&m, NAME("attn_k.bias"), 0, KVH * HD, 0.0f, 0.1f);
      tsr(&m, NAME("attn_v.bias"), 0, KVH * HD, 0.0f, 0.1f);
      tsr(&m, NAME("attn_output.bias"), 0, H, 0.0f, 0.1f);
    }
    if (qknorm) {
      tsr(&m, NAME("attn_q_norm.weight"), 0, HD, 1.0f, 0.1f);
      tsr(&m, NAME("attn_k_norm.weight"), 0, HD, 1.0f, 0.1f);
    }
#undef NAME
  }
  return build(&m, len);
}

static void compare_llama(const char* what, uint32_t wtype, uint32_t etype,
                          const char* arch, bool bias, bool qknorm, bool untied,
                          int ngl, float tol) {
  size_t len = 0;
  uint8_t* blob = llama_fixture(&len, wtype, etype, arch, bias, qknorm, untied);
  char err[256] = {0};
  int32_t ids[N_TOK];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 37 + 3) % V);

  oc_kv_set_type(OC_KV_F16); /* match the GPU's f16 KV cache */

  GgufFile ga, gb;
  CHECK(gguf_parse(&ga, blob, len, err, sizeof err) == 0);
  CHECK(gguf_parse(&gb, blob, len, err, sizeof err) == 0);
  LlamaModel cpu, gpu;
  CHECK(llama_load(&cpu, &ga, 0, err, sizeof err) == 0);
  CHECK(llama_load(&gpu, &gb, 0, err, sizeof err) == 0);

  LlamaCuda* c = NULL;
  if (llama_cuda_init(&c, &gpu, 1, ngl, err, sizeof err) != 0) {
    fprintf(stderr, "FAIL %s: llama_cuda_init: %s\n", what, err);
    exit(1);
  }

  float ref[V];
  float worst = 0.0f;
  size_t worst_at = 0, worst_pos = 0;
  int tok_mismatch = 0;
  for (size_t i = 0; i < N_TOK; ++i) {
    float* a = llama_forward(&cpu, ids[i], i, true);
    CHECK(a != NULL);
    memcpy(ref, a, V * sizeof(float));

    int failed = 0;
    float* b = llama_cuda_step(c, &gpu, ids[i], i, true, &failed);
    if (failed || !b) {
      fprintf(stderr, "FAIL %s: cuda step failed at pos %zu\n", what, i);
      exit(1);
    }
    for (size_t j = 0; j < V; ++j) {
      CHECK(isfinite(ref[j]) && isfinite(b[j]));
      float d = fabsf(b[j] - ref[j]) / (1.0f + fabsf(ref[j]));
      if (d > worst) { worst = d; worst_at = j; worst_pos = i; }
    }
    if (amax(ref, V) != amax(b, V)) tok_mismatch++;
  }

  printf("%-40s ngl=%-2d max_rel_diff=%.3e (logit %zu @ pos %zu) argmax_mismatch=%d/%d  %s\n",
         what, ngl, (double)worst, worst_at, worst_pos, tok_mismatch, N_TOK,
         worst <= tol && tok_mismatch == 0 ? "OK" : "FAIL");
  if (worst > tol || tok_mismatch) fails++;

  llama_cuda_free(c);
  llama_free(&cpu);
  llama_free(&gpu);
  free(blob);
}

int main(void) {
  oc_pool_init(1);
  if (!getenv("OC_ISA")) oc_force_isa(OC_ISA_AVX2); /* exact f32 CPU kernels */
  printf("cpu reference isa: %s\n", oc_isa_active_name());
  const int ALL = -1;

  /* weight-type coverage (token_embd carries the same type: the head is tied) */
  compare("f32", OC_F32, OC_F32, false, ALL, 2e-3f);
  compare("f16", OC_F16, OC_F16, false, ALL, 2e-3f);
  compare("q8_0", OC_Q8_0, OC_Q8_0, false, ALL, 2e-3f);
  compare("q4_0", OC_Q4_0, OC_Q4_0, false, ALL, 2e-3f);
  compare("q4_k", OC_Q4_K, OC_Q4_K, false, ALL, 2e-3f);
  compare("q5_k", OC_Q5_K, OC_Q5_K, false, ALL, 2e-3f);
  compare("q6_k", OC_Q6_K, OC_Q6_K, false, ALL, 2e-3f);
  compare("al5_xs", OC_AL5_XS, OC_AL5_XS, false, ALL, 2e-3f);
  /* what a Q4_K_M download actually is: Q4_K weights, Q6_K output head */
  compare("q4_k weights + q6_k head", OC_Q4_K, OC_Q6_K, false, ALL, 2e-3f);

  /* rotoquant KV cache */
  compare("f32 + kv-quant(int4)", OC_F32, OC_F32, true, ALL, 2e-3f);

  /* partial offload: 1 of 3 layers, 2 of 3, all 3 */
  compare("q4_k -ngl 1", OC_Q4_K, OC_Q4_K, false, 1, 2e-3f);
  compare("q4_k -ngl 2", OC_Q4_K, OC_Q4_K, false, 2, 2e-3f);
  compare("q4_k -ngl 3", OC_Q4_K, OC_Q4_K, false, 3, 2e-3f);

  /* ===================== llama-family dense path ===================== */
  printf("\n-- llama family --\n");
  const char* NORMAL = "llama"; /* ggml NORMAL adjacent-pair rope */
  const char* NEOX = "qwen3";   /* NeoX split-half rope */

  /* every weight type, classic Llama (NORMAL rope, tied head, no bias/qk-norm) */
  compare_llama("llama f32", OC_F32, OC_F32, NORMAL, false, false, false, ALL, 2e-3f);
  compare_llama("llama f16", OC_F16, OC_F16, NORMAL, false, false, false, ALL, 2e-3f);
  compare_llama("llama q8_0", OC_Q8_0, OC_Q8_0, NORMAL, false, false, false, ALL, 2e-3f);
  compare_llama("llama q4_0", OC_Q4_0, OC_Q4_0, NORMAL, false, false, false, ALL, 2e-3f);
  compare_llama("llama q4_k", OC_Q4_K, OC_Q4_K, NORMAL, false, false, false, ALL, 2e-3f);
  compare_llama("llama q5_k", OC_Q5_K, OC_Q5_K, NORMAL, false, false, false, ALL, 2e-3f);
  /* q6_k tolerance is looser ONLY here: the fixture fills K-quant blocks with
   * random bytes, and Q6_K's per-sub-block scales are random int8 (up to ±127),
   * so a "weight" reaches d*127*32 ≈ ±2.0 — far beyond a real model. With ALL
   * seven layer projections that large, the benign f32 warp-reduction-order
   * difference vs the sequential CPU dot compounds over 3 layers to ~3.4e-3
   * relative. It is NOT a decode bug: argmax still matches 12/12 (the hard gate
   * below), the same dqv<Q6_K> passes at 7.5e-5 in the gemma4 suite, and the
   * q6_k-HEAD llama cases pass at <1e-4. A real stride/scale/pairing bug is O(1)
   * and breaks argmax. */
  compare_llama("llama q6_k", OC_Q6_K, OC_Q6_K, NORMAL, false, false, false, ALL, 5e-3f);
  compare_llama("llama al5_xs", OC_AL5_XS, OC_AL5_XS, NORMAL, false, false, false, ALL, 2e-3f);

  /* what a Q4_K_M download is: Q4_K weights, Q6_K token_embd (tied head) */
  compare_llama("llama q4_k + q6_k head (tied)", OC_Q4_K, OC_Q6_K, NORMAL, false, false, false, ALL, 2e-3f);
  /* untied output.weight: the head must read a SEPARATE tensor */
  compare_llama("llama q4_k untied q6_k head", OC_Q4_K, OC_Q6_K, NORMAL, false, false, true, ALL, 2e-3f);

  /* NeoX rope + Qwen3 per-head q/k RMSNorm */
  compare_llama("qwen3 neox + qk-norm", OC_Q4_K, OC_Q4_K, NEOX, false, true, false, ALL, 2e-3f);
  /* NeoX rope + Qwen2 q/k/v/o biases */
  compare_llama("qwen2-style neox + qkv-bias", OC_Q4_K, OC_Q4_K, NEOX, true, false, false, ALL, 2e-3f);
  /* everything at once: bias + qk-norm + untied */
  compare_llama("neox bias+qknorm+untied", OC_Q4_K, OC_Q6_K, NEOX, true, true, true, ALL, 2e-3f);
  /* NORMAL rope with bias + qk-norm (rotates the right pairs under all knobs) */
  compare_llama("normal bias+qknorm", OC_Q5_K, OC_Q5_K, NORMAL, true, true, false, ALL, 2e-3f);

  /* partial offload: 1 of 3, 2 of 3 (CPU runs the tail + head), with knobs on */
  compare_llama("llama q4_k -ngl 1", OC_Q4_K, OC_Q4_K, NORMAL, false, false, false, 1, 2e-3f);
  compare_llama("llama q4_k -ngl 2", OC_Q4_K, OC_Q4_K, NORMAL, false, false, false, 2, 2e-3f);
  compare_llama("qwen3 -ngl 2 bias+qknorm untied", OC_Q4_K, OC_Q6_K, NEOX, true, true, true, 2, 2e-3f);

  oc_pool_free();
  if (fails) {
    fprintf(stderr, "\n%d CUDA equivalence case(s) FAILED\n", fails);
    return 1;
  }
  printf("\nall CUDA kernels == CPU reference forward\n");
  return 0;
}
