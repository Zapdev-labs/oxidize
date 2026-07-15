/* Vision encoder acceptance test. Builds a tiny synthetic CLIP/SigLIP vision
 * GGUF in memory (the same builder the model tests use), runs it through
 * vision_encode(), and asserts every output value matches a SECOND, fully
 * independent forward written here from naive triple-loop matmuls, its own
 * softmax and its own GELU/LayerNorm. vision.c goes through oc_matmul, the
 * threaded attention and the ISA kernels; this reference goes through none of
 * them, so a wiring, stride, ordering or activation bug diverges. Two configs:
 * a full CLIP tower (class token, pre/post-LN, all biases, 2-layer projector)
 * and a SigLIP-style minimal one (no class token, no pre/post-LN, no biases,
 * single-layer projector) to cover every optional branch. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gguf.h"
#include "../src/tensor.h"
#include "../src/vision.h"
#include "gguf_build.h"
#include "tests.h"

/* ---- independent reference ------------------------------------------------- */

static const float* td(const GgufFile* g, const char* name) {
  const GgufTensorInfo* t = gguf_tensor(g, name);
  return t ? (const float*)t->data : NULL; /* fixtures are all F32 */
}
static const float* lt(const GgufFile* g, size_t l, const char* suf) {
  char nm[128];
  snprintf(nm, sizeof nm, "v.blk.%zu.%s", l, suf);
  return td(g, nm);
}

/* y[t][r] = bias[r] + sum_c W[r][c] * x[t][c]; W row-major [rows][cols]. */
static void ref_linear(const float* W, size_t rows, size_t cols, const float* b,
                       const float* X, float* Y, size_t nt) {
  for (size_t t = 0; t < nt; ++t)
    for (size_t r = 0; r < rows; ++r) {
      float s = b ? b[r] : 0.0f;
      for (size_t c = 0; c < cols; ++c) s += W[r * cols + c] * X[t * cols + c];
      Y[t * rows + r] = s;
    }
}
static void ref_ln(float* out, const float* x, const float* w, const float* b,
                   size_t n, float eps) {
  float mean = 0.0f;
  for (size_t i = 0; i < n; ++i) mean += x[i];
  mean /= (float)n;
  float var = 0.0f;
  for (size_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
  var /= (float)n;
  float inv = 1.0f / sqrtf(var + eps);
  for (size_t i = 0; i < n; ++i)
    out[i] = (x[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}
static float ref_gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static float ref_gelu_quick(float x) { return x / (1.0f + expf(-1.702f * x)); }

/* Bidirectional MHA with a locally rolled softmax (no oc_softmax). */
static void ref_attn(const float* Q, const float* K, const float* V, float* att,
                     size_t nseq, size_t H, size_t nh) {
  size_t hd = H / nh;
  float scale = 1.0f / sqrtf((float)hd);
  float* sc = malloc(nseq * sizeof(float));
  CHECK(sc != NULL);
  for (size_t h = 0; h < nh; ++h) {
    size_t off = h * hd;
    for (size_t i = 0; i < nseq; ++i) {
      float mx = -INFINITY;
      for (size_t s = 0; s < nseq; ++s) {
        float d = 0.0f;
        for (size_t e = 0; e < hd; ++e) d += Q[i * H + off + e] * K[s * H + off + e];
        sc[s] = d * scale;
        if (sc[s] > mx) mx = sc[s];
      }
      float sum = 0.0f;
      for (size_t s = 0; s < nseq; ++s) {
        sc[s] = expf(sc[s] - mx);
        sum += sc[s];
      }
      for (size_t e = 0; e < hd; ++e) {
        float a = 0.0f;
        for (size_t s = 0; s < nseq; ++s) a += sc[s] * V[s * H + off + e];
        att[i * H + off + e] = a / sum;
      }
    }
  }
  free(sc);
}

/* Full independent forward -> out[np * PROJ]. Reads which optional tensors are
 * present exactly as vision.c does. use_gelu is unset in the fixtures, so the
 * MLP uses quick-GELU and the projector the tanh GELU. */
static void ref_forward(const GgufFile* g, const float* img, size_t S, size_t P,
                        size_t H, size_t NH, size_t NL, size_t INTER, size_t proj0,
                        size_t PROJ, float* out) {
  size_t grid = S / P, np = grid * grid, pdim = 3 * P * P;
  size_t cls = gguf_tensor(g, "v.class_embd") ? 1 : 0;
  size_t nseq = np + cls;
  float eps = 1e-5f;

  float* patches = malloc(np * pdim * sizeof(float));
  float* tok = malloc(nseq * H * sizeof(float));
  float* nrm = malloc(nseq * H * sizeof(float));
  float* q = malloc(nseq * H * sizeof(float));
  float* k = malloc(nseq * H * sizeof(float));
  float* v = malloc(nseq * H * sizeof(float));
  float* att = malloc(nseq * H * sizeof(float));
  float* ob = malloc(nseq * H * sizeof(float));
  float* f1 = malloc(nseq * INTER * sizeof(float));
  float* mm0 = malloc(np * proj0 * sizeof(float));
  CHECK(patches && tok && nrm && q && k && v && att && ob && f1 && mm0);

  for (size_t py = 0; py < grid; ++py)
    for (size_t px = 0; px < grid; ++px) {
      float* pd = patches + (py * grid + px) * pdim;
      for (size_t ic = 0; ic < 3; ++ic)
        for (size_t kh = 0; kh < P; ++kh)
          for (size_t kw = 0; kw < P; ++kw)
            pd[(ic * P + kh) * P + kw] =
                img[(ic * S + py * P + kh) * S + px * P + kw];
    }

  ref_linear(td(g, "v.patch_embd.weight"), H, pdim, td(g, "v.patch_embd.bias"),
             patches, tok + cls * H, np);
  if (cls) memcpy(tok, td(g, "v.class_embd"), H * sizeof(float));
  const float* pos = td(g, "v.position_embd.weight");
  for (size_t i = 0; i < nseq * H; ++i) tok[i] += pos[i];

  const float* prew = td(g, "v.pre_ln.weight");
  if (prew)
    for (size_t t = 0; t < nseq; ++t)
      ref_ln(tok + t * H, tok + t * H, prew, td(g, "v.pre_ln.bias"), H, eps);

  for (size_t l = 0; l < NL; ++l) {
    for (size_t t = 0; t < nseq; ++t)
      ref_ln(nrm + t * H, tok + t * H, lt(g, l, "ln1.weight"), lt(g, l, "ln1.bias"),
             H, eps);
    ref_linear(lt(g, l, "attn_q.weight"), H, H, lt(g, l, "attn_q.bias"), nrm, q, nseq);
    ref_linear(lt(g, l, "attn_k.weight"), H, H, lt(g, l, "attn_k.bias"), nrm, k, nseq);
    ref_linear(lt(g, l, "attn_v.weight"), H, H, lt(g, l, "attn_v.bias"), nrm, v, nseq);
    ref_attn(q, k, v, att, nseq, H, NH);
    ref_linear(lt(g, l, "attn_out.weight"), H, H, lt(g, l, "attn_out.bias"), att, ob,
               nseq);
    for (size_t i = 0; i < nseq * H; ++i) tok[i] += ob[i];

    for (size_t t = 0; t < nseq; ++t)
      ref_ln(nrm + t * H, tok + t * H, lt(g, l, "ln2.weight"), lt(g, l, "ln2.bias"),
             H, eps);
    ref_linear(lt(g, l, "ffn_up.weight"), INTER, H, lt(g, l, "ffn_up.bias"), nrm, f1,
               nseq);
    for (size_t i = 0; i < nseq * INTER; ++i) f1[i] = ref_gelu_quick(f1[i]);
    ref_linear(lt(g, l, "ffn_down.weight"), H, INTER, lt(g, l, "ffn_down.bias"), f1,
               ob, nseq);
    for (size_t i = 0; i < nseq * H; ++i) tok[i] += ob[i];
  }

  const float* postw = td(g, "v.post_ln.weight");
  if (postw)
    for (size_t t = 0; t < nseq; ++t)
      ref_ln(tok + t * H, tok + t * H, postw, td(g, "v.post_ln.bias"), H, eps);

  ref_linear(td(g, "mm.0.weight"), proj0, H, td(g, "mm.0.bias"), tok + cls * H, mm0,
             np);
  for (size_t i = 0; i < np * proj0; ++i) mm0[i] = ref_gelu_tanh(mm0[i]);
  if (gguf_tensor(g, "mm.2.weight"))
    ref_linear(td(g, "mm.2.weight"), PROJ, proj0, td(g, "mm.2.bias"), mm0, out, np);
  else
    memcpy(out, mm0, np * PROJ * sizeof(float));

  free(patches);
  free(tok);
  free(nrm);
  free(q);
  free(k);
  free(v);
  free(att);
  free(ob);
  free(f1);
  free(mm0);
}

/* ---- fixture + comparison -------------------------------------------------- */

#define VS 8    /* image_size */
#define VP 2    /* patch_size -> 4x4 = 16 patches (>=8 exercises the GEMM path) */
#define VH 8    /* hidden */
#define VNH 4   /* heads -> head_dim 2; >=4 so oc_parallel_for actually dispatches
                 * the attention across pool threads (n<4 stays serial) */
#define VNL 2   /* layers */
#define VINT 16 /* ffn width */
#define VPROJ 8 /* projector output == LM hidden */

static void run_case(int full) {
  size_t grid = VS / VP, np = grid * grid, pdim = 3 * VP * VP;
  size_t nseq = np + (full ? 1u : 0u);
  size_t proj0 = full ? 16u : (size_t)VPROJ; /* distinct proj0 catches a stride bug */

  GgufB m = {0};
  kv_str(&m, "general.architecture", "clip");
  kv_u32(&m, "clip.vision.image_size", VS);
  kv_u32(&m, "clip.vision.patch_size", VP);
  kv_u32(&m, "clip.vision.embedding_length", VH);
  kv_u32(&m, "clip.vision.attention.head_count", VNH);
  kv_u32(&m, "clip.vision.block_count", VNL);
  kv_u32(&m, "clip.vision.feed_forward_length", VINT);
  kv_u32(&m, "clip.vision.projection_dim", VPROJ);
  kv_f32(&m, "clip.vision.attention.layer_norm_epsilon", 1e-5f);

  tsr(&m, "v.patch_embd.weight", VH, pdim, 0.0f, 0.4f);
  if (full) tsr(&m, "v.patch_embd.bias", 0, VH, 0.0f, 0.1f);
  if (full) tsr(&m, "v.class_embd", 0, VH, 0.0f, 0.3f);
  tsr(&m, "v.position_embd.weight", nseq, VH, 0.0f, 0.2f);
  if (full) {
    tsr(&m, "v.pre_ln.weight", 0, VH, 1.0f, 0.1f);
    tsr(&m, "v.pre_ln.bias", 0, VH, 0.0f, 0.1f);
  }
  /* tsr() stores the name pointer, not a copy, so each name needs its own
   * persistent storage (mirrors test_model.c's per-layer name table). */
  static char names[VNL][20][56];
  for (size_t l = 0; l < VNL; ++l) {
    char(*nm)[56] = names[l];
    size_t i = 0;
#define NAME(suf) (snprintf(nm[i], 56, "v.blk.%zu." suf, l), nm[i++])
    tsr(&m, NAME("ln1.weight"), 0, VH, 1.0f, 0.1f);
    tsr(&m, NAME("ln2.weight"), 0, VH, 1.0f, 0.1f);
    tsr(&m, NAME("attn_q.weight"), VH, VH, 0.0f, 0.4f);
    tsr(&m, NAME("attn_k.weight"), VH, VH, 0.0f, 0.4f);
    tsr(&m, NAME("attn_v.weight"), VH, VH, 0.0f, 0.4f);
    tsr(&m, NAME("attn_out.weight"), VH, VH, 0.0f, 0.4f);
    tsr(&m, NAME("ffn_up.weight"), VINT, VH, 0.0f, 0.4f);
    tsr(&m, NAME("ffn_down.weight"), VH, VINT, 0.0f, 0.3f);
    if (full) {
      tsr(&m, NAME("ln1.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("ln2.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("attn_q.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("attn_k.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("attn_v.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("attn_out.bias"), 0, VH, 0.0f, 0.1f);
      tsr(&m, NAME("ffn_up.bias"), 0, VINT, 0.0f, 0.1f);
      tsr(&m, NAME("ffn_down.bias"), 0, VH, 0.0f, 0.1f);
    }
#undef NAME
  }
  if (full) {
    tsr(&m, "v.post_ln.weight", 0, VH, 1.0f, 0.1f);
    tsr(&m, "v.post_ln.bias", 0, VH, 0.0f, 0.1f);
  }
  tsr(&m, "mm.0.weight", proj0, VH, 0.0f, 0.4f);
  if (full) tsr(&m, "mm.0.bias", 0, proj0, 0.0f, 0.1f);
  if (full) {
    tsr(&m, "mm.2.weight", VPROJ, proj0, 0.0f, 0.4f);
    tsr(&m, "mm.2.bias", 0, VPROJ, 0.0f, 0.1f);
  }

  size_t len = 0;
  uint8_t* blob = build(&m, &len);
  GgufFile g;
  char err[256];
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);

  VisionEncoder* v = NULL;
  if (vision_load(&v, &g, err, sizeof err) != 0) {
    fprintf(stderr, "vision_load failed: %s\n", err);
    CHECK(0);
  }
  const VisionConfig* C = vision_config(v);
  CHECK(C->n_patches == np && C->n_positions == nseq);
  CHECK(C->proj_dim == (size_t)VPROJ && C->hidden == VH && C->n_layer == VNL);
  CHECK((int)C->has_class_token == full);

  float* img = malloc(3 * VS * VS * sizeof(float));
  CHECK(img != NULL);
  for (size_t i = 0; i < 3 * VS * VS; ++i) img[i] = rndf();

  size_t nt = 0, dim = 0;
  float* got = vision_encode(v, img, &nt, &dim, err, sizeof err);
  CHECK(got != NULL);
  CHECK(nt == np && dim == (size_t)VPROJ);

  float* want = malloc(np * VPROJ * sizeof(float));
  CHECK(want != NULL);
  ref_forward(&g, img, VS, VP, VH, VNH, VNL, VINT, proj0, VPROJ, want);

  float maxdiff = 0.0f;
  for (size_t i = 0; i < np * VPROJ; ++i) {
    float d = fabsf(got[i] - want[i]);
    if (d > maxdiff) maxdiff = d;
  }
  CHECK(maxdiff < 1e-3f);

  free(want);
  free(got);
  free(img);
  vision_free(v);
  gguf_close(&g);
  free(blob);
  printf("ok vision %s tower (max|diff| = %.2e over %zu values)\n",
         full ? "full-CLIP" : "min-SigLIP", (double)maxdiff, np * (size_t)VPROJ);
}

void test_vision(void) {
  oc_pool_init(2);
  run_case(1);
  run_case(0);
}
