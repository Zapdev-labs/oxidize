/* oxidize-c unit tests. `make test` runs this. Assert-based, no framework. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gguf.h"
#include "../src/quant.h"
#include "../src/tensor.h"
#include "../src/tokenizer.h"
#include "tests.h"

static void test_gguf_fixture(const char* path) {
  GgufFile g;
  char err[256];
  CHECK(gguf_open(&g, path, err, sizeof(err)) == 0);
  CHECK(g.version == 3);
  CHECK(g.alignment >= 1);
  CHECK(g.data_section_start % g.alignment == 0);
  gguf_close(&g);
  printf("ok gguf fixture parse (%s)\n", path);
}

static void test_q4_0_roundtrip(void) {
  float src[64], dec[64];
  for (int i = 0; i < 64; ++i) src[i] = (float)(i % 16) * 0.25f - 2.0f;
  uint8_t q[2 * OC_BLK_Q4_0];
  oc_quantize_row_q4_0(src, q, 64);
  CHECK(oc_dequant_row(OC_Q4_0, q, dec, 64) == 0);
  for (int i = 0; i < 64; ++i) CHECK(fabsf(dec[i] - src[i]) < 0.2f);
  /* dot must match dequant+dot */
  float x[64];
  for (int i = 0; i < 64; ++i) x[i] = 0.1f * (float)i;
  float d1 = oc_dot_row(OC_Q4_0, q, x, 64);
  float d2 = oc_dot_f32(dec, x, 64);
  CHECK(fabsf(d1 - d2) < 1e-3f * (1.0f + fabsf(d2)));
  printf("ok q4_0 roundtrip\n");
}

static void test_q4_k_dot(void) {
  /* pseudo-random Q4_K block; dot must match dequant+dot */
  uint8_t blk[OC_BLK_Q4_K];
  unsigned s = 777;
  for (size_t i = 0; i < sizeof blk; ++i) {
    s = s * 1103515245u + 12345u;
    blk[i] = (uint8_t)(s >> 16);
  }
  uint16_t d = oc_f32_to_f16(0.02f), mn = oc_f32_to_f16(0.01f);
  blk[0] = (uint8_t)(d & 0xff);
  blk[1] = (uint8_t)(d >> 8);
  blk[2] = (uint8_t)(mn & 0xff);
  blk[3] = (uint8_t)(mn >> 8);
  float dec[OC_QK_K], x[OC_QK_K];
  CHECK(oc_dequant_row(OC_Q4_K, blk, dec, OC_QK_K) == 0);
  for (int i = 0; i < OC_QK_K; ++i) x[i] = 0.4f - 0.007f * (float)i;
  float d1 = oc_dot_row(OC_Q4_K, blk, x, OC_QK_K);
  float d2 = oc_dot_f32(dec, x, OC_QK_K);
  CHECK(fabsf(d1 - d2) < 1e-2f * (1.0f + fabsf(d2)));
  printf("ok q4_k dot\n");
}

static void test_al5xs(void) {
  uint8_t codes[32], back[32], qs[12];
  for (int i = 0; i < 32; ++i) codes[i] = (uint8_t)((i * 5 + 3) & 7);
  al5xs_pack(codes, qs);
  al5xs_unpack(qs, back);
  for (int i = 0; i < 32; ++i) CHECK(back[i] == codes[i]);
  /* full block dequant: f16 scale 2.0 + codes */
  uint8_t blk[OC_BLK_AL5_XS];
  uint16_t sc = oc_f32_to_f16(0.25f);
  blk[0] = (uint8_t)(sc & 0xff);
  blk[1] = (uint8_t)(sc >> 8);
  memcpy(blk + 2, qs, 12);
  float out[32];
  CHECK(oc_dequant_row(OC_AL5_XS, blk, out, 32) == 0);
  for (int i = 0; i < 32; ++i)
    CHECK(fabsf(out[i] - ((float)codes[i] - 4.0f) * 0.25f) < 1e-3f);
  /* dot matches dequant+dot */
  float x[32];
  for (int i = 0; i < 32; ++i) x[i] = 0.5f - 0.03f * (float)i;
  CHECK(fabsf(oc_dot_row(OC_AL5_XS, blk, x, 32) - oc_dot_f32(out, x, 32)) < 1e-3f);
  printf("ok al5xs pack/unpack + dequant\n");
}

static void test_al5xs_encode(void) {
  unsigned seed = 12345;
  for (int trial = 0; trial < 100; ++trial) {
    float w[32];
    for (int i = 0; i < 32; ++i) {
      seed = seed * 1103515245u + 12345u;
      w[i] = ((float)((seed >> 16) & 0x7fff) / 16384.0f - 1.0f) * 0.3f;
    }
    uint8_t blk[OC_BLK_AL5_XS];
    oc_al5xs_encode_block(w, blk);
    float d = oc_f16_to_f32((uint16_t)(blk[0] | (uint16_t)blk[1] << 8));
    CHECK(isfinite(d));
    float dec[32];
    CHECK(oc_dequant_row(OC_AL5_XS, blk, dec, 32) == 0);
    float mse_opt = 0.0f, mse_naive = 0.0f, amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      if (fabsf(w[i]) > amax) amax = fabsf(w[i]);
    float dn = amax / 4.0f;
    for (int i = 0; i < 32; ++i) {
      float eo = w[i] - dec[i];
      int q = (int)lrintf(w[i] / dn + 4.0f);
      if (q < 0) q = 0;
      if (q > 7) q = 7;
      float en = w[i] - ((float)q - 4.0f) * dn;
      mse_opt += eo * eo;
      mse_naive += en * en;
    }
    CHECK(mse_opt < mse_naive); /* refined scale strictly beats max/4 */
  }
  /* all-zero block: scale 0, codes all 4 */
  float z[32] = {0};
  uint8_t blk[OC_BLK_AL5_XS], codes[32];
  oc_al5xs_encode_block(z, blk);
  CHECK(blk[0] == 0 && blk[1] == 0);
  al5xs_unpack(blk + 2, codes);
  for (int i = 0; i < 32; ++i) CHECK(codes[i] == 4);
  printf("ok al5xs encode\n");
}

static void test_rotoquant(void) {
  /* FHT: self-inverse, orthonormal (dot invariance) */
  unsigned seed = 777;
  float a[256], b[256], a0[256], b0[256];
  for (int i = 0; i < 256; ++i) {
    seed = seed * 1103515245u + 12345u;
    a[i] = ((float)((seed >> 16) & 0x7fff) / 16384.0f - 1.0f);
    seed = seed * 1103515245u + 12345u;
    b[i] = ((float)((seed >> 16) & 0x7fff) / 16384.0f - 1.0f) * 2.0f;
  }
  memcpy(a0, a, sizeof(a));
  memcpy(b0, b, sizeof(b));
  float dot0 = oc_dot_f32(a, b, 256);
  oc_fht(a, 256);
  oc_fht(b, 256);
  CHECK(fabsf(oc_dot_f32(a, b, 256) - dot0) < 1e-3f * (1.0f + fabsf(dot0)));
  oc_fht(a, 256);
  for (int i = 0; i < 256; ++i) CHECK(fabsf(a[i] - a0[i]) < 1e-4f);
  /* int4 codec: round-trip error bounded by half a quant step */
  uint8_t q[128];
  float meta[2], dec[256];
  oc_kvq_encode(b, 256, q, meta);
  oc_kvq_decode(q, 256, meta, dec);
  for (int i = 0; i < 256; ++i)
    CHECK(fabsf(dec[i] - b[i]) <= 0.5f * meta[0] + 1e-5f);
  /* constant row: scale 0, exact */
  float c[8] = {3, 3, 3, 3, 3, 3, 3, 3}, cd[8];
  oc_kvq_encode(c, 8, q, meta);
  oc_kvq_decode(q, 8, meta, cd);
  for (int i = 0; i < 8; ++i) CHECK(fabsf(cd[i] - 3.0f) < 1e-6f);
  printf("ok rotoquant fht + int4 codec\n");
}

static void test_rmsnorm(void) {
  float x[4] = {1, 2, 3, 4};
  float w[4] = {1, 1, 1, 1};
  float out[4];
  oc_rms_norm(out, x, w, 4, 1e-6f);
  float rms = sqrtf((1 + 4 + 9 + 16) / 4.0f + 1e-6f);
  for (int i = 0; i < 4; ++i) CHECK(fabsf(out[i] - x[i] / rms) < 1e-5f);
  printf("ok rmsnorm\n");
}

static void test_softmax(void) {
  float x[3] = {1.0f, 2.0f, 3.0f};
  oc_softmax(x, 3);
  float sum = x[0] + x[1] + x[2];
  CHECK(fabsf(sum - 1.0f) < 1e-5f);
  CHECK(x[2] > x[1] && x[1] > x[0]);
  printf("ok softmax\n");
}

static void test_matvec(void) {
  oc_pool_init(2);
  /* 3x4 f32 matrix */
  float W[12] = {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  float x[4] = {1, 2, 3, 4};
  float y[3];
  OcCtx* c = oc_ctx_new();
  CHECK(c != NULL);
  oc_matvec(c, y, OC_F32, (const uint8_t*)W, 3, 4, x);
  CHECK(fabsf(y[0] - 1) < 1e-6f && fabsf(y[1] - 2) < 1e-6f && fabsf(y[2] - 10) < 1e-6f);

  /* and the same matrix as a 2-token batch: Y is [n_tokens][rows] */
  float x2[8] = {1, 2, 3, 4, 0, 0, 1, 0};
  float y2[6];
  oc_matmul(c, y2, OC_F32, (const uint8_t*)W, 3, 4, x2, 2);
  CHECK(fabsf(y2[0] - 1) < 1e-6f && fabsf(y2[1] - 2) < 1e-6f && fabsf(y2[2] - 10) < 1e-6f);
  CHECK(fabsf(y2[3] - 0) < 1e-6f && fabsf(y2[4] - 0) < 1e-6f && fabsf(y2[5] - 1) < 1e-6f);
  oc_ctx_free(c);
  printf("ok matvec + matmul\n");
}

/* ---- tokenizer -------------------------------------------------------------
 * The vocabs below are built into in-memory GGUF blobs, so the tokenizer sees
 * exactly the metadata layout it would see in a real model file. */
static void append(uint8_t** buf, size_t* len, const void* p, size_t n) {
  *buf = realloc(*buf, *len + n);
  CHECK(*buf != NULL);
  memcpy(*buf + *len, p, n);
  *len += n;
}
static void app_u32(uint8_t** b, size_t* l, uint32_t v) { append(b, l, &v, 4); }
static void app_u64(uint8_t** b, size_t* l, uint64_t v) { append(b, l, &v, 8); }
static void app_str(uint8_t** b, size_t* l, const char* s) {
  app_u64(b, l, strlen(s));
  append(b, l, s, strlen(s));
}

/* Vocab-only GGUF: tokens, scores (= piece length, so the Viterbi prefers the
 * longer piece), optional token_type array, optional tokenizer model name. */
static uint8_t* build_vocab_gguf(const char** pieces, size_t np, const char* model,
                                 const int32_t* types, size_t* out_len) {
  uint8_t* buf = NULL;
  size_t len = 0;
  uint64_t n_kv = 3 + (model ? 1u : 0u) + (types ? 1u : 0u);
  append(&buf, &len, "GGUF", 4);
  app_u32(&buf, &len, 3); /* version */
  app_u64(&buf, &len, 0); /* tensor count */
  app_u64(&buf, &len, n_kv);

  app_str(&buf, &len, "tokenizer.ggml.tokens");
  app_u32(&buf, &len, GGUF_T_ARRAY);
  app_u32(&buf, &len, GGUF_T_STRING);
  app_u64(&buf, &len, np);
  for (size_t i = 0; i < np; ++i) app_str(&buf, &len, pieces[i]);

  app_str(&buf, &len, "tokenizer.ggml.scores");
  app_u32(&buf, &len, GGUF_T_ARRAY);
  app_u32(&buf, &len, GGUF_T_F32);
  app_u64(&buf, &len, np);
  for (size_t i = 0; i < np; ++i) {
    float s = (float)strlen(pieces[i]);
    append(&buf, &len, &s, 4);
  }

  if (types) {
    app_str(&buf, &len, "tokenizer.ggml.token_type");
    app_u32(&buf, &len, GGUF_T_ARRAY);
    app_u32(&buf, &len, GGUF_T_I32);
    app_u64(&buf, &len, np);
    for (size_t i = 0; i < np; ++i) app_u32(&buf, &len, (uint32_t)types[i]);
  }

  app_str(&buf, &len, "tokenizer.ggml.add_bos_token");
  app_u32(&buf, &len, GGUF_T_BOOL);
  uint8_t z = 0;
  append(&buf, &len, &z, 1);

  if (model) {
    app_str(&buf, &len, "tokenizer.ggml.model");
    app_u32(&buf, &len, GGUF_T_STRING);
    app_str(&buf, &len, model);
  }
  *out_len = len;
  return buf;
}

static void detok(const Tokenizer* t, const int32_t* ids, size_t n, char* out,
                  size_t cap) {
  size_t w = 0;
  for (size_t i = 0; i < n; ++i) {
    CHECK(w + 8 < cap);
    w += tokenizer_decode_token(t, ids[i], out + w, cap - w - 1);
  }
  out[w] = 0;
}

/* <unk>, "▁", "▁hello", "▁world", a-z, and all 256 <0xXX> byte tokens */
static size_t spm_vocab(const char** pieces) {
  static char letters[26][2];
  static char bytes[256][8];
  size_t np = 0;
  pieces[np++] = "<unk>";
  pieces[np++] = "\xe2\x96\x81";
  pieces[np++] = "\xe2\x96\x81hello";
  pieces[np++] = "\xe2\x96\x81world";
  for (int i = 0; i < 26; ++i) {
    letters[i][0] = (char)('a' + i);
    letters[i][1] = 0;
    pieces[np++] = letters[i];
  }
  for (int i = 0; i < 256; ++i) {
    snprintf(bytes[i], 8, "<0x%02X>", i);
    pieces[np++] = bytes[i];
  }
  return np;
}

static void test_tokenizer_roundtrip(void) {
  const char* pieces[300];
  size_t np = spm_vocab(pieces), len;
  uint8_t* buf = build_vocab_gguf(pieces, np, NULL, NULL, &len);

  GgufFile g;
  char err[256];
  CHECK(gguf_parse(&g, buf, len, err, sizeof(err)) == 0);
  Tokenizer t;
  CHECK(tokenizer_init(&t, &g) == 0);
  CHECK(t.n_vocab == np);

  size_t n = 0;
  int32_t* ids = tokenizer_encode(&t, "hello world", false, &n);
  CHECK(ids && n == 2);
  CHECK(ids[0] == 2 && ids[1] == 3); /* ▁hello, ▁world */

  char out[256];
  detok(&t, ids, n, out, sizeof out);
  CHECK(strcmp(out, " hello world") == 0); /* leading ▁ decodes to space */
  free(ids);

  /* plain ascii falls back to letter pieces */
  ids = tokenizer_encode(&t, "abc", false, &n);
  CHECK(ids && n >= 1);
  detok(&t, ids, n, out, sizeof out);
  CHECK(strstr(out, "abc") != NULL);
  free(ids);

  tokenizer_free(&t);
  gguf_close(&g);
  free(buf);
  printf("ok tokenizer roundtrip\n");
}

/* Byte fallback, unicode, empty string. add_space_prefix defaults on, so every
 * round-trip gains exactly one leading space. */
static void test_tokenizer_spm_bytes(void) {
  const char* pieces[300];
  size_t np = spm_vocab(pieces), len;
  uint8_t* buf = build_vocab_gguf(pieces, np, NULL, NULL, &len);
  GgufFile g;
  char err[256];
  CHECK(gguf_parse(&g, buf, len, err, sizeof(err)) == 0);
  Tokenizer t;
  CHECK(tokenizer_init(&t, &g) == 0);
  CHECK(t.add_space_prefix);

  static const char* texts[] = {
      "",                          /* empty */
      "\xc3\xa9",                  /* é: 2-byte utf-8, no piece -> <0xC3><0xA9> */
      "h\xc3\xa9llo w\xc3\xb6rld", /* mixed ascii + 2-byte */
      "\xe2\x82\xac",              /* €: 3-byte */
      "\xf0\x9f\x8e\x89",          /* 🎉: 4-byte (utf8_len == 4 path) */
      "a\xf0\x9f\x8e\x89z",        /* fallback wedged between real pieces */
      "\x01\x02\x7f",              /* raw control bytes */
  };
  char out[256], want[256];
  for (size_t k = 0; k < sizeof texts / sizeof *texts; ++k) {
    size_t n = 0;
    int32_t* ids = tokenizer_encode(&t, texts[k], false, &n);
    CHECK(ids != NULL);
    for (size_t i = 0; i < n; ++i) CHECK(ids[i] >= 0 && (size_t)ids[i] < t.n_vocab);
    detok(&t, ids, n, out, sizeof out);
    snprintf(want, sizeof want, " %s", texts[k]); /* the add_space_prefix ▁ */
    if (strcmp(out, want) != 0) {
      fprintf(stderr, "FAIL spm roundtrip [%zu]: got \"%s\" want \"%s\"\n", k, out,
              want);
      exit(1);
    }
    free(ids);
  }

  /* the fallback really emits <0xXX> byte tokens, not unk */
  size_t n = 0;
  int32_t* ids = tokenizer_encode(&t, "\xc3\xa9", false, &n);
  CHECK(n == 3); /* ▁ + <0xC3> + <0xA9> */
  CHECK(ids[1] == tokenizer_piece_id(&t, "<0xC3>", 6));
  CHECK(ids[2] == tokenizer_piece_id(&t, "<0xA9>", 6));
  CHECK(ids[1] != (int32_t)t.unk_id && ids[2] != (int32_t)t.unk_id);
  free(ids);

  tokenizer_free(&t);
  gguf_close(&g);
  free(buf);
  printf("ok tokenizer spm byte-fallback + unicode + empty\n");
}

/* GPT-2 byte-level BPE. The byte->codepoint table is rebuilt here from the
 * spec, independently of tokenizer.c, and the vocab is generated from it. */
static void test_tokenizer_bpe(void) {
  static char bpe[256][8];
  const char* pieces[300];
  int32_t types[300];
  uint16_t b2c[256];
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                    (b >= 0xAE && b <= 0xFF);
    b2c[b] = printable ? (uint16_t)b : (uint16_t)(256 + extra++);
  }
  CHECK(b2c[' '] == 0x120); /* GPT-2's "Ġ" */
  size_t np = 0;
  for (int b = 0; b < 256; ++b) {
    unsigned cp = b2c[b];
    size_t w = 0;
    if (cp < 0x80) {
      bpe[b][w++] = (char)cp;
    } else {
      bpe[b][w++] = (char)(0xC0 | (cp >> 6));
      bpe[b][w++] = (char)(0x80 | (cp & 0x3F));
    }
    bpe[b][w] = 0;
    pieces[np] = bpe[b];
    types[np++] = 1;
  }
  pieces[np] = "hello";
  types[np++] = 1;
  pieces[np] = "\xc4\xa0world"; /* Ġworld */
  types[np++] = 1;
  pieces[np] = "<|endoftext|>";
  types[np++] = 3; /* control: must decode to nothing */
  size_t len;
  uint8_t* buf = build_vocab_gguf(pieces, np, "gpt2", types, &len);

  GgufFile g;
  char err[256];
  CHECK(gguf_parse(&g, buf, len, err, sizeof(err)) == 0);
  Tokenizer t;
  CHECK(tokenizer_init(&t, &g) == 0);
  CHECK(t.is_bpe && !t.add_space_prefix);
  for (int b = 0; b < 256; ++b) CHECK(t.byte_to_cp[b] == b2c[b]);

  /* greedy longest match picks the merged pieces */
  size_t n = 0;
  int32_t* ids = tokenizer_encode(&t, "hello world", false, &n);
  CHECK(ids && n == 2);
  CHECK(ids[0] == (int32_t)(np - 3) && ids[1] == (int32_t)(np - 2));
  char out[256];
  detok(&t, ids, n, out, sizeof out);
  CHECK(strcmp(out, "hello world") == 0);
  free(ids);

  /* arbitrary byte strings round-trip through the single-byte pieces */
  static const char* texts[] = {"", "\xc3\xa9", "hi\nthere", "\xf0\x9f\x8e\x89",
                                " leading and trailing "};
  for (size_t k = 0; k < sizeof texts / sizeof *texts; ++k) {
    ids = tokenizer_encode(&t, texts[k], false, &n);
    CHECK(ids != NULL);
    detok(&t, ids, n, out, sizeof out);
    if (strcmp(out, texts[k]) != 0) {
      fprintf(stderr, "FAIL bpe roundtrip [%zu]: got \"%s\" want \"%s\"\n", k, out,
              texts[k]);
      exit(1);
    }
    free(ids);
  }

  /* control token decodes to zero bytes; out-of-range ids are refused */
  char one[8];
  CHECK(tokenizer_decode_token(&t, (int32_t)(np - 1), one, sizeof one) == 0);
  CHECK(tokenizer_decode_token(&t, -1, one, sizeof one) == 0);
  CHECK(tokenizer_decode_token(&t, (int32_t)np, one, sizeof one) == 0);

  tokenizer_free(&t);
  gguf_close(&g);
  free(buf);
  printf("ok tokenizer bpe (byte-level, control token, roundtrip)\n");
}

int main(int argc, char** argv) {
  const char* fixture = argc > 1 ? argv[1]
                                 : "../oxidize-core/tests/fixtures/valid-v3.gguf";
  test_gguf_fixture(fixture);
  test_gguf_corpus(fixture);
  test_quant_golden();
  test_quant_dot();
  test_matmul();
  test_quant_q8_act();
  test_q4_0_roundtrip();
  test_q4_k_dot();
  test_al5xs();
  test_al5xs_encode();
  test_rotoquant();
  test_rmsnorm();
  test_softmax();
  test_matvec();
  test_forward_batch();
  test_sampler();
  test_tokenizer_roundtrip();
  test_tokenizer_spm_bytes();
  test_tokenizer_bpe();
  test_tools();
  test_vision();
  test_train();
  test_distributed();
  oc_pool_free();
  printf("all tests passed\n");
  return 0;
}
