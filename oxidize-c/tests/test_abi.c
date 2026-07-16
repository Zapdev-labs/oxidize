/* Stable C ABI acceptance test (tests/../src/oxidize.h).
 *
 * Standalone binary (its own main) so it needn't touch the other test harness.
 * It builds a tiny but COMPLETE llama-family GGUF in memory — real weights plus
 * a real tokenizer — writes it to a temp file, and drives it entirely through
 * the public ABI: open -> metadata -> session -> generate with a callback.
 *
 * The committed fixtures under oxidize-core/tests/fixtures are parser fixtures
 * with no architecture, so they cannot be *run*; a runnable model has to be
 * synthesized here (same shape the forward-batch test uses, plus vocab KVs).
 *
 * Assertions the task asks for:
 *   - ox_generate returns 0 and the callback fired;
 *   - an OxModelOptions struct_size mismatch is rejected;
 *   - a bad path yields a clean, non-empty error string.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/oxidize.h"

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

/* ---- minimal GGUF v3 writer (mirrors tests/test_model.c's builder) -------- */

typedef struct {
  uint8_t* b;
  size_t n, cap;
} Buf;

static void put(Buf* z, const void* p, size_t n) {
  if (z->n + n > z->cap) {
    z->cap = (z->n + n) * 2 + 4096;
    z->b = realloc(z->b, z->cap);
    CHECK(z->b != NULL);
  }
  memcpy(z->b + z->n, p, n);
  z->n += n;
}
static void put_u32(Buf* z, uint32_t v) { put(z, &v, 4); }
static void put_u64(Buf* z, uint64_t v) { put(z, &v, 8); }
static void put_f32(Buf* z, float v) { put(z, &v, 4); }
static void put_str(Buf* z, const char* s) {
  put_u64(z, strlen(s));
  put(z, s, strlen(s));
}
static void put_pad32(Buf* z) {
  const uint8_t zero = 0;
  while (z->n % 32) put(z, &zero, 1);
}

/* Metadata accumulates into its own buffer so the header can front-run it with
 * the final key count. */
static size_t g_nkv = 0;
static void kv_u32(Buf* kv, const char* k, uint32_t v) {
  put_str(kv, k); put_u32(kv, 4 /*GGUF_T_U32*/); put_u32(kv, v); g_nkv++;
}
static void kv_f32(Buf* kv, const char* k, float v) {
  put_str(kv, k); put_u32(kv, 6 /*GGUF_T_F32*/); put_f32(kv, v); g_nkv++;
}
static void kv_str(Buf* kv, const char* k, const char* v) {
  put_str(kv, k); put_u32(kv, 8 /*GGUF_T_STRING*/); put_str(kv, v); g_nkv++;
}
static void kv_str_arr(Buf* kv, const char* k, const char** items, size_t n) {
  put_str(kv, k);
  put_u32(kv, 9 /*GGUF_T_ARRAY*/);
  put_u32(kv, 8 /*elem GGUF_T_STRING*/);
  put_u64(kv, n);
  for (size_t i = 0; i < n; ++i) put_str(kv, items[i]);
  g_nkv++;
}

typedef struct {
  const char* name;
  size_t rows, cols; /* rows==0 => 1-D of `cols` values */
  float centre;
} Tsr;

static unsigned g_rs = 1234u;
static float rndf(void) { /* [-1, 1) */
  g_rs = g_rs * 1103515245u + 12345u;
  return (float)((int)((g_rs >> 16) & 0xffff) - 32768) * (1.0f / 32768.0f);
}

/* Serialize header + kv buffer + f32 tensors to a GGUF v3 blob (alignment 32). */
static uint8_t* build(Buf* kv, size_t n_kv, const Tsr* t, size_t n_t, size_t* len_out) {
  Buf z = {NULL, 0, 0};
  put(&z, "GGUF", 4);
  put_u32(&z, 3);
  put_u64(&z, n_t);
  put_u64(&z, n_kv);
  put(&z, kv->b, kv->n);

  uint64_t off = 0;
  for (size_t i = 0; i < n_t; ++i) {
    size_t rows = t[i].rows, cols = t[i].cols;
    size_t nvals = rows ? rows * cols : cols;
    put_str(&z, t[i].name);
    put_u32(&z, rows ? 2 : 1);
    put_u64(&z, cols);
    if (rows) put_u64(&z, rows);
    put_u32(&z, 0 /*GGUF_T_F32 tensor*/);
    put_u64(&z, off);
    off += (nvals * 4 + 31) & ~(uint64_t)31;
  }
  put_pad32(&z);

  for (size_t i = 0; i < n_t; ++i) {
    size_t rows = t[i].rows, cols = t[i].cols;
    size_t nvals = rows ? rows * cols : cols;
    for (size_t k = 0; k < nvals; ++k) put_f32(&z, t[i].centre + 0.1f * rndf());
    put_pad32(&z);
  }
  *len_out = z.n;
  return z.b;
}

/* A tiny runnable llama (H=64, 3 layers, vocab 32) + a 32-token SPM vocab whose
 * pieces cover the prompt. Writes the blob to `path` (from mkstemp). */
static void write_model(const char* path) {
  const size_t H = 64, NL = 3, NH = 4, KVH = 2, HD = 16, FF = 128, V = 32;

  /* 32 tokens: 0=<unk> (a valid fallback id), 1=<s>, 2=</s>, 3..31 = 'a'..'y'.
   * The prompt "hello" is entirely within a..y, so it segments to single-char
   * pieces and never hits byte fallback. */
  static char letters[29][2];
  const char* toks[V];
  toks[0] = "<unk>"; toks[1] = "<s>"; toks[2] = "</s>";
  for (size_t i = 0; i < 29; ++i) { letters[i][0] = (char)('a' + i); letters[i][1] = 0; toks[3 + i] = letters[i]; }

  Buf kv = {NULL, 0, 0};
  g_nkv = 0;
  kv_str(&kv, "general.architecture", "llama");
  kv_u32(&kv, "llama.embedding_length", (uint32_t)H);
  kv_u32(&kv, "llama.block_count", (uint32_t)NL);
  kv_u32(&kv, "llama.attention.head_count", (uint32_t)NH);
  kv_u32(&kv, "llama.attention.head_count_kv", (uint32_t)KVH);
  kv_u32(&kv, "llama.feed_forward_length", (uint32_t)FF);
  kv_u32(&kv, "llama.context_length", 64);
  kv_f32(&kv, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
  kv_f32(&kv, "llama.rope.freq_base", 1e4f);
  kv_u32(&kv, "llama.rope.dimension_count", (uint32_t)HD);
  kv_str_arr(&kv, "tokenizer.ggml.tokens", toks, V);
  kv_u32(&kv, "tokenizer.ggml.unknown_token_id", 0);
  kv_u32(&kv, "tokenizer.ggml.bos_token_id", 1);
  kv_u32(&kv, "tokenizer.ggml.add_bos_token", 0);   /* keep encode simple */
  kv_u32(&kv, "tokenizer.ggml.add_space_prefix", 0); /* no leading U+2581 */
  /* No tokenizer.ggml.model => SPM path (not gpt2). No eos/eot => no early stop,
   * so max_tokens tokens are produced and the callback definitely fires. */

  Tsr t[3 /*global*/ + 3 * 9 /*per layer*/];
  size_t n = 0;
  t[n++] = (Tsr){"token_embd.weight", V, H, 0.0f};
  t[n++] = (Tsr){"output.weight", V, H, 0.0f};
  t[n++] = (Tsr){"output_norm.weight", 0, H, 1.0f};
  static char names[3][9][48];
  for (size_t l = 0; l < NL; ++l) {
    char(*nm)[48] = names[l];
    int i = 0;
#define NAME(suffix) (snprintf(nm[i], 48, "blk.%zu." suffix, l), nm[i++])
    t[n++] = (Tsr){NAME("attn_q.weight"), NH * HD, H, 0.0f};
    t[n++] = (Tsr){NAME("attn_k.weight"), KVH * HD, H, 0.0f};
    t[n++] = (Tsr){NAME("attn_v.weight"), KVH * HD, H, 0.0f};
    t[n++] = (Tsr){NAME("attn_output.weight"), H, NH * HD, 0.0f};
    t[n++] = (Tsr){NAME("ffn_gate.weight"), FF, H, 0.0f};
    t[n++] = (Tsr){NAME("ffn_up.weight"), FF, H, 0.0f};
    t[n++] = (Tsr){NAME("ffn_down.weight"), H, FF, 0.0f};
    t[n++] = (Tsr){NAME("attn_norm.weight"), 0, H, 1.0f};
    t[n++] = (Tsr){NAME("ffn_norm.weight"), 0, H, 1.0f};
#undef NAME
  }

  size_t len = 0;
  uint8_t* blob = build(&kv, g_nkv, t, n, &len);
  free(kv.b);

  FILE* f = fopen(path, "wb");
  CHECK(f != NULL);
  CHECK(fwrite(blob, 1, len, f) == len);
  CHECK(fclose(f) == 0);
  free(blob);
}

/* ---- callback ------------------------------------------------------------- */

typedef struct {
  int calls;
  size_t bytes;
  char text[4096];
} Acc;

static int on_tok(const char* piece, size_t len, void* user) {
  Acc* a = user;
  a->calls++;
  if (a->bytes + len < sizeof a->text) {
    memcpy(a->text + a->bytes, piece, len);
    a->bytes += len;
  }
  return 0; /* keep generating */
}

int main(int argc, char** argv) {
  /* argv[1] is the committed parser fixture (no architecture): opening it must
   * fail cleanly, not crash. */
  const char* noarch = argc > 1 ? argv[1]
                                : "../oxidize-core/tests/fixtures/valid-v3.gguf";

  char err[256];
  OxModel* m = NULL;

  /* (1) bad path -> clean error, *out left NULL. */
  err[0] = 0;
  CHECK(ox_model_open(&m, "/no/such/model.gguf", NULL, err, sizeof err) == -1);
  CHECK(m == NULL);
  CHECK(err[0] != 0);
  printf("ok abi bad-path error: %s\n", err);

  /* (2) architecture-less fixture -> clean error. */
  err[0] = 0;
  CHECK(ox_model_open(&m, noarch, NULL, err, sizeof err) == -1);
  CHECK(m == NULL);
  CHECK(err[0] != 0);
  printf("ok abi no-arch fixture error: %s\n", err);

  /* Build a real runnable model. */
  char path[] = "/tmp/oxidize_abi_XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  write_model(path);

  /* (3) OxModelOptions struct_size mismatch -> rejected before any load. */
  err[0] = 0;
  OxModelOptions bad = {.struct_size = 1};
  CHECK(ox_model_open(&m, path, &bad, err, sizeof err) == -1);
  CHECK(m == NULL);
  CHECK(err[0] != 0);
  printf("ok abi struct_size mismatch rejected: %s\n", err);

  /* (4) real open with valid options. */
  OxModelOptions opts = {.struct_size = sizeof opts, .ctx = 0, .threads = 0,
                         .seed = 42, .kv_quant = 0};
  err[0] = 0;
  CHECK(ox_model_open(&m, path, &opts, err, sizeof err) == 0);
  CHECK(m != NULL);

  /* metadata: struct_size gate + populated fields. */
  OxMetadata md_bad = {.struct_size = 7};
  CHECK(ox_metadata(m, &md_bad) == -1);
  OxMetadata md = {.struct_size = sizeof md};
  CHECK(ox_metadata(m, &md) == 0);
  CHECK(strcmp(md.arch, "llama") == 0);
  CHECK(md.vocab == 32);
  CHECK(md.ctx == 64);
  CHECK(md.n_tensors == 3 + 3 * 9);
  CHECK(md.isa && md.isa[0]);
  printf("ok abi metadata: arch=%s vocab=%zu ctx=%zu tensors=%zu isa=%s\n",
         md.arch, md.vocab, md.ctx, md.n_tensors, md.isa);

  /* (5) session + generate: callback must fire, return 0. */
  OxSession* s = ox_session_new(m);
  CHECK(s != NULL);
  ox_session_set_temperature(s, 0.0f); /* greedy: deterministic */
  ox_session_set_repeat_penalty(s, 1.1f);

  Acc acc = {0};
  err[0] = 0;
  CHECK(ox_generate(s, "hello", 8, on_tok, &acc, err, sizeof err) == 0);
  CHECK(acc.calls > 0);
  printf("ok abi generate: %d tokens, %zu bytes streamed\n", acc.calls, acc.bytes);

  /* continuation turn on the same session (pos > 0 path). */
  Acc acc2 = {0};
  err[0] = 0;
  CHECK(ox_generate(s, "hi", 4, on_tok, &acc2, err, sizeof err) == 0);
  CHECK(acc2.calls > 0);
  printf("ok abi multi-turn continue: %d tokens\n", acc2.calls);

  /* (6) interleaved sessions: A then B then A again. Without per-session KV,
   * B would overwrite A's cache and A's continuation would be corrupt / fail.
   * With it, A's pos and KV stay intact across B's turn. */
  OxSession* a = ox_session_new(m);
  OxSession* b = ox_session_new(m);
  CHECK(a != NULL && b != NULL);
  ox_session_set_temperature(a, 0.0f);
  ox_session_set_temperature(b, 0.0f);
  Acc aa = {0}, bb = {0}, aa2 = {0};
  err[0] = 0;
  CHECK(ox_generate(a, "hello", 4, on_tok, &aa, err, sizeof err) == 0);
  CHECK(aa.calls > 0);
  err[0] = 0;
  CHECK(ox_generate(b, "abcd", 4, on_tok, &bb, err, sizeof err) == 0);
  CHECK(bb.calls > 0);
  err[0] = 0;
  CHECK(ox_generate(a, "hi", 4, on_tok, &aa2, err, sizeof err) == 0);
  CHECK(aa2.calls > 0);
  printf("ok abi interleaved sessions: A=%d B=%d A2=%d tokens\n", aa.calls,
         bb.calls, aa2.calls);
  ox_session_free(a);
  ox_session_free(b);

  /* ox_session_reset: after a multi-turn generate, reset must allow a fresh
   * first-turn generate (pos==0 + cleared recent/KV) without context-full. */
  {
    OxSession* r = ox_session_new(m);
    CHECK(r != NULL);
    ox_session_set_temperature(r, 0.0f);
    Acc ar = {0};
    err[0] = 0;
    CHECK(ox_generate(r, "hello", 8, on_tok, &ar, err, sizeof err) == 0);
    CHECK(ar.calls > 0);
    Acc ar2 = {0};
    err[0] = 0;
    CHECK(ox_generate(r, "more", 8, on_tok, &ar2, err, sizeof err) == 0);
    ox_session_reset(r);
    ox_session_reset(NULL); /* safe on NULL */
    Acc ar3 = {0};
    err[0] = 0;
    CHECK(ox_generate(r, "hello", 8, on_tok, &ar3, err, sizeof err) == 0);
    CHECK(ar3.calls > 0);
    printf("ok abi session_reset: before=%d+%d after=%d tokens\n", ar.calls,
           ar2.calls, ar3.calls);
    ox_session_free(r);
  }

  /* NULL prompt is a clean error, not a crash. */
  err[0] = 0;
  CHECK(ox_generate(s, NULL, 4, on_tok, &acc2, err, sizeof err) == -1);
  CHECK(err[0] != 0);

  ox_session_free(s);
  ox_model_close(m);

  CHECK(ox_version() && ox_version()[0]);
  CHECK(ox_isa() && ox_isa()[0]);
  printf("ok abi version=%s isa=%s\n", ox_version(), ox_isa());

  unlink(path);
  printf("all abi tests passed\n");
  return 0;
}
