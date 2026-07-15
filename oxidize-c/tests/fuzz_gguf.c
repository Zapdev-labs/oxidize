/* GGUF parser robustness.
 *
 * oc_gguf_fuzz_one() is the single "parse this blob" entry: it must neither
 * crash nor leak for ANY byte string. Two drivers use it:
 *   - LLVMFuzzerTestOneInput   (`make fuzz`, needs clang)
 *   - test_gguf_corpus()       (`make test`, no clang: a deterministic mutation
 *                               corpus over the valid fixture + hand-built
 *                               pathological blobs)
 * Run `make asan` to make the leak/UB half of "must not crash" actually bite.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gguf.h"
#include "tests.h"

int oc_gguf_fuzz_one(const uint8_t* data, size_t len) {
  GgufFile g;
  char err[256];
  if (gguf_parse(&g, data, len, err, sizeof err) != 0) return 0;
  /* Anything gguf_parse accepts, the loaders will then walk: do it here so a
   * blob that parses but produces a poisoned GgufFile is caught too. */
  free(gguf_architecture(&g));
  free(gguf_get_str(&g, "general.name"));
  uint32_t u;
  float f;
  gguf_get_u32(&g, "general.alignment", &u);
  gguf_get_f32(&g, "general.alignment", &f);
  gguf_get_arr(&g, "tokenizer.ggml.tokens");
  for (size_t i = 0; i < g.n_kv; ++i) {
    const GgufValue* v = gguf_find(&g, g.kvs[i].key);
    if (v && v->kind == GGUF_T_ARRAY)
      for (size_t j = 0; j < v->v.arr.n; ++j) (void)v->v.arr.items[j].kind;
  }
  for (size_t i = 0; i < g.n_tensors; ++i) {
    const GgufTensorInfo* ti = gguf_tensor(&g, g.tensors[i].name);
    CHECK(ti != NULL);
    CHECK(ti->n_dims <= GGUF_MAX_DIMS);
    CHECK(ti->offset <= len);        /* parser promises the offset is in range */
    CHECK(ti->data == data + ti->offset);
  }
  CHECK(g.alignment > 0 && (g.alignment & (g.alignment - 1)) == 0);
  CHECK(g.data_section_start <= len);
  if (g.n_tensors) CHECK(g.data_section_start % g.alignment == 0);
  gguf_close(&g);
  return 1;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t len);
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t len) {
  oc_gguf_fuzz_one(data, len);
  return 0;
}

/* ---- deterministic in-tree corpus ---------------------------------------- */

static uint8_t* slurp(const char* path, size_t* len) {
  FILE* fp = fopen(path, "rb");
  CHECK(fp != NULL);
  CHECK(fseek(fp, 0, SEEK_END) == 0);
  long n = ftell(fp);
  CHECK(n > 0);
  rewind(fp);
  uint8_t* b = malloc((size_t)n);
  CHECK(b != NULL);
  CHECK(fread(b, 1, (size_t)n, fp) == (size_t)n);
  fclose(fp);
  *len = (size_t)n;
  return b;
}

static uint32_t rng = 0x9E3779B9u;
static uint32_t next_rand(void) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

static void put64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

/* Array-of-array: 12 bytes per level, so a file this size could drive
 * rd_value() recursion thousands of frames deep if nesting were allowed. */
static void nested_array_bomb(size_t depth) {
  size_t len = 24 + 9 + 4 + 12 * depth + 13;
  uint8_t* b = calloc(1, len);
  CHECK(b != NULL);
  size_t o = 0;
  memcpy(b + o, "GGUF", 4);
  o += 4;
  b[o] = 3;
  o += 4;              /* version */
  put64(b + o, 0);
  o += 8;              /* tensor count */
  put64(b + o, 1);
  o += 8;              /* kv count */
  put64(b + o, 1);
  o += 8;              /* key length */
  b[o++] = 'x';
  b[o] = GGUF_T_ARRAY;
  o += 4;
  for (size_t d = 0; d < depth; ++d) {
    b[o] = GGUF_T_ARRAY;
    o += 4;            /* element kind */
    put64(b + o, 1);
    o += 8;            /* one element */
  }
  b[o] = GGUF_T_U8;
  o += 4;
  put64(b + o, 1);
  o += 8;
  b[o++] = 0xAA;
  CHECK(o <= len);
  oc_gguf_fuzz_one(b, o);
  free(b);
}

void test_gguf_corpus(const char* fixture) {
  size_t len;
  uint8_t* base = slurp(fixture, &len);
  uint8_t* buf = malloc(len);
  CHECK(buf != NULL);
  size_t accepted = 0, cases = 0;

  CHECK(oc_gguf_fuzz_one(base, len) == 1); /* the fixture itself must parse */

  /* 1. truncation at every length, including 0 (and a NULL-ish empty read) */
  for (size_t n = 0; n <= len; ++n, ++cases) accepted += oc_gguf_fuzz_one(base, n);

  /* 2. every single-byte value at every offset (33k parses on a 132B fixture) */
  for (size_t i = 0; i < len; ++i)
    for (unsigned v = 0; v < 256; ++v, ++cases) {
      memcpy(buf, base, len);
      buf[i] = (uint8_t)v;
      accepted += oc_gguf_fuzz_one(buf, len);
    }

  /* 3. targeted field corruption: the counts, lengths and offsets that decide
   *    how much memory the parser is about to trust. Offsets are for
   *    valid-v3.gguf (1 kv "general.alignment"=64, 1 f32 tensor). */
  static const size_t U64_FIELDS[] = {
      8,    /* tensor count */
      16,   /* metadata count */
      24,   /* kv[0] key length */
      57,   /* tensor[0] name length */
      90,   /* tensor[0] dims[0] */
      98,   /* tensor[0] dims[1] */
      110,  /* tensor[0] data offset (rebased onto data_section_start) */
  };
  static const uint64_t POISON[] = {
      0, 1, 0x7fff, 0x100000, 0x7fffffffffffffffull, 0xfffffffffffffffeull,
      0xffffffffffffffffull, (uint64_t)1 << 62,
  };
  CHECK(len > 110 + 8);
  for (size_t f = 0; f < sizeof U64_FIELDS / sizeof *U64_FIELDS; ++f)
    for (size_t p = 0; p < sizeof POISON / sizeof *POISON; ++p, ++cases) {
      memcpy(buf, base, len);
      put64(buf + U64_FIELDS[f], POISON[p]);
      accepted += oc_gguf_fuzz_one(buf, len);
    }
  /* alignment kv value (u32 at 53): 0, non-powers of two, huge */
  static const uint32_t ALIGNS[] = {0, 1, 3, 5, 0x80000000u, 0xffffffffu};
  for (size_t a = 0; a < sizeof ALIGNS / sizeof *ALIGNS; ++a, ++cases) {
    memcpy(buf, base, len);
    memcpy(buf + 53, &ALIGNS[a], 4);
    accepted += oc_gguf_fuzz_one(buf, len);
  }

  /* 4. random multi-byte mutations + random truncation, fixed seed */
  rng = 0x9E3779B9u;
  for (int it = 0; it < 20000; ++it, ++cases) {
    memcpy(buf, base, len);
    unsigned nmut = 1 + next_rand() % 6;
    for (unsigned m = 0; m < nmut; ++m)
      buf[next_rand() % len] = (uint8_t)next_rand();
    size_t n = 1 + next_rand() % len;
    accepted += oc_gguf_fuzz_one(buf, n);
  }

  /* 5. recursion: a legal-looking file whose only kv is N nested arrays */
  for (size_t d = 1; d <= 65536; d *= 4, ++cases) nested_array_bomb(d);

  free(buf);
  free(base);
  printf("ok gguf corpus: %zu blobs, %zu accepted, no crash/leak\n", cases, accepted);
}
