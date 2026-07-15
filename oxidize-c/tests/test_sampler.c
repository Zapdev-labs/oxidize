/* Sampler tests: determinism is asserted bit-for-bit against golden token
 * sequences captured from the pre-refactor (malloc+full-qsort) implementation,
 * so the scratch-buffer + partial-top-k rewrite provably did not move any
 * number. Greedy is asserted to allocate nothing. */
#include <math.h>

#include "../src/sampler.h"
#include "tests.h"

/* Same fixed, distinct logit vector the golden capture used. */
static void fill_logits(float* lg) {
  for (int i = 0; i < 64; ++i)
    lg[i] = (float)((i * 37) % 64) * 0.1f - 3.0f + (float)i * 0.001f;
}

/* Golden sequences from the old sampler (12 tokens each). */
static void test_sampler_determinism(void) {
  float lg[64];
  fill_logits(lg);
  struct {
    SamplerConfig s;
    int32_t want[12];
    uint64_t rng_end;
  } cases[] = {
      {{0.8f, 8, 0.9f, 0x123456789ABCDEFull, 0.0f, 1.0f, 0.0f, 0.0f, NULL, 0},
       {57, 50, 12, 57, 31, 24, 19, 12, 50, 19, 38, 57},
       0x9331bdf4c1153878ull},
      {{1.0f, 0, 1.0f, 0xDEADBEEFCAFEull, 0.05f, 1.0f, 0.0f, 0.0f, NULL, 0},
       {38, 36, 48, 53, 50, 12, 57, 31, 31, 19, 57, 12},
       0x3b94d4b3affea2f6ull},
      {{0.7f, 0, 1.0f, 0x9E3779B97F4A7C15ull, 0.0f, 1.0f, 0.0f, 0.0f, NULL, 0},
       {19, 57, 24, 31, 50, 5, 55, 12, 50, 55, 19, 62},
       0x909bda9496fc9645ull},
      {{0.5f, 1, 1.0f, 0x1ull, 0.0f, 1.0f, 0.0f, 0.0f, NULL, 0},
       {19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19},
       0xe5e737f9c348e4cull},
  };
  for (size_t k = 0; k < sizeof cases / sizeof *cases; ++k) {
    SamplerConfig s = cases[k].s;
    for (int t = 0; t < 12; ++t)
      CHECK(sample_token(&s, lg, 64) == cases[k].want[t]);
    CHECK(s.rng == cases[k].rng_end); /* RNG advanced exactly once/token */
    sampler_free(&s);
  }

  /* Two configs with the same seed must produce the same stream. */
  SamplerConfig a = {0.9f, 16, 0.95f, 42, 0.02f, 1.0f, 0.0f, 0.0f, NULL, 0};
  SamplerConfig b = a;
  for (int t = 0; t < 20; ++t)
    CHECK(sample_token(&a, lg, 64) == sample_token(&b, lg, 64));
  sampler_free(&a);
  sampler_free(&b);
  printf("ok sampler determinism (golden bit-for-bit)\n");
}

static void test_sampler_greedy_no_alloc(void) {
  float lg[64];
  fill_logits(lg);
  SamplerConfig s = {0.0f, 0, 1.0f, 7, 0.0f, 1.0f, 0.0f, 0.0f, NULL, 0};
  int32_t first = sample_token(&s, lg, 64);
  for (int t = 0; t < 100; ++t) CHECK(sample_token(&s, lg, 64) == first);
  CHECK(s.scratch == NULL);     /* greedy never touched the scratch buffer */
  CHECK(s.scratch_cap == 0);
  CHECK(first == 19);           /* argmax of the fixed vector */
  printf("ok sampler greedy zero-alloc + argmax\n");
}

static void test_sampler_penalties(void) {
  /* frequency penalty: repeatedly-seen token gets pushed below its rivals. */
  float lg[3] = {4.0f, 3.9f, 3.8f};
  int32_t recent[4] = {0, 0, 0, 0};
  SamplerConfig s = {0.0f, 0, 1.0f, 1, 0.0f, 1.0f, 0.6f, 0.0f, NULL, 0};
  sampler_penalize(&s, lg, 3, recent, 4);
  CHECK(lg[0] < lg[1]); /* 4.0 - 4*0.6 = 1.6 now lowest */
  CHECK(sample_token(&s, lg, 3) != 0);

  /* presence penalty is flat and applied once regardless of count. */
  float lg2[3] = {3.0f, 2.9f, 2.8f};
  int32_t seen[3] = {0, 0, 0};
  SamplerConfig p = {0.0f, 0, 1.0f, 1, 0.0f, 1.0f, 0.0f, 1.0f, NULL, 0};
  sampler_penalize(&p, lg2, 3, seen, 3);
  CHECK(fabsf(lg2[0] - 2.0f) < 1e-6f); /* 3.0 - 1.0 exactly once */
  CHECK(fabsf(lg2[1] - 2.9f) < 1e-6f); /* untouched */
  CHECK(sample_token(&p, lg2, 3) != 0);

  /* both off => no change (defaults). */
  float lg3[2] = {1.0f, 2.0f};
  SamplerConfig off = {0.0f, 0, 1.0f, 1, 0.0f, 1.0f, 0.0f, 0.0f, NULL, 0};
  sampler_penalize(&off, lg3, 2, recent, 4);
  CHECK(lg3[0] == 1.0f && lg3[1] == 2.0f);
  printf("ok sampler frequency + presence penalties\n");
}

void test_sampler(void) {
  test_sampler_determinism();
  test_sampler_greedy_no_alloc();
  test_sampler_penalties();
}
