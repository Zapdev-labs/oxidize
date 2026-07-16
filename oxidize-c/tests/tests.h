/* Shared test harness: assert macro + the suites main() runs. */
#ifndef OC_TESTS_H
#define OC_TESTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

/* tests/test_quant.c */
void test_quant_golden(void);
void test_quant_dot(void);
void test_matmul(void);
void test_quant_q8_act(void);

/* tests/test_sampler.c — determinism (bit-for-bit vs the pre-refactor sampler)
 * + greedy allocates nothing + frequency/presence penalties. */
void test_sampler(void);

/* tests/test_model.c — batched prefill must equal sequential decode. This is
 * the acceptance test for the causal mask, the SWA ring and the DeltaNet scan;
 * all three fail silently rather than loudly when they are wrong. */
void test_forward_batch(void);

/* tests/test_tools.c — quantize/prune/merge over temp GGUFs; every output is
 * re-opened with gguf_open, so the tools' writer is proven valid. */
void test_tools(void);

/* tests/test_distributed.c — pipeline parallelism: a model split across ranks
 * over a socket must produce logits BIT-IDENTICAL to the single-process run.
 * A wrong layer-range split or a dropped framing byte shows as a mismatch. */
void test_distributed(void);

/* tests/test_vision.c — the CLIP/SigLIP vision tower forward must equal an
 * independent naive reference (own matmul/softmax/gelu/layernorm), over a full
 * CLIP config and a minimal SigLIP one. */
void test_vision(void);

/* tests/test_train.c — gradient checks (analytic vs central finite differences
 * of an independent double reference) for the linear softmax classifier and the
 * LoRA adapter, two convergence runs, and a LoRA GGUF export round-trip. */
void test_train(void);

/* tests/fuzz_gguf.c — one parse attempt; 1 if the blob was accepted. Also the
 * libFuzzer entry point's body, so the corpus runner and the fuzzer exercise
 * exactly the same code. */
int oc_gguf_fuzz_one(const uint8_t* data, size_t len);
/* Deterministic mutation corpus over `fixture` + hand-built pathological
 * blobs. Runs under `make test`; no clang needed. */
void test_gguf_corpus(const char* fixture);

#endif
