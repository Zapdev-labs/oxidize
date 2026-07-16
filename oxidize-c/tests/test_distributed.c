/* Acceptance test for pipeline parallelism (src/distributed.c).
 *
 * THE invariant: splitting a model across N ranks that pass the residual stream
 * over a socket must produce logits BIT-IDENTICAL to running the whole model in
 * one process. A wrong layer-range split, a dropped framing byte or an endian
 * slip shows up here as a mismatch. Ranks run in-process over socketpairs (one
 * worker thread each); the head thread also runs the single-process reference
 * and compares every logit with memcmp.
 *
 * Covers head, middle-relay and last-rank paths (nranks = 2 and 3) plus a
 * framing round-trip. The model is a synthetic 6-layer llama built in memory
 * with the same GGUF writer the other model tests use. */
#include <pthread.h>
#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/distributed.h"
#include "../src/model.h"
#include "../src/model_llama.h"
#include "../src/tensor.h"
#include "gguf_build.h"

#define NRANKS_MAX 4

/* Synthetic dense llama: 6 layers so both a 2-way ([0,3)|[3,6)) and a 3-way
 * ([0,2)|[2,4)|[4,6)) split have non-trivial middle slices. */
static uint8_t* llama6_fixture(size_t* len) {
  GgufB m = {{NULL, 0, 0}, 0, {{0}}, 0};
  const size_t H = 64, NL = 6, NH = 4, KVH = 2, FF = 128, V = 32, HD = 16;
  rs = 4321u;
  kv_str(&m, "general.architecture", "llama");
  kv_u32(&m, "llama.embedding_length", H);
  kv_u32(&m, "llama.block_count", NL);
  kv_u32(&m, "llama.attention.head_count", NH);
  kv_u32(&m, "llama.attention.head_count_kv", KVH);
  kv_u32(&m, "llama.feed_forward_length", FF);
  kv_u32(&m, "llama.context_length", 64);
  kv_f32(&m, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
  kv_f32(&m, "llama.rope.freq_base", 1e4f);
  kv_u32(&m, "llama.rope.dimension_count", HD);

  tsr(&m, "token_embd.weight", V, H, 0.0f, 0.2f);
  tsr(&m, "output.weight", V, H, 0.0f, 0.2f); /* untied head */
  tsr(&m, "output_norm.weight", 0, H, 1.0f, 0.1f);
  static char names[6][12][48];
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
#undef NAME
  }
  return build(&m, len);
}

static void load_inst(const uint8_t* blob, size_t len, Model* out) {
  char err[256] = {0};
  GgufFile g;
  CHECK(gguf_parse(&g, blob, len, err, sizeof err) == 0);
  CHECK(model_load(out, &g, 0, false, err, sizeof err) == 0);
  CHECK(out->family == MODEL_LLAMA);
}

static void* worker_main(void* arg) {
  OcPipe* p = (OcPipe*)arg;
  char err[256] = {0};
  int rc = oc_pipe_serve(p, err, sizeof err);
  if (rc != 0) fprintf(stderr, "FAIL serve rank %d: %s\n", p->rank, err);
  return (void*)(intptr_t)rc;
}

/* A length-prefixed float frame must survive the socket byte-for-byte (including
 * -0.0), and an oversize frame must be refused rather than overrun the buffer. */
static void test_framing(void) {
  int sv[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  float snd[5] = {1.5f, -2.25f, 0.0f, 3.14159f, -0.0f};
  char err[256] = {0};
  CHECK(oc_dist_send(sv[0], OC_DIST_HIDDEN, 42, snd, 5, err, sizeof err) == 0);
  uint32_t kind, n;
  uint64_t pos;
  float rcv[8];
  CHECK(oc_dist_recv(sv[1], &kind, &pos, rcv, 8, &n, err, sizeof err) == 0);
  CHECK(kind == OC_DIST_HIDDEN && pos == 42 && n == 5);
  CHECK(memcmp(snd, rcv, 5 * sizeof(float)) == 0);

  /* a frame larger than the receiver's capacity is rejected */
  CHECK(oc_dist_send(sv[0], OC_DIST_LOGITS, 0, snd, 5, err, sizeof err) == 0);
  uint32_t k2, n2;
  uint64_t p2;
  float small[2];
  CHECK(oc_dist_recv(sv[1], &k2, &p2, small, 2, &n2, err, sizeof err) == -1);

  close(sv[0]);
  close(sv[1]);
  printf("ok distributed framing round-trip + capacity guard\n");
}

/* Run the fixture split across `nranks` ranks over socketpairs and assert every
 * logit equals the single-process reference, bit for bit. */
static void run_split(const uint8_t* blob, size_t len, int nranks) {
  oc_kv_set_type(OC_KV_F32);
  Model ref;
  load_inst(blob, len, &ref);
  size_t vocab = ref.vocab;
  size_t hidden = ((LlamaModel*)ref.handle)->hidden;
  size_t N = ((LlamaModel*)ref.handle)->n_layers;
  CHECK((int)N >= nranks);

  Model rm[NRANKS_MAX];
  OcPipe pipe[NRANKS_MAX];
  for (int r = 0; r < nranks; ++r) load_inst(blob, len, &rm[r]);

  /* sp[r] joins rank r (next_fd) to rank r+1 (prev_fd). */
  int sp[NRANKS_MAX][2];
  for (int r = 0; r < nranks - 1; ++r)
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sp[r]) == 0);

  for (int r = 0; r < nranks; ++r) {
    memset(&pipe[r], 0, sizeof pipe[r]);
    pipe[r].rank = r;
    pipe[r].nranks = nranks;
    pipe[r].model = &rm[r];
    pipe[r].hidden = hidden;
    pipe[r].vocab = vocab;
    pipe[r].n_layers = N;
    pipe[r].l0 = (size_t)r * N / (size_t)nranks;
    pipe[r].l1 = (size_t)(r + 1) * N / (size_t)nranks;
    pipe[r].prev_fd = r > 0 ? sp[r - 1][1] : -1;
    pipe[r].next_fd = r < nranks - 1 ? sp[r][0] : -1;
  }

  pthread_t th[NRANKS_MAX];
  for (int r = 1; r < nranks; ++r)
    CHECK(pthread_create(&th[r], NULL, worker_main, &pipe[r]) == 0);

  const size_t N_TOK = 8;
  int32_t ids[8];
  for (size_t i = 0; i < N_TOK; ++i) ids[i] = (int32_t)((i * 5 + 2) % vocab);
  float* lp = malloc(vocab * sizeof(float));
  CHECK(lp != NULL);
  char err[256] = {0};
  for (size_t i = 0; i < N_TOK; ++i) {
    float* lref = ref.forward(ref.handle, ids[i], i, true);
    CHECK(lref != NULL);
    CHECK(oc_pipe_head_step(&pipe[0], ids[i], i, lp, err, sizeof err) == 0);
    if (memcmp(lref, lp, vocab * sizeof(float)) != 0) {
      for (size_t k = 0; k < vocab; ++k)
        if (lref[k] != lp[k]) {
          fprintf(stderr,
                  "FAIL split nranks=%d tok %zu logit %zu: ref %.9g pipe %.9g\n",
                  nranks, i, k, (double)lref[k], (double)lp[k]);
          break;
        }
      CHECK(0);
    }
  }
  CHECK(oc_pipe_head_shutdown(&pipe[0], err, sizeof err) == 0);
  for (int r = 1; r < nranks; ++r) {
    void* rv = NULL;
    pthread_join(th[r], &rv);
    CHECK((intptr_t)rv == 0);
  }

  free(lp);
  for (int r = 0; r < nranks - 1; ++r) {
    close(sp[r][0]);
    close(sp[r][1]);
  }
  for (int r = 0; r < nranks; ++r) model_free(&rm[r]);
  model_free(&ref);
  printf("ok distributed split==single (nranks=%d, %zu layers, bit-identical)\n",
         nranks, N);
}

void test_distributed(void) {
  test_framing();
  size_t len = 0;
  uint8_t* blob = llama6_fixture(&len);
  run_split(blob, len, 2);
  run_split(blob, len, 3);
  free(blob);
}
