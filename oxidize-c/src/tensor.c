#define _GNU_SOURCE
#include "tensor.h"

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "quant.h"

/* ---- NUMA weight replication (Linux) ---------------------------------------
 * One copy of the weight region per node, placed by first-touch from a thread
 * pinned to that node. Workers pin themselves round-robin across nodes on
 * their first job after replication and read their node-local copy. */
#define OC_MAX_NODES 8
#define OC_MAX_CPUS 1024
static struct {
  const uint8_t* base;
  size_t size;
  const uint8_t* rep[OC_MAX_NODES];
  int nodes;                       /* >1 => replication active */
  int cpus[OC_MAX_NODES][OC_MAX_CPUS];
  int ncpu[OC_MAX_NODES];
} g_numa;
static _Thread_local int t_node = 0;
static _Thread_local int t_pinned = 0;

/* Parse "0,2,4-6" style cpulist; returns count. */
static int parse_cpulist(const char* path, int* out, int max) {
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  int n = 0, a, b;
  while (n < max && fscanf(f, "%d", &a) == 1) {
    b = a;
    int c = fgetc(f);
    if (c == '-') {
      if (fscanf(f, "%d", &b) != 1) b = a;
      c = fgetc(f);
    }
    for (int i = a; i <= b && n < max; ++i) out[n++] = i;
    if (c != ',') break;
  }
  fclose(f);
  return n;
}

static int pin_to_cpu(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  return pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

typedef struct {
  int cpu;
  void* dst;
  const void* src;
  size_t sz;
} RepJob;

static void* rep_copy_main(void* arg) {
  RepJob* j = arg;
  pin_to_cpu(j->cpu); /* first-touch places pages on this cpu's node */
  memcpy(j->dst, j->src, j->sz);
  return NULL;
}

void oc_numa_replicate(const void* base, size_t size) {
  if (g_numa.nodes > 1 || !base || !size) return;
  /* Discover nodes with online cpus that are also in our affinity mask. */
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return;
  int nodes = 0;
  for (int n = 0; n < OC_MAX_NODES; ++n) {
    char p[128];
    snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", n);
    int raw[OC_MAX_CPUS];
    int cnt = parse_cpulist(p, raw, OC_MAX_CPUS);
    if (cnt == 0) break;
    int kept = 0;
    for (int i = 0; i < cnt; ++i)
      if (CPU_ISSET(raw[i], &allowed)) g_numa.cpus[nodes][kept++] = raw[i];
    if (kept > 0) {
      g_numa.ncpu[nodes] = kept;
      ++nodes;
    }
  }
  if (nodes < 2) return; /* single node (or numactl-confined): nothing to do */

  pthread_t th[OC_MAX_NODES];
  RepJob jobs[OC_MAX_NODES];
  int made = 0;
  for (int n = 0; n < nodes; ++n) {
    void* dst = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (dst == MAP_FAILED) {
      for (int k = 0; k < made; ++k) munmap((void*)g_numa.rep[k], size);
      return; /* not enough RAM for replicas: run as before */
    }
    g_numa.rep[n] = dst;
    jobs[n] = (RepJob){g_numa.cpus[n][0], dst, base, size};
    pthread_create(&th[n], NULL, rep_copy_main, &jobs[n]);
    ++made;
  }
  for (int n = 0; n < nodes; ++n) pthread_join(th[n], NULL);
  g_numa.base = base;
  g_numa.size = size;
  g_numa.nodes = nodes; /* set last: activates translation */
  fprintf(stderr, "numa: replicated %.1f GiB to %d nodes (ncpu:",
          (double)size / (1024.0 * 1024.0 * 1024.0), nodes);
  for (int n = 0; n < nodes; ++n) fprintf(stderr, " %d", g_numa.ncpu[n]);
  /* verify placement: sample the actual node of a few pages per replica */
  for (int n = 0; n < nodes; ++n) {
    int on0 = -1, mid = -1; /* raw syscall; avoids a libnuma dependency */
    syscall(SYS_get_mempolicy, &on0, NULL, 0, (void*)g_numa.rep[n], 3UL /* F_NODE|F_ADDR */);
    syscall(SYS_get_mempolicy, &mid, NULL, 0, (void*)(g_numa.rep[n] + size / 2), 3UL);
    fprintf(stderr, " rep%d@N%d/N%d", n, on0, mid);
  }
  fprintf(stderr, ")\n");
}

/* Pin worker `idx` round-robin across nodes; remembers its node. */
static void numa_pin_self(int idx) {
  if (g_numa.nodes < 2 || t_pinned) return;
  t_pinned = 1;
  int node = idx % g_numa.nodes;
  int k = (idx / g_numa.nodes) % g_numa.ncpu[node];
  int rc = pin_to_cpu(g_numa.cpus[node][k]);
  if (rc == 0) t_node = node;
  if (getenv("OXC_NUMA_DBG"))
    fprintf(stderr, "numa: worker %d -> node %d cpu %d rc=%d now-on=%d\n", idx,
            node, g_numa.cpus[node][k], rc, sched_getcpu());
}

static inline const uint8_t* numa_local(const uint8_t* W) {
  if (g_numa.nodes > 1 && W >= g_numa.base && W < g_numa.base + g_numa.size)
    return g_numa.rep[t_node] + (W - g_numa.base);
  return W;
}

/* ---- persistent thread pool ------------------------------------------------
 * Generation-counter SPIN barrier: main bumps `gen` (release) and each worker
 * busy-polls it, runs its static chunk of [0,n), and bumps `done`; main spins
 * until done == workers. Spinning (with pause/yield) instead of futex wakes
 * matters here: decode does ~400 dispatches/token and condvar wakeups across
 * two sockets cost more than the matvec chunks themselves. */
#include <stdatomic.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define oc_cpu_relax() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
/* ARM spin-loop hint: yield the pipeline slot without a full memory barrier. */
#define oc_cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define oc_cpu_relax() ((void)0)
#endif

typedef struct {
  pthread_t* threads;
  int n; /* total workers incl. main thread's share? main participates too */
  atomic_ulong gen;
  atomic_int done;
  atomic_int shutdown;
  /* current job (written before the gen bump, read after) */
  oc_range_fn fn;
  void* ctx;
  size_t total;
  size_t step;       /* rows per grab */
  atomic_size_t next; /* work-stealing cursor */
} Pool;

static Pool g_pool;

/* Spin briefly, then yield so idle pools don't burn a whole socket. */
static inline void spin_wait(int* spins) {
  if (++*spins < 4096) oc_cpu_relax();
  else { sched_yield(); *spins = 0; }
}

static void run_chunk(int idx, int nthreads) {
  (void)nthreads;
  numa_pin_self(idx);
  /* Work-stealing: grab `step` rows at a time so a stalled core (IRQ, other
   * socket, monitoring daemon) only delays its current chunk, not 1/N of the
   * matvec. */
  size_t n = g_pool.total, step = g_pool.step;
  for (;;) {
    size_t i0 = atomic_fetch_add_explicit(&g_pool.next, step, memory_order_relaxed);
    if (i0 >= n) break;
    size_t i1 = i0 + step < n ? i0 + step : n;
    g_pool.fn(g_pool.ctx, i0, i1);
  }
}

static void* worker_main(void* arg) {
  int idx = (int)(intptr_t)arg;
  unsigned long seen = 0;
  for (;;) {
    int spins = 0;
    unsigned long g;
    while ((g = atomic_load_explicit(&g_pool.gen, memory_order_acquire)) == seen) {
      if (atomic_load_explicit(&g_pool.shutdown, memory_order_relaxed)) return NULL;
      spin_wait(&spins);
    }
    seen = g;
    run_chunk(idx, g_pool.n);
    atomic_fetch_add_explicit(&g_pool.done, 1, memory_order_release);
  }
}

static int default_thread_count(void) {
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof allowed, &allowed) == 0) {
    int packages[CPU_SETSIZE];
    int cores[CPU_SETSIZE];
    int count = 0;
    int topology_cpus = 0;
    int available = CPU_COUNT(&allowed);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (!CPU_ISSET(cpu, &allowed)) continue;
      char path[128];
      int package = -1, core = -1;
      snprintf(path, sizeof path,
               "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
      FILE* f = fopen(path, "r");
      if (f) {
        if (fscanf(f, "%d", &package) != 1) package = -1;
        fclose(f);
      }
      snprintf(path, sizeof path,
               "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
      f = fopen(path, "r");
      if (f) {
        if (fscanf(f, "%d", &core) != 1) core = -1;
        fclose(f);
      }
      if (package < 0 || core < 0) continue;
      ++topology_cpus;
      int seen = 0;
      for (int i = 0; i < count; ++i)
        if (packages[i] == package && cores[i] == core) {
          seen = 1;
          break;
        }
      if (!seen) {
        packages[count] = package;
        cores[count] = core;
        ++count;
      }
    }
    if (count > 0 && topology_cpus == available) return count;
    if (available > 0) return available;
  }
  long online = sysconf(_SC_NPROCESSORS_ONLN);
  return online > 0 ? (int)online : 4;
}

void oc_pool_init(int n_threads) {
  if (g_pool.n > 0) return;
  const char* env = getenv("OC_THREADS");
  if (env && atoi(env) > 0) n_threads = atoi(env);
  if (n_threads <= 0) n_threads = default_thread_count();
  atomic_store(&g_pool.gen, 0);
  atomic_store(&g_pool.done, 0);
  atomic_store(&g_pool.shutdown, 0);
  /* Worker index 0 is the calling thread; spawn n-1 helpers (indices 1..n-1).
   * n is what we ACTUALLY got, not what we asked for: the barrier waits for
   * exactly n-1 completions, so a worker the kernel refused (EAGAIN under
   * RLIMIT_NPROC / cgroup pids.max) would never bump `done` and every
   * oc_parallel_for would spin forever. Short a worker is slow; short a
   * worker the barrier still counts is a hang. */
  g_pool.threads = n_threads > 1 ? calloc((size_t)(n_threads - 1), sizeof(pthread_t)) : NULL;
  int made = 0;
  if (g_pool.threads)
    for (int i = 1; i < n_threads; ++i)
      if (pthread_create(&g_pool.threads[made], NULL, worker_main,
                         (void*)(intptr_t)(made + 1)) == 0)
        ++made;
  if (made < n_threads - 1)
    fprintf(stderr, "oc: pool wanted %d threads, got %d\n", n_threads, made + 1);
  g_pool.n = made + 1; /* set last: nothing may dispatch before it is final */
}

void oc_pool_free(void) {
  if (g_pool.n == 0) return;
  atomic_store(&g_pool.shutdown, 1);
  for (int i = 1; i < g_pool.n; ++i) pthread_join(g_pool.threads[i - 1], NULL);
  free(g_pool.threads);
  memset(&g_pool, 0, sizeof g_pool);
}

int oc_pool_size(void) { return g_pool.n > 0 ? g_pool.n : 1; }

void oc_parallel_for(size_t n, oc_range_fn fn, void* ctx) {
  if (n == 0) return;
  if (g_pool.n <= 1 || n < 4) { /* not worth dispatching */
    fn(ctx, 0, n);
    return;
  }
  g_pool.fn = fn;
  g_pool.ctx = ctx;
  g_pool.total = n;
  size_t step = n / ((size_t)g_pool.n * 8);
  g_pool.step = step ? step : 1;
  atomic_store_explicit(&g_pool.next, 0, memory_order_relaxed);
  atomic_store_explicit(&g_pool.done, 0, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_pool.gen, 1, memory_order_release);

  run_chunk(0, g_pool.n); /* main thread takes chunk 0 */

  int spins = 0;
  while (atomic_load_explicit(&g_pool.done, memory_order_acquire) != g_pool.n - 1)
    spin_wait(&spins);
}

/* ---- kernels ---------------------------------------------------------------- */

float oc_dot_f32(const float* a, const float* b, size_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  float sum = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

/* ---- op context ------------------------------------------------------------- */

struct OcCtx {
  /* int8 activation scratch (oc_matvec fast path) */
  int8_t* q;
  float* qd;
  int32_t* bsum;
  size_t q_cap;
  /* oc_matmul panels, both padded to a 16-token width (np): xp = activations
   * packed k-major [cols][np]; yt = accumulators [rows][np], which the workers
   * scatter into the caller's Y. */
  float* xp;
  size_t xp_cap;
  float* yt;
  size_t yt_cap;
};

OcCtx* oc_ctx_new(void) { return calloc(1, sizeof(OcCtx)); }

void oc_ctx_free(OcCtx* c) {
  if (!c) return;
  free(c->q);
  free(c->qd);
  free(c->bsum);
  free(c->xp);
  free(c->yt);
  free(c);
}

static int grow_f(float** p, size_t* cap, size_t n) {
  if (*cap >= n) return 1;
  float* v = realloc(*p, n * sizeof(float));
  if (!v) return 0;
  *p = v;
  *cap = n;
  return 1;
}

static void oom(const char* what) {
  fprintf(stderr, "oc: out of memory (%s)\n", what);
  abort();
}

/* ---- matvec (decode: one token) --------------------------------------------- */

typedef struct {
  float* y;
  uint32_t t;
  const uint8_t* W;
  size_t cols;
  size_t row_bytes;
  const float* x;
  OcQ8Act q8; /* valid when q8.q != NULL */
} MatvecJob;

static void matvec_rows(void* ctx, size_t r0, size_t r1) {
  MatvecJob* j = ctx;
  const uint8_t* W = numa_local(j->W);
  if (j->q8.q) {
    for (size_t r = r0; r < r1; ++r)
      j->y[r] = oc_dot_row_q8(j->t, W + r * j->row_bytes, &j->q8, j->cols);
  } else {
    for (size_t r = r0; r < r1; ++r)
      j->y[r] = oc_dot_row(j->t, W + r * j->row_bytes, j->x, j->cols);
  }
}

void oc_matvec(OcCtx* c, float* y, uint32_t ggml_type, const uint8_t* W,
               size_t rows, size_t cols, const float* x) {
  MatvecJob j = {y, ggml_type, W, cols, oc_row_bytes(ggml_type, cols), x,
                 {NULL, NULL, NULL}};
  /* int8 fast path: worth the quantize cost only for real matrices */
  if (cols % 256 == 0 && rows >= 48 && oc_q8_dot_supported(ggml_type)) {
    if (cols > c->q_cap) {
      free(c->q);
      free(c->qd);
      free(c->bsum);
      c->q = malloc(cols);
      c->qd = malloc(cols / 256 * sizeof(float));
      c->bsum = malloc(cols / 16 * sizeof(int32_t));
      c->q_cap = c->q && c->qd && c->bsum ? cols : 0;
    }
    if (c->q_cap >= cols) {
      oc_q8_quantize(x, cols, c->q, c->qd, c->bsum);
      j.q8.q = c->q;
      j.q8.d = c->qd;
      j.q8.bsum = c->bsum;
    }
  }
  oc_parallel_for(rows, matvec_rows, &j);
}

/* ---- matmul (prefill: a batch of tokens) ------------------------------------
 * Y[t][r] = dot(W[r], X[t]). Loop order is (row tile) x (k block) x (row pair):
 *
 *   - each weight row-block is dequantized ONCE and immediately reused by all
 *     n_tokens accumulators, instead of once per token. W streams from DRAM a
 *     single time for the whole batch — the reason this exists.
 *   - the activation panel slice for one k block (kb * np floats) is loaded to
 *     L1 once and re-read by all OC_GEMM_TR rows of the tile. Drop the row
 *     tiling and that slice comes from L2 once per ROW instead, which costs
 *     more bandwidth than the dequant it was meant to save.
 *   - rows go through the kernel in PAIRS, so one panel load feeds two FMAs.
 *     Measured: one row at a time is load-bound and leaves ~40% on the table.
 *
 * Panel width is padded to a multiple of 16 so the kernels need no token tail;
 * the padding lanes multiply by zero and are dropped on the way out.
 *
 * The accumulator is yt (row-major, so a row's token slots are contiguous for
 * the kernel) and each worker scatters its own rows into the caller's
 * [n_tokens][rows] Y — every token's activation row has to be contiguous for
 * the norms and elementwise ops that consume it, and doing the scatter in the
 * worker keeps it off the single calling thread (it was 12% there). */

#define OC_GEMM_TR 64  /* rows sharing one L1-resident panel slice */
#define OC_GEMM_KB 256 /* k block: one K-quant block, and a 4x256 dequant buffer
                        * (4 KiB of worker stack) with the panel slice beside it
                        * in L1. Also the only k step that lands on a block
                        * boundary for K-quants (256), Q4_0/Q8_0/AL5_XS (32) and
                        * f32/f16 (1) alike. */

/* Below this, materializing a dequantized row costs more than reusing it saves,
 * and oc_matvec's fused dot kernels — which never write the row to memory at
 * all — win outright. Measured, not guessed: at gemma-4-31B shapes on a Zen3+
 * the crossover sits at 8 (make gemm-bench). */
#define OC_GEMM_MIN_TOKENS 8

typedef struct {
  float* yt;
  float* Y;
  const float* xp;
  const uint8_t* W;
  uint32_t t;
  size_t cols, row_bytes, rows, n, np;
} GemmJob;

static void gemm_rows(void* ctx, size_t r0, size_t r1) {
  GemmJob* j = ctx;
  const uint8_t* W = numa_local(j->W);
  const size_t np = j->np, cols = j->cols;
  float rb[4 * OC_GEMM_KB]; /* 4 dequantized rows, contiguous for oc_gemm_row4 */
  for (size_t rt = r0; rt < r1; rt += OC_GEMM_TR) {
    size_t rt1 = rt + OC_GEMM_TR < r1 ? rt + OC_GEMM_TR : r1;
    memset(j->yt + rt * np, 0, (rt1 - rt) * np * sizeof(float));
    for (size_t k0 = 0; k0 < cols; k0 += OC_GEMM_KB) {
      size_t kb = cols - k0 < OC_GEMM_KB ? cols - k0 : OC_GEMM_KB;
      const float* xs = j->xp + k0 * np;
      size_t off = oc_row_bytes(j->t, k0); /* k0 is block-aligned by construction */
      const uint8_t* wr = W + rt * j->row_bytes + off;
      size_t r = rt;
      for (; r + 4 <= rt1; r += 4, wr += 4 * j->row_bytes) {
        for (size_t i = 0; i < 4; ++i)
          oc_dequant_row(j->t, wr + i * j->row_bytes, rb + i * kb, kb);
        oc_gemm_row4(j->yt + r * np, rb, kb, xs, kb, np);
      }
      for (; r < rt1; ++r, wr += j->row_bytes) { /* 1-3 rows left over */
        oc_dequant_row(j->t, wr, rb, kb);
        oc_gemm_row(j->yt + r * np, rb, xs, kb, np);
      }
    }
    /* scatter this tile into Y [n_tokens][rows], dropping the pad lanes */
    for (size_t r = rt; r < rt1; ++r) {
      const float* a = j->yt + r * np;
      for (size_t i = 0; i < j->n; ++i) j->Y[i * j->rows + r] = a[i];
    }
  }
}

void oc_matmul(OcCtx* c, float* Y, uint32_t ggml_type, const uint8_t* W,
               size_t rows, size_t cols, const float* X, size_t n_tokens) {
  size_t row_bytes = oc_row_bytes(ggml_type, cols);
  if (row_bytes == 0 || n_tokens == 0 || rows == 0) {
    fprintf(stderr, "oc_matmul: no kernel for type %u with %zu cols (%zu rows, "
                    "%zu tokens)\n", ggml_type, cols, rows, n_tokens);
    abort(); /* silently returning a wrong number here poisons every logit */
  }
  if (n_tokens < OC_GEMM_MIN_TOKENS) {
    for (size_t i = 0; i < n_tokens; ++i)
      oc_matvec(c, Y + i * rows, ggml_type, W, rows, cols, X + i * cols);
    return;
  }
  const size_t np = (n_tokens + 15) & ~(size_t)15;
  if (!grow_f(&c->xp, &c->xp_cap, cols * np)) oom("gemm activation panel");
  if (!grow_f(&c->yt, &c->yt_cap, rows * np)) oom("gemm accumulator");

  /* On this exact shape+type oc_matvec (decode) takes the int8/VNNI path, so it
   * sees activations rounded to int8. Prefill must consume the SAME rounded
   * activations or the KV cache it writes disagrees with what decode would have
   * written — the batched==sequential invariant breaks on VNNI hardware only
   * (where q8 is bound), so the AVX2 test box never sees it. Round-trip each
   * token's row through the same quantizer before packing. No-op elsewhere:
   * oc_q8_dot_supported is false unless VNNI is the resolved ISA. */
  int rt = cols % 256 == 0 && rows >= 48 && oc_q8_dot_supported(ggml_type);
  if (rt) {
    if (cols > c->q_cap) {
      free(c->q);
      free(c->qd);
      free(c->bsum);
      c->q = malloc(cols);
      c->qd = malloc(cols / 256 * sizeof(float));
      c->bsum = malloc(cols / 16 * sizeof(int32_t));
      c->q_cap = c->q && c->qd && c->bsum ? cols : 0;
    }
    rt = c->q_cap >= cols; /* alloc failed => fall back to raw f32 (both paths) */
  }

  if (rt) {
    /* token-major: quantize a whole row (block scales span 256 lanes), then
     * scatter its dequantized-int8 values into the column-major panel. */
    for (size_t i = 0; i < n_tokens; ++i) {
      oc_q8_quantize(X + i * cols, cols, c->q, c->qd, c->bsum);
      for (size_t k = 0; k < cols; ++k)
        c->xp[k * np + i] = (float)c->q[k] * c->qd[k / 256];
    }
    for (size_t k = 0; k < cols; ++k)
      for (size_t i = n_tokens; i < np; ++i) c->xp[k * np + i] = 0.0f;
  } else {
    /* X [n][cols] -> xp [cols][np], pad lanes zeroed. Written sequentially; the
     * strided reads walk n cache lines that stay hot for the next 16 k. */
    for (size_t k = 0; k < cols; ++k) {
      float* dst = c->xp + k * np;
      for (size_t i = 0; i < n_tokens; ++i) dst[i] = X[i * cols + k];
      for (size_t i = n_tokens; i < np; ++i) dst[i] = 0.0f;
    }
  }

  GemmJob j = {.yt = c->yt,
               .Y = Y,
               .xp = c->xp,
               .W = W,
               .t = ggml_type,
               .cols = cols,
               .row_bytes = row_bytes,
               .rows = rows,
               .n = n_tokens,
               .np = np};
  oc_parallel_for(rows, gemm_rows, &j);
}


void oc_rms_norm(float* out, const float* x, const float* w, size_t n, float eps) {
  float sum_sq = 0.0f;
  for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
  float inv_rms = 1.0f / sqrtf(sum_sq / (float)n + eps);
  for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv_rms * w[i];
}

void oc_softmax(float* x, size_t n) {
  if (n == 0) return;
  float mx = x[0];
  for (size_t i = 1; i < n; ++i)
    if (x[i] > mx) mx = x[i];
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float e = expf(x[i] - mx);
    x[i] = e;
    sum += e;
  }
  float inv = (float)(1.0 / sum);
  for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

void oc_rope(float* vec, size_t head_dim, size_t num_heads, size_t pos,
             float theta, size_t rope_dim, const float* freqs) {
  size_t rope_len = rope_dim == 0 ? head_dim : rope_dim;
  if (rope_len > head_dim) rope_len = head_dim;
  if (pos == 0 || rope_len == 0) return; /* matches apply_rope_f32: identity at pos 0 */
  size_t half = rope_len / 2;
  float posf = (float)pos;
  /* Geometric frequency recurrence (matches oxidize-cpp apply_rope). */
  float mul = powf(theta, -2.0f / (float)rope_len);
  for (size_t h = 0; h < num_heads; ++h) {
    float* p = vec + h * head_dim;
    float freq = 1.0f;
    for (size_t i = 0; i < half; ++i) {
      float x0 = p[i], x1 = p[half + i];
      float angle = posf * freq;
      if (freqs) angle /= freqs[i];
      float c = cosf(angle), s = sinf(angle);
      p[i] = x0 * c - x1 * s;
      p[half + i] = x0 * s + x1 * c;
      freq *= mul;
    }
  }
}

void oc_rope_normal(float* vec, size_t head_dim, size_t num_heads, size_t pos,
                    float theta, size_t rope_dim) {
  size_t rope_len = rope_dim == 0 ? head_dim : rope_dim;
  if (rope_len > head_dim) rope_len = head_dim;
  if (pos == 0 || rope_len == 0) return; /* identity at pos 0, like oc_rope */
  size_t half = rope_len / 2;
  float posf = (float)pos;
  float mul = powf(theta, -2.0f / (float)rope_len);
  for (size_t h = 0; h < num_heads; ++h) {
    float* p = vec + h * head_dim;
    float freq = 1.0f;
    for (size_t i = 0; i < half; ++i) {
      float x0 = p[2 * i], x1 = p[2 * i + 1]; /* adjacent pair, not split half */
      float angle = posf * freq;
      float c = cosf(angle), s = sinf(angle);
      p[2 * i] = x0 * c - x1 * s;
      p[2 * i + 1] = x0 * s + x1 * c;
      freq *= mul;
    }
  }
}

void oc_geglu(float* gate, const float* up, float* out, size_t n) {
  const float K = 0.797884560f; /* sqrt(2/pi) */
  for (size_t i = 0; i < n; ++i) {
    float g = gate[i];
    float gelu = 0.5f * g * (1.0f + tanhf(K * (g + 0.044715f * g * g * g)));
    out[i] = gelu * up[i];
  }
}

/* ---- KV-cache precision ---------------------------------------------------- */

/* Set once during argument parsing, before the pool spins or any model loads;
 * read single-threaded inside the loaders. Not touched by the worker threads
 * (the resolved type lives on the model), so a plain global is race-free here —
 * same contract as oc_force_isa. */
static OcKvType g_kv_type = OC_KV_F32;
void oc_kv_set_type(OcKvType t) { g_kv_type = t; }
OcKvType oc_kv_get_type(void) { return g_kv_type; }

const char* oc_kv_type_name(OcKvType t) {
  switch (t) {
    case OC_KV_F16: return "f16";
    case OC_KV_Q8: return "q8";
    case OC_KV_Q4: return "q4";
    default: return "f32";
  }
}

size_t oc_kv_elem_bytes(OcKvType t) {
  switch (t) {
    case OC_KV_F16: return 2;
    case OC_KV_Q8: return 1;
    case OC_KV_Q4: return 0; /* packed; stored via oc_kvq_* not the byte path */
    default: return 4;
  }
}

void oc_kv_encode(OcKvType t, const float* x, size_t n, uint8_t* out, float* scale) {
  if (t == OC_KV_F16) {
    /* Byte-wise little-endian halves: `out` is only 1-byte aligned (stack tmp
     * in the batch round-trip), so no uint16_t* cast — that would be a UBSAN
     * alignment trap. */
    for (size_t i = 0; i < n; ++i) {
      uint16_t h = oc_f32_to_f16(x[i]);
      out[2 * i] = (uint8_t)(h & 0xff);
      out[2 * i + 1] = (uint8_t)(h >> 8);
    }
    if (scale) *scale = 0.0f;
    return;
  }
  /* Q8: symmetric int8, one scale per head. amax/127 keeps +127 and -127
   * reachable and symmetric; all-zero head => scale 0, codes 0. */
  float amax = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float a = fabsf(x[i]);
    if (a > amax) amax = a;
  }
  float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
  int8_t* q = (int8_t*)out;
  for (size_t i = 0; i < n; ++i) {
    long v = lroundf(x[i] * id);
    if (v > 127) v = 127;
    else if (v < -127) v = -127;
    q[i] = (int8_t)v;
  }
  if (scale) *scale = d;
}

void oc_kv_decode(OcKvType t, const uint8_t* in, size_t n, float scale, float* out) {
  if (t == OC_KV_F16) {
    for (size_t i = 0; i < n; ++i)
      out[i] = oc_f16_to_f32((uint16_t)(in[2 * i] | (uint16_t)in[2 * i + 1] << 8));
    return;
  }
  const int8_t* q = (const int8_t*)in;
  for (size_t i = 0; i < n; ++i) out[i] = (float)q[i] * scale;
}
