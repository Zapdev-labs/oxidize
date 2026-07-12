/* oxidize-c CLI: one-shot generation or HTTP/WebSocket server. */
#include "oc.h"
#include "gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef __NR_set_mempolicy
#define __NR_set_mempolicy 238
#endif
#define OC_MPOL_DEFAULT 0
#define OC_MPOL_BIND 2
#define OC_MPOL_INTERLEAVE 3

int oc_serve(oc_model *m, oc_tokenizer *tok, const char *host, int port,
             float temperature, size_t draft_k, int spec_mode,
             size_t max_seqs, size_t max_queue, const char *worker_id, int device_id);

/* SMT threads thrash the memory-bound gemv; default to physical cores per
 * socket (cpuinfo "cpu cores"). Dual-socket boxes still report per-socket. */
static int physical_cores(void) {
  int cores = 0;
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (f) {
    char line[256];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "cpu cores", 9) == 0) {
        const char *c = strchr(line, ':');
        if (c) cores = atoi(c + 1);
        break;
      }
    }
    fclose(f);
  }
  return cores > 0 ? cores : 4;
}

static int parse_cpulist_into(const char *s, int *out, int cap) {
  int n = 0;
  const char *p = s;
  while (*p && n < cap) {
    while (*p == ' ' || *p == '\n' || *p == '\r') ++p;
    if (!*p) break;
    char *end = NULL;
    long lo = strtol(p, &end, 10);
    if (end == p) break;
    p = end;
    long hi = lo;
    if (*p == '-') {
      hi = strtol(p + 1, &end, 10);
      p = end;
    }
    for (long c = lo; c <= hi && n < cap; ++c) out[n++] = (int)c;
    if (*p == ',') ++p;
  }
  return n;
}

static int node_cpus(int node, int *out, int cap) {
  char path[128];
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[512];
  if (!fgets(buf, sizeof buf, f)) { fclose(f); return 0; }
  fclose(f);
  return parse_cpulist_into(buf, out, cap);
}

static int node0_cpus(int *out, int cap) { return node_cpus(0, out, cap); }

static int all_node_cpus(int *out, int cap) {
  int n = 0;
  DIR *d = opendir("/sys/devices/system/node");
  if (!d) return 0;
  struct dirent *de;
  while ((de = readdir(d)) != NULL && n < cap) {
    if (strncmp(de->d_name, "node", 4) != 0) continue;
    char path[256];
    snprintf(path, sizeof path, "/sys/devices/system/node/%s/cpulist", de->d_name);
    FILE *f = fopen(path, "r");
    if (!f) continue;
    char buf[512];
    if (fgets(buf, sizeof buf, f))
      n += parse_cpulist_into(buf, out + n, cap - n);
    fclose(f);
  }
  closedir(d);
  return n;
}

static int node1_cpus(int *out, int cap) { return node_cpus(1, out, cap); }

static void build_cpu_node_map(void) {
  int cpus[256];
  int n0 = node0_cpus(cpus, 128);
  int n1 = node1_cpus(cpus + n0, 128 - n0);
  int n = n0 + n1;
  if (n <= 0) return;
  int *map = malloc(256 * sizeof(int));
  if (!map) return;
  for (int i = 0; i < 256; ++i) map[i] = 0;
  for (int i = 0; i < n0; ++i) if (cpus[i] < 256) map[cpus[i]] = 0;
  for (int i = 0; i < n1; ++i)
    if (cpus[n0 + i] < 256) map[cpus[n0 + i]] = 1;
  oc_numa_set_cpu_node_map(map, 256);
  free(map);
}

/* --numa single|interleave|replicate: bind memory + pin OpenMP threads. */
static int g_numa_replicate = 0;

static void init_numa(const char *mode, int *threads_inout) {
  int cpus[256];
  int ncpu = 0;
  unsigned long nodemask = 0;
  int policy = OC_MPOL_DEFAULT;
  if (!mode || !strcmp(mode, "single")) {
    ncpu = node0_cpus(cpus, 256);
    nodemask = 1ul;
    policy = OC_MPOL_BIND;
  } else if (!strcmp(mode, "replicate")) {
    g_numa_replicate = 1;
    build_cpu_node_map();
    oc_numa_set_replica_node(1);
    oc_numa_bind_alloc_node(0);
    int n0 = node0_cpus(cpus, 128);
    int n1 = node1_cpus(cpus + n0, 128 - n0);
    int per0 = *threads_inout > 0 ? *threads_inout / 2 : n0 / 4;
    int per1 = *threads_inout > 0 ? *threads_inout / 2 : n1 / 4;
    if (per0 < 1) per0 = 1;
    if (per1 < 1) per1 = 1;
    int out[256], n = 0;
    for (int i = 0; i < per0 && i < n0; ++i) out[n++] = cpus[i];
    for (int i = 0; i < per1 && i < n1; ++i) out[n++] = cpus[n0 + i];
    memcpy(cpus, out, (size_t)n * sizeof(int));
    ncpu = n;
    *threads_inout = n;
  } else if (!strcmp(mode, "interleave") || !strcmp(mode, "all")) {
    ncpu = all_node_cpus(cpus, 256);
    DIR *d = opendir("/sys/devices/system/node");
    if (d) {
      struct dirent *de;
      while ((de = readdir(d)) != NULL) {
        if (strncmp(de->d_name, "node", 4) != 0) continue;
        int id = atoi(de->d_name + 4);
        if (id >= 0 && id < 64) nodemask |= 1ul << id;
      }
      closedir(d);
    }
    policy = !strcmp(mode, "interleave") ? OC_MPOL_INTERLEAVE : OC_MPOL_DEFAULT;
  } else {
    char *end = NULL;
    long nid = strtol(mode, &end, 10);
    if (end && *end == '\0' && nid >= 0 && nid < 64) {
      ncpu = node_cpus((int)nid, cpus, 256);
      nodemask = 1ul << (unsigned)nid;
      policy = OC_MPOL_BIND;
    } else {
      fprintf(stderr, "unknown --numa %s (use single|interleave|all|replicate|<node-id>)\n", mode);
      return;
    }
  }
  if (ncpu <= 0) return;
  if (policy != OC_MPOL_DEFAULT)
    syscall(__NR_set_mempolicy, policy, &nodemask, 64);
  int thr = *threads_inout;
  if (thr <= 0) {
    /* Prefer physical cores on the bound set (even indices on Xeon HT). */
    thr = ncpu;
    if (!mode || !strcmp(mode, "single") ||
        (mode[0] >= '0' && mode[0] <= '9')) {
      int phys = physical_cores();
      if (phys > 0 && phys < thr) thr = phys;
      if (thr > 16 && phys >= 16) thr = 16; /* dual-socket dense default */
    }
  }
  if (thr > ncpu) thr = ncpu;
  *threads_inout = thr;
#ifdef _OPENMP
  omp_set_num_threads(thr);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid < ncpu) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpus[tid % ncpu], &set);
      sched_setaffinity(0, sizeof set, &set);
    }
  }
#else
  (void)cpus;
#endif
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct { oc_tokenizer *tok; } stream_ud;
static void print_token(uint32_t id, void *ud) {
  stream_ud *s = ud;
  char frag[256];
  size_t fn = oc_detokenize(s->tok, id, frag, sizeof frag);
  fwrite(frag, 1, fn, stdout);
  fflush(stdout);
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --model <path.gguf> [options]\n"
          "       %s prune --input in.gguf --output out.gguf [options]\n"
          "       %s finetune --model m.gguf --data data.jsonl [options]\n",
          prog, prog, prog);
  fprintf(stderr,
          "  --prompt <str>      prompt text (raw)\n"
          "  --chat              wrap prompt in ChatML (qwen chat models)\n"
          "  --max-tokens N      decode budget (default 64)\n"
          "  --ctx N             context/KV limit (default 8192)\n"
          "  --kv-int8           int8 KV cache (4x smaller, faster full ctx)\n"
          "  --draft N           speculative draft length (default 4, 0=off)\n"
          "  --spec MODE         drafting: ngram (default) | mtp | off\n"
          "  --threads N         worker threads (default: physical cores / NUMA)\n"
          "  --numa MODE         single (default) | interleave | all | replicate | <node-id>\n"
          "  --temperature F     sampling temperature (default 0 = greedy)\n"
          "  --top-k N --top-p F sampling filters\n"
          "  --min-p F           minimum relative token probability (0..1)\n"
          "  --frequency-penalty F  penalty per recent token occurrence\n"
          "  --presence-penalty F   penalty applied once to seen tokens\n"
          "  --penalty-last-n N  recent-token penalty window (default 256)\n"
          "  --seed N            RNG seed\n"
          "  --no-bos            don't prepend BOS\n"
          "  --stream            stream text to stdout\n"
          "  --serve             run HTTP+WebSocket server instead of one-shot\n"
          "  --port N            server port (default 8090)\n"
          "  --host A            bind address (default 127.0.0.1; 0.0.0.0 for LAN)\n");
  fprintf(stderr,
          "  --max-seqs N        CUDA Gemma batch slots (1..64, default 1)\n"
          "  --max-queue N       waiting request limit (default 0)\n"
          "  --gpu-list IDS      worker identity; run one process per visible GPU\n");
  exit(2);
}

static size_t parse_size_option(const char *option, const char *value,
                                size_t minimum) {
  char *end = NULL;
  errno = 0;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (!value[0] || value[0] == '-' || value[0] == '+' || errno || !end ||
      *end || parsed < minimum ||
      parsed > (unsigned long long)SIZE_MAX)
    oc_die("bad %s %s", option, value);
  return (size_t)parsed;
}

static float parse_float_option(const char *option, const char *value,
                                float minimum, float maximum) {
  char *end = NULL;
  errno = 0;
  float parsed = strtof(value, &end);
  if (!value[0] || errno || !end || *end || !isfinite(parsed) ||
      parsed < minimum || parsed > maximum)
    oc_die("bad %s %s", option, value);
  return parsed;
}

static void validate_gpu_list(const char *value) {
  const char *p = value;
  size_t count = 0;
  while (*p) {
    char *end = NULL;
    errno = 0;
    long id = strtol(p, &end, 10);
    if (end == p || errno || id < 0) oc_die("bad --gpu-list %s", value);
    for (const char *prior = value; prior < p;) {
      char *prior_end = NULL;
      long prior_id = strtol(prior, &prior_end, 10);
      if (prior_id == id) oc_die("duplicate GPU id in --gpu-list %s", value);
      prior = *prior_end == ',' ? prior_end + 1 : prior_end;
    }
    ++count;
    p = end;
    if (!*p) break;
    if (*p != ',') oc_die("bad --gpu-list %s", value);
    ++p;
    if (!*p) oc_die("bad --gpu-list %s", value);
  }
  if (!count) oc_die("bad --gpu-list %s", value);
}

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "prune"))
    return oc_prune_main(argc - 2, argv + 2);
  if (argc > 1 && !strcmp(argv[1], "finetune"))
    return oc_finetune_main(argc - 2, argv + 2);

  const char *model_path = NULL, *prompt = "Hello", *host = "127.0.0.1";
  size_t max_tokens = 64, top_k = 0, ctx = 8192, draft = 4, penalty_last_n = 256;
  int kv_int8 = 0;
  int spec_mode = OC_SPEC_NGRAM;
  float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
  float frequency_penalty = 0.0f, presence_penalty = 0.0f;
  int threads = 0, port = 8090;
  size_t max_seqs = 1, max_queue = 0;
  bool no_bos = false, stream = false, serve = false, chat = false;
  const char *numa_mode = "single", *gpu_list = NULL;
  uint64_t seed = 0;

  for (int i = 1; i < argc; ++i) {
#define VAL() (i + 1 < argc ? argv[++i] : (usage(argv[0]), (char *)0))
    if (!strcmp(argv[i], "--model")) model_path = VAL();
    else if (!strcmp(argv[i], "--prompt")) prompt = VAL();
    else if (!strcmp(argv[i], "--max-tokens")) max_tokens = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--ctx")) ctx = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--draft")) draft = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--spec")) {
      const char *v = VAL();
      if (!strcmp(v, "ngram")) spec_mode = OC_SPEC_NGRAM;
      else if (!strcmp(v, "mtp")) spec_mode = OC_SPEC_MTP;
      else if (!strcmp(v, "off")) spec_mode = OC_SPEC_OFF;
      else usage(argv[0]);
    }
    else if (!strcmp(argv[i], "--threads")) threads = atoi(VAL());
    else if (!strcmp(argv[i], "--max-seqs")) {
      max_seqs = parse_size_option("--max-seqs", VAL(), 1);
      if (max_seqs > 64) oc_die("--max-seqs must be at most 64");
    }
    else if (!strcmp(argv[i], "--max-queue"))
      max_queue = parse_size_option("--max-queue", VAL(), 0);
    else if (!strcmp(argv[i], "--gpu-list")) {
      gpu_list = VAL();
      validate_gpu_list(gpu_list);
    }
    else if (!strcmp(argv[i], "--numa")) numa_mode = VAL();
    else if (!strcmp(argv[i], "--temperature"))
      temperature = parse_float_option("--temperature", VAL(), 0.0f, INFINITY);
    else if (!strcmp(argv[i], "--top-k")) top_k = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--top-p"))
      top_p = parse_float_option("--top-p", VAL(), 0.0f, 1.0f);
    else if (!strcmp(argv[i], "--min-p"))
      min_p = parse_float_option("--min-p", VAL(), 0.0f, 1.0f);
    else if (!strcmp(argv[i], "--frequency-penalty"))
      frequency_penalty =
          parse_float_option("--frequency-penalty", VAL(), 0.0f, INFINITY);
    else if (!strcmp(argv[i], "--presence-penalty"))
      presence_penalty =
          parse_float_option("--presence-penalty", VAL(), 0.0f, INFINITY);
    else if (!strcmp(argv[i], "--penalty-last-n"))
      penalty_last_n = parse_size_option("--penalty-last-n", VAL(), 0);
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(VAL(), 0, 10);
    else if (!strcmp(argv[i], "--kv-int8")) kv_int8 = 1;
    else if (!strcmp(argv[i], "--no-bos")) no_bos = true;
    else if (!strcmp(argv[i], "--stream")) stream = true;
    else if (!strcmp(argv[i], "--serve")) serve = true;
    else if (!strcmp(argv[i], "--chat")) chat = true;
    else if (!strcmp(argv[i], "--port")) {
      const char *p = VAL();
      char *end = NULL;
      long v = strtol(p, &end, 10);
      if (!p[0] || (end && *end) || v < 1 || v > 65535) oc_die("bad --port %s", p);
      port = (int)v;
    }
    else if (!strcmp(argv[i], "--host")) host = VAL();
    else usage(argv[0]);
#undef VAL
  }
  if (!model_path) usage(argv[0]);
  if (serve && max_seqs > 1 && kv_int8)
    oc_die("serve: --max-seqs > 1 does not support --kv-int8");
  oc_gen_seed(seed);
  if (getenv("OMP_NUM_THREADS") && threads <= 0)
    threads = atoi(getenv("OMP_NUM_THREADS"));
  if (serve && threads > 4) threads -= 2;
  init_numa(numa_mode, &threads);
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#else
  (void)threads;
#endif

  double t0 = now_s();
  oc_model *m = oc_model_load(model_path, ctx, kv_int8);
  if (g_numa_replicate) oc_model_numa_replicate(m);
  double t_load = now_s() - t0;
  const oc_config *c = &m->cfg;
  size_t n_gdn = 0;
  for (size_t l = 0; l < c->layer_count; ++l)
    if (m->layers[l].is_gdn) n_gdn++;
  fprintf(stderr,
          "loaded in %.1fs: layers=%zu (%zu GDN + %zu attn) hidden=%zu heads=%zu/"
          "%zu head_dim=%zu vocab=%zu ctx=%zu (kv %zu) inter=%zu mtp=%s\n",
          t_load, c->layer_count, n_gdn, c->layer_count - n_gdn, c->hidden_size,
          c->n_heads, c->kv_heads, c->head_dim, c->vocab_size, c->context_size,
          m->kv_ctx, c->intermediate_size, m->mtp ? "yes" : "no");

  oc_tokenizer *tok = oc_tokenizer_load(m->g);
  if (!tok) oc_die("no usable tokenizer in GGUF");

  if (serve) {
    if (!getenv("OXIDIZE_API_KEY") && !getenv("OXIDIZE_API_KEYS"))
      oc_die("serve: set OXIDIZE_API_KEY or OXIDIZE_API_KEYS");
    if (max_seqs > 1) {
      if (!gpu_list) oc_die("serve: --max-seqs > 1 requires --gpu-list <worker-identity>");
#ifndef OC_CUDA
      oc_die("serve: --max-seqs > 1 requires an OC_CUDA build");
#else
      if (!m->gemma || !m->gpu_active)
        oc_die("serve: --max-seqs > 1 requires a CUDA-resident Gemma model");
#endif
    }
    return oc_serve(m, tok, host, port, temperature, draft, spec_mode,
                    max_seqs, max_queue, gpu_list ? gpu_list : "scalar",
                    0);
  }

  /* ---- one-shot CLI ---- */
  char *wrapped = NULL;
  const char *ptext = prompt;
  if (chat) {
    size_t need = strlen(prompt) + 128;
    wrapped = malloc(need);
    snprintf(wrapped, need,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    ptext = wrapped;
  }
  size_t n_prompt;
  uint32_t *ids = oc_tokenize(tok, ptext, !no_bos && !chat, &n_prompt);
  free(wrapped);
  if (n_prompt == 0) oc_die("empty prompt after tokenize");
  if (n_prompt + max_tokens + 2 > m->kv_ctx)
    max_tokens = m->kv_ctx > n_prompt + 2 ? m->kv_ctx - n_prompt - 2 : 0;

  oc_gen g = {0};
  g.m = m;
  g.tok = tok;
  g.temperature = temperature;
  g.top_k = top_k;
  g.top_p = top_p;
  g.min_p = min_p;
  g.frequency_penalty = frequency_penalty;
  g.presence_penalty = presence_penalty;
  g.penalty_last_n = penalty_last_n;
  g.draft_k = draft;
  g.spec_mode = spec_mode;
  stream_ud sud = {tok};
  if (stream) { g.on_token = print_token; g.ud = &sud; }

  uint32_t *out = malloc(max_tokens * sizeof(uint32_t));
  double t1 = now_s();
  size_t n_out = oc_generate(&g, ids, n_prompt, max_tokens, out);
  double t_gen = now_s() - t1;
  if (stream) fputc('\n', stdout);

  fprintf(stderr, "gen: %zu prompt + %zu tokens in %.2fs (%.2f tok/s)",
          n_prompt, n_out, t_gen, t_gen > 0 ? (double)n_out / t_gen : 0.0);
  if (g.drafted)
    fprintf(stderr, "  [mtp: drafted %zu accepted %zu = %.0f%%]", g.drafted,
            g.accepted, 100.0 * (double)g.accepted / (double)g.drafted);
  fprintf(stderr, "\n");
  if (!stream) {
    char frag[256];
    printf("text: ");
    for (size_t i = 0; i < n_out; ++i) {
      size_t fn = oc_detokenize(tok, out[i], frag, sizeof frag);
      fwrite(frag, 1, fn, stdout);
    }
    printf("\n");
  }
  free(out);
  free(ids);
  oc_tokenizer_free(tok);
  oc_model_free(m);
  return 0;
}
