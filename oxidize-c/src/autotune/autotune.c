/*
 * autotune.c — CPU detection, GGUF fingerprint, tuning plan.
 *
 * Port of oxidize-core/src/autotune/{detect,fingerprint,rules}.rs.
 *
 * Linux detection: /proc/cpuinfo (model name), /sys/devices/system/cpu/
 * (core counts), /sys/devices/system/node/ (NUMA), /proc/meminfo (RAM).
 * SIMD via oc_simd_caps() (which uses __builtin_cpu_supports). Other
 * platforms get a conservative scalar-only info.
 *
 * The plan() function is PURE: same (cpu, model) inputs → same plan output,
 * with every decision captured in a rationale string (no side effects).
 */
/* _GNU_SOURCE enables cpu_set_t + sched_setaffinity on glibc. Must be the
 * first non-comment thing in the file. */
#ifdef __linux__
#  define _GNU_SOURCE
#  include <sched.h>     /* MUST be first: cpu_set_t needs _GNU_SOURCE      */
#endif
#include "oxidize/autotune.h"

#include "oxidize/gguf.h"
#include "oxidize/log.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"
#include "oxidize/simd.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Local shim: read a u32 metadata value, return 0 if absent (used by
 * fingerprint to read arch-prefixed config keys with 0 defaults). */
static uint32_t meta_u32_or_zero(const OcGgufFile *f, const char *key)
{
    uint32_t v = 0;
    return oc_gguf_metadata_get_u32(f, key, &v) ? v : 0;
}

/* ─── CPU detection ───────────────────────────────────────────────────── */

#if defined(__linux__)
#  define OC_LINUX 1
#else
#  define OC_LINUX 0
#endif

/* Read the first integer from the first line of `path`. Returns false if the
 * file cannot be opened or no integer is found. */
#if OC_LINUX
static bool read_first_int(const char *path, long *out)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) return false;
    bool ok = (fscanf(f, "%ld", out) == 1);
    fclose(f);
    return ok;
}
#endif

/* Count entries (dirs matching "node[0-9]+") under /sys/devices/system/node/. */
static uint32_t count_numa_nodes(void)
{
#if OC_LINUX
    /* /sys/devices/system/node/possible gives "0-N" or "0". */
    long possible = 1;
    if (read_first_int("/sys/devices/system/node/possible", &possible)) {
        /* The file contains "0" or "0-3"; fscanf reads the leading 0, but we
         * need the count. Re-parse for the "N" in "0-N". */
    }
    /* Simpler: count /sys/devices/system/node/nodeN dirs via /proc. Use the
     * kernel's online node count from /sys/devices/system/node/online if it
     * exists (format "0-3" → 4 nodes). */
    FILE *f = fopen("/sys/devices/system/node/online", "r");
    if (f == NULL) return 1;   /* no NUMA sysfs → assume UMA */
    char buf[64];
    buf[0] = '\0';
    if (fgets(buf, sizeof(buf), f) == NULL) { fclose(f); return 1; }
    fclose(f);
    /* Parse "a-b" → b - a + 1, else count comma-separated. */
    long lo, hi;
    if (sscanf(buf, "%ld-%ld", &lo, &hi) == 2) {
        return (uint32_t)(hi - lo + 1);
    }
    /* Single node or comma list: count commas + 1. */
    uint32_t n = 1;
    for (char *p = buf; *p; p++) if (*p == ',') n++;
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

static uint32_t count_logical_cores(void)
{
#if OC_LINUX
    long n = 0;
    if (read_first_int("/sys/devices/system/cpu/online", &n)) {
        /* Format "0-95". Re-parse the high end. */
        FILE *f = fopen("/sys/devices/system/cpu/online", "r");
        if (f == NULL) return 1;
        char buf[64];
        if (fgets(buf, sizeof(buf), f) == NULL) { fclose(f); return 1; }
        fclose(f);
        long lo, hi;
        if (sscanf(buf, "%ld-%ld", &lo, &hi) == 2) return (uint32_t)(hi + 1);
        return 1;
    }
    return 1;
#else
    return 1;
#endif
}

static uint64_t read_meminfo_field(const char *key)
{
#if OC_LINUX
    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) return 0;
    char line[256];
    uint64_t val = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, key) == line) {
            /* "Key:    12345 kB" */
            sscanf(line + strlen(key), ": %llu kB", (unsigned long long *)&val);
            val *= 1024ULL;   /* kB → bytes */
            break;
        }
    }
    fclose(f);
    return val;
#else
    (void)key;
    return 0;
#endif
}

static void read_model_name(char *buf, size_t cap)
{
#if OC_LINUX
    buf[0] = '\0';
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == NULL) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ') colon++;
                size_t len = strlen(colon);
                while (len > 0 && (colon[len-1] == '\n' || colon[len-1] == ' ')) len--;
                if (len >= cap) len = cap - 1;
                memcpy(buf, colon, len);
                buf[len] = '\0';
            }
            break;
        }
    }
    fclose(f);
#else
    if (cap > 0) buf[0] = '\0';
#endif
}

OcError oc_autotune_detect_cpu(OcCpuInfo *out)
{
    if (out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->logical_cores = count_logical_cores();
    out->physical_cores = out->logical_cores;  /* conservative; /proc/cpuinfo
                                                * core id counting is fiddly */
    out->numa_nodes = count_numa_nodes();
    out->is_dual_socket = (out->numa_nodes >= 2);
    out->total_ram_bytes = read_meminfo_field("MemTotal");
    out->available_ram_bytes = read_meminfo_field("MemAvailable");
    out->simd = *oc_simd_caps();
    read_model_name(out->model_name, sizeof(out->model_name));
    return OC_OK;
}

/* ─── GGUF fingerprint ────────────────────────────────────────────────── */

OcError oc_autotune_fingerprint_gguf(const OcGgufMmappedFile *m,
                                     OcModelFingerprint *out)
{
    if (m == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->file_bytes = oc_gguf_map_total_bytes(m);
    out->arch = oc_gguf_arch_from_file(&m->unified);

    /* Read the canonical config fields (best-effort; defaults if absent). */
    const OcGgufFile *f = &m->unified;
    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(f, "general.architecture", &arch_str, &arch_len)) {
        /* ok */
    } else {
        arch_str = oc_model_arch_name(out->arch);
    }
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch_str ? arch_str : "llama");
    char key[160];

    snprintf(key, sizeof(key), "%sblock_count", prefix);    out->n_layer = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%sembedding_length", prefix); out->n_embd = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%sattention.head_count", prefix); out->n_head = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%sattention.head_count_kv", prefix); out->n_head_kv = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%sfeed_forward_length", prefix); out->n_ff = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%scontext_length", prefix); out->n_ctx = meta_u32_or_zero(f, key);
    snprintf(key, sizeof(key), "%svocab_size", prefix);
    if (!oc_gguf_metadata_get_u32(f, key, &out->vocab_size)) {
        oc_gguf_metadata_get_u32(f, "general.vocab_size", &out->vocab_size);
    }
    out->uses_moe = oc_model_arch_uses_moe(out->arch);

    /* Dominant quant type by weight-byte share, and parameter estimate. */
    OcArena *arena = oc_arena_new(1u << 20);
    if (arena == NULL) return OC_ERR_OOM;
    OcGgufTensorInfo *infos = NULL;
    size_t n = 0;
    OcError e = oc_gguf_map_mapped_tensor_infos(m, arena, &infos, &n);
    if (e != OC_OK) { oc_arena_free(arena); return e; }

    /* Per-qtype byte accumulator. */
    uint64_t bytes_per_qtype[OC_QUANT__COUNT];
    memset(bytes_per_qtype, 0, sizeof(bytes_per_qtype));
    uint64_t total_weight_bytes = 0;
    for (size_t i = 0; i < n; i++) {
        OcGgufQuantizationType qt = oc_quant_type_from_ggml_id(infos[i].ggml_type);
        if (qt >= OC_QUANT__COUNT) continue;
        size_t elems = 1;
        for (uint32_t d = 0; d < infos[i].n_dims; d++) elems *= (size_t)infos[i].dims[d];
        size_t bsz = oc_quantized_size(qt, elems);
        bytes_per_qtype[qt] += bsz;
        total_weight_bytes += bsz;
        /* Parameter estimate: sum of elements across weight tensors (exclude
         * 1-D norms; only 2-D+ count as weight matrices). */
        if (infos[i].n_dims >= 2) {
            out->param_count += elems;
        }
    }
    /* Find dominant qtype. */
    uint64_t best_bytes = 0;
    for (int q = 0; q < (int)OC_QUANT__COUNT; q++) {
        if (bytes_per_qtype[q] > best_bytes) {
            best_bytes = bytes_per_qtype[q];
            out->dominant_qtype = (OcGgufQuantizationType)q;
        }
    }
    out->dominant_qtype_fraction = (total_weight_bytes > 0)
        ? (double)best_bytes / (double)total_weight_bytes : 0.0;

    oc_arena_free(arena);
    return OC_OK;
}

/* ─── Plan (pure function) ────────────────────────────────────────────── */

OcTuningPlan oc_autotune_plan(const OcCpuInfo *cpu,
                              const OcModelFingerprint *model)
{
    OcTuningPlan p;
    memset(&p, 0, sizeof(p));
    p.simd_level = cpu ? cpu->simd.level : OC_SIMD_SCALAR;
    p.use_simd_dequant = (p.simd_level >= OC_SIMD_AVX2);

    /* Memory headroom: model fits in RAM with 2x headroom? */
    uint64_t avail = cpu ? cpu->available_ram_bytes : 0;
    uint64_t msize = model ? model->file_bytes : 0;
    bool fits_2x = (msize > 0 && avail >= msize * 2);
    p.use_hugepages = fits_2x;
    p.mlock_weights = (msize > 0 && avail >= (msize * 10) / 7);  /* 30% headroom */

    /* Threads + NUMA: mirror the learned dual-socket Cascade Lake rules. */
    bool big_model = (msize > (192ULL << 30));   /* > 192 GB */
    if (cpu && cpu->is_dual_socket) {
        if (big_model) {
            p.numa = OC_NUMA_INTERLEAVE;
            p.threads = 48;
            p.rationale_numa = "dual-socket + model >192GB → interleave across sockets";
            p.rationale_threads = "large model on dual-socket → 48 threads";
        } else {
            p.numa = OC_NUMA_SINGLE;
            p.threads = (cpu->logical_cores >= 16) ? 16 : cpu->logical_cores;
            p.rationale_numa = "dual-socket + model <=192GB → single-socket binding";
            p.rationale_threads = "dense model <=192GB → 16 threads (single socket)";
        }
    } else {
        p.numa = OC_NUMA_NONE;
        uint32_t lc = cpu ? cpu->logical_cores : 1;
        p.threads = (lc >= 16) ? 16 : lc;
        p.rationale_numa = "UMA or single-socket → no NUMA policy";
        p.rationale_threads = "UMA → up to 16 threads";
    }

    p.rationale_simd = (p.simd_level == OC_SIMD_AVX512) ? "AVX-512 BW+DQ+VNNI detected → use AVX-512 dequant kernels"
                       : (p.simd_level == OC_SIMD_AVX2) ? "AVX2+FMA+F16C detected → use AVX2 dequant kernels"
                       : "no SIMD acceleration available → scalar fallback";
    if (fits_2x) {
        p.rationale_memory = "available RAM >= 2x model → enable hugepages + mlock";
    } else if (msize > 0) {
        p.rationale_memory = "tight RAM headroom → skip hugepages/mlock";
    } else {
        p.rationale_memory = "no model size known → conservative memory policy";
    }
    return p;
}

const char *oc_autotune_numa_name(OcNumaPolicy p)
{
    switch (p) {
    case OC_NUMA_SINGLE:     return "single";
    case OC_NUMA_INTERLEAVE: return "interleave";
    case OC_NUMA_NONE:
    default:                  return "none";
    }
}

void oc_autotune_plan_dump(const OcTuningPlan *plan,
                           const OcCpuInfo *cpu,
                           const OcModelFingerprint *model)
{
    fprintf(stderr, "=== autotune plan ===\n");
    if (cpu) {
        fprintf(stderr, "cpu:     %s (%u logical, %u NUMA, %llu GB RAM)\n",
                cpu->model_name[0] ? cpu->model_name : "?",
                cpu->logical_cores, cpu->numa_nodes,
                (unsigned long long)(cpu->total_ram_bytes >> 30));
        fprintf(stderr, "simd:    %s\n", cpu->simd.name);
    }
    if (model) {
        fprintf(stderr, "model:   %llu MB, %u layers, arch=%s, qtype=%s (%.0f%%)\n",
                (unsigned long long)(model->file_bytes >> 20),
                model->n_layer, oc_model_arch_name(model->arch),
                oc_quant_type_name(model->dominant_qtype),
                model->dominant_qtype_fraction * 100.0);
    }
    fprintf(stderr, "plan:\n");
    fprintf(stderr, "  threads:     %u   (%s)\n", plan->threads,
            plan->rationale_threads ? plan->rationale_threads : "");
    fprintf(stderr, "  numa:        %s   (%s)\n", oc_autotune_numa_name(plan->numa),
            plan->rationale_numa ? plan->rationale_numa : "");
    fprintf(stderr, "  simd:        %s   (%s)\n",
            plan->simd_level == OC_SIMD_AVX512 ? "avx512"
            : plan->simd_level == OC_SIMD_AVX2 ? "avx2" : "scalar",
            plan->rationale_simd ? plan->rationale_simd : "");
    fprintf(stderr, "  hugepages:   %s\n", plan->use_hugepages ? "yes" : "no");
    fprintf(stderr, "  mlock:       %s\n", plan->mlock_weights ? "yes" : "no");
    fprintf(stderr, "  simd dequant:%s   (%s)\n",
            plan->use_simd_dequant ? "yes" : "no",
            plan->rationale_memory ? plan->rationale_memory : "");
}

/* ─── Apply (autotune-plan-apply feature) ────────────────────────────────
 *
 * Applies the memory-side of the plan (hugepages + mlock) to the mmap'd
 * GGUF. Thread/NUMA policy is caller-side (set via pthread_setaffinity on
 * worker threads) because it must be applied per-thread, not globally. */

OcError oc_autotune_apply(const OcTuningPlan *plan, OcGgufMmappedFile *m)
{
    if (plan == NULL || m == NULL) return OC_ERR_INVALID_ARG;
    if (plan->use_hugepages) {
        /* oc_gguf_map_advise_hugepage is best-effort; logs on failure. */
        OcError e = oc_gguf_map_advise_hugepage(m);
        if (e != OC_OK) {
            oc_log(OC_LOG_WARN, "autotune: hugepage advise failed (%s)",
                   oc_error_msg(e));
        }
    }
    if (plan->mlock_weights) {
        /* oc_gguf_map_mlock_with_headroom already checks the 30% headroom
         * policy internally; returns true on success, false on skip/fail. */
        bool ok = oc_gguf_map_mlock_with_headroom(m);
        if (!ok) {
            oc_log(OC_LOG_WARN, "autotune: mlock skipped (headroom too tight)");
        }
    }
    return OC_OK;
}

OcError oc_autotune_bind_to_numa_node(uint32_t node)
{
#if OC_LINUX
    /* Use libnuma if available, else best-effort via CPU affinity to the
     * node's CPUs. The simplest portable approach is to read the node's
     * CPU list from /sys/devices/system/node/nodeN/cpulist and apply it
     * via sched_setaffinity. */
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/node/node%u/cpulist", node);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        /* Node doesn't exist or single-socket: no-op is fine. */
        return OC_OK;
    }
    char cpulist[256];
    if (fgets(cpulist, sizeof(cpulist), f) == NULL) {
        fclose(f);
        return OC_OK;
    }
    fclose(f);
    /* Parse "a-b" or "a,b,c" into a cpu_set_t. We support the common
     * "a-b" range form; comma lists are handled by iterating. */
    cpu_set_t set;
    CPU_ZERO(&set);
    char *p = cpulist;
    while (*p) {
        unsigned long lo, hi;
        if (sscanf(p, "%lu-%lu", &lo, &hi) == 2) {
            for (unsigned long c = lo; c <= hi && c < CPU_SETSIZE; c++) {
                CPU_SET((int)c, &set);
            }
            /* skip past the range */
            while (*p && *p != ',' && *p != '\n') p++;
        } else if (sscanf(p, "%lu", &lo) == 1) {
            if (lo < CPU_SETSIZE) CPU_SET((int)lo, &set);
            while (*p && *p != ',' && *p != '\n') p++;
        } else {
            p++;
        }
        if (*p == ',' || *p == '\n') p++;
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        oc_log(OC_LOG_WARN, "autotune: sched_setaffinity to node %u failed",
               node);
    }
    return OC_OK;
#else
    (void)node;
    return OC_OK;
#endif
}
