/*
 * profiler.c — Inference profiler implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/profiler.h"

#include <stdio.h>
#include <string.h>
#include <time.h>


uint64_t oc_prof_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}


void oc_profiler_init(OcProfiler *p)
{
    if (!p) return;
    memset(p, 0, sizeof(*p));
    p->enabled = true;
}

void oc_profiler_enable(OcProfiler *p, bool enabled)
{
    if (!p) return;
    p->enabled = enabled;
}

void oc_profiler_record(OcProfiler *p, OcProfEvent event, uint64_t ns)
{
    if (!p || !p->enabled || event >= OC_PROF__COUNT) return;
    OcProfileEntry *e = &p->entries[event];
    e->total_ns += ns;
    e->count++;
    if (ns < e->min_ns || e->min_ns == 0) e->min_ns = ns;
    if (ns > e->max_ns) e->max_ns = ns;
    if (event != OC_PROF_TOTAL) {
        p->entries[OC_PROF_TOTAL].total_ns += ns;
        p->entries[OC_PROF_TOTAL].count++;
    }
}

uint64_t oc_profiler_total_ns(const OcProfiler *p, OcProfEvent event)
{
    if (!p || event >= OC_PROF__COUNT) return 0;
    return p->entries[event].total_ns;
}

double oc_profiler_avg_ns(const OcProfiler *p, OcProfEvent event)
{
    if (!p || event >= OC_PROF__COUNT || p->entries[event].count == 0)
        return 0.0;
    return (double)p->entries[event].total_ns / (double)p->entries[event].count;
}

double oc_profiler_pct(const OcProfiler *p, OcProfEvent event)
{
    if (!p || event >= OC_PROF__COUNT) return 0.0;
    uint64_t total = p->entries[OC_PROF_TOTAL].total_ns;
    if (total == 0) return 0.0;
    return 100.0 * (double)p->entries[event].total_ns / (double)total;
}

void oc_profiler_reset(OcProfiler *p)
{
    if (!p) return;
    memset(p->entries, 0, sizeof(p->entries));
    p->tokens_profiled = 0;
}


void oc_prof_scope_begin(OcProfileScope *s, OcProfiler *p, OcProfEvent e)
{
    if (!s) return;
    s->prof = p;
    s->event = e;
    s->start_ns = oc_prof_now_ns();
}

void oc_prof_scope_end(OcProfileScope *s)
{
    if (!s || !s->prof) return;
    uint64_t elapsed = oc_prof_now_ns() - s->start_ns;
    oc_profiler_record(s->prof, s->event, elapsed);
    s->prof = NULL; /* prevent double-record */
}


const char *oc_prof_event_name(OcProfEvent e)
{
    static const char *names[] = {
        "embedding", "rmsnorm", "rope", "attention",
        "mlp_gate", "mlp_up", "mlp_down", "swiglu",
        "residual", "sampling", "matvec", "dequant",
        "kv_cache", "logits", "total",
    };
    if (e >= OC_PROF__COUNT) return "unknown";
    return names[e];
}


size_t oc_profiler_format(const OcProfiler *p, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "┌──────────────────────────────────────────────────────────┐\n"
        "│ Inference Profiler Breakdown                              │\n"
        "├──────────────┬──────────┬──────────┬──────────┬─────────┤\n"
        "│ Operation    │ Total ms │ Calls    │ Avg us   │ %%       │\n"
        "├──────────────┼──────────┼──────────┼──────────┼─────────┤\n");
    if (n < 0 || (size_t)n >= cap) return 0;
    size_t off = (size_t)n;

    for (int i = 0; i < OC_PROF_TOTAL; i++) {
        const OcProfileEntry *e = &p->entries[i];
        if (e->count == 0) continue;
        double total_ms = (double)e->total_ns / 1e6;
        double avg_us = (double)e->total_ns / (double)e->count / 1e3;
        double pct = oc_profiler_pct(p, (OcProfEvent)i);
        int w = snprintf(buf + off, cap - off,
            "│ %-12s │ %8.2f │ %8llu │ %8.2f │ %6.1f%% │\n",
            oc_prof_event_name((OcProfEvent)i),
            total_ms, (unsigned long long)e->count,
            avg_us, pct);
        if (w < 0 || (size_t)w >= cap - off) return 0;
        off += (size_t)w;
    }

    uint64_t total = p->entries[OC_PROF_TOTAL].total_ns;
    double total_ms = (double)total / 1e6;
    int w = snprintf(buf + off, cap - off,
        "├──────────────┼──────────┼──────────┼──────────┼─────────┤\n"
        "│ %-12s │ %8.2f │ %8llu │          │ 100.0%% │\n"
        "└──────────────┴──────────┴──────────┴──────────┴─────────┘\n",
        "total", total_ms, (unsigned long long)p->entries[OC_PROF_TOTAL].count);
    if (w < 0 || (size_t)w >= cap - off) return 0;
    off += (size_t)w;

    return off;
}

size_t oc_profiler_format_json(const OcProfiler *p, char *buf, size_t cap)
{
    int n = snprintf(buf, cap, "{\"events\":[");
    if (n < 0 || (size_t)n >= cap) return 0;
    size_t off = (size_t)n;

    for (int i = 0; i < OC_PROF__COUNT; i++) {
        const OcProfileEntry *e = &p->entries[i];
        if (e->count == 0) continue;
        int w = snprintf(buf + off, cap - off,
            "%s{\"name\":\"%s\",\"total_ns\":%llu,\"count\":%llu,"
            "\"min_ns\":%llu,\"max_ns\":%llu,\"avg_ns\":%.1f,\"pct\":%.1f}",
            off > 10 ? "," : "",
            oc_prof_event_name((OcProfEvent)i),
            (unsigned long long)e->total_ns,
            (unsigned long long)e->count,
            (unsigned long long)e->min_ns,
            (unsigned long long)e->max_ns,
            oc_profiler_avg_ns(p, (OcProfEvent)i),
            oc_profiler_pct(p, (OcProfEvent)i));
        if (w < 0 || (size_t)w >= cap - off) return 0;
        off += (size_t)w;
    }

    int w2 = snprintf(buf + off, cap - off, "],\"tokens_profiled\":%llu}",
                      (unsigned long long)p->tokens_profiled);
    if (w2 < 0 || (size_t)w2 >= cap - off) return 0;
    off += (size_t)w2;
    return off;
}

void oc_profiler_print(const OcProfiler *p)
{
    char buf[4096];
    if (oc_profiler_format(p, buf, sizeof(buf)) > 0)
        fprintf(stderr, "%s", buf);
}
