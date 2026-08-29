#ifndef OXIDIZE_PROFILER_H
#define OXIDIZE_PROFILER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    OC_PROF_EMBEDDING    = 0,
    OC_PROF_RMSNORM      = 1,
    OC_PROF_ROPE         = 2,
    OC_PROF_ATTENTION    = 3,
    OC_PROF_MLP_GATE     = 4,
    OC_PROF_MLP_UP       = 5,
    OC_PROF_MLP_DOWN     = 6,
    OC_PROF_SWIGLU       = 7,
    OC_PROF_RESIDUAL     = 8,
    OC_PROF_SAMPLING     = 9,
    OC_PROF_MATVEC       = 10,
    OC_PROF_DEQUANT      = 11,
    OC_PROF_KV_CACHE     = 12,
    OC_PROF_LOGITS       = 13,
    OC_PROF_TOTAL        = 14,
    OC_PROF__COUNT,
} OcProfEvent;


typedef struct OcProfileEntry {
    uint64_t total_ns;    /* total time in nanoseconds             */
    uint64_t count;        /* number of calls                       */
    uint64_t min_ns;       /* minimum duration                       */
    uint64_t max_ns;       /* maximum duration                       */
} OcProfileEntry;

typedef struct OcProfiler {
    OcProfileEntry entries[OC_PROF__COUNT];
    bool enabled;
    uint64_t tokens_profiled;
} OcProfiler;

/* Initialize a profiler. */
void oc_profiler_init(OcProfiler *p);

/* Enable/disable profiling. */
void oc_profiler_enable(OcProfiler *p, bool enabled);

/* Record a timing. `ns` is the duration in nanoseconds. */
void oc_profiler_record(OcProfiler *p, OcProfEvent event, uint64_t ns);

/* Get the total time for an event type. */
uint64_t oc_profiler_total_ns(const OcProfiler *p, OcProfEvent event);

/* Get the average time for an event type (ns). */
double oc_profiler_avg_ns(const OcProfiler *p, OcProfEvent event);

/* Get the percentage of total time for an event type. */
double oc_profiler_pct(const OcProfiler *p, OcProfEvent event);

/* Format profiler results as a human-readable table. */
size_t oc_profiler_format(const OcProfiler *p, char *buf, size_t cap);

/* Format as JSON. */
size_t oc_profiler_format_json(const OcProfiler *p, char *buf, size_t cap);

/* Reset all counters. */
void oc_profiler_reset(OcProfiler *p);

/* Print to stderr. */
void oc_profiler_print(const OcProfiler *p);

/* Get event name. */
const char *oc_prof_event_name(OcProfEvent e);


typedef struct OcProfileScope {
    OcProfiler *prof;
    OcProfEvent event;
    uint64_t start_ns;
} OcProfileScope;

/* Get current time in nanoseconds. */
uint64_t oc_prof_now_ns(void);

/* Begin a scoped timing. */
void oc_prof_scope_begin(OcProfileScope *s, OcProfiler *p, OcProfEvent e);

/* End a scoped timing (records the duration). */
void oc_prof_scope_end(OcProfileScope *s);

#define OC_PROFILE_SCOPE(prof_ptr, event) \
    OcProfileScope _oc_prof_scope; \
    oc_prof_scope_begin(&_oc_prof_scope, prof_ptr, event); \
    /* end at scope exit via cleanup attribute */ \
    __attribute__((cleanup(oc_prof_scope_end))) OcProfileScope *_oc_prof_scope_ptr \
        __attribute__((unused)) = &_oc_prof_scope

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PROFILER_H */
