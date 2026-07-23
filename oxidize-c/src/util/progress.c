/*
 * progress.c — Progress tracking implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/progress.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_progress_init(OcProgress *prog)
{
    if (!prog) return OC_ERR_INVALID_ARG;
    memset(prog, 0, sizeof(*prog));
    return OC_OK;
}

OcError oc_progress_add_stage(OcProgress *prog, const char *name, uint64_t total)
{
    if (!prog || !name) return OC_ERR_INVALID_ARG;
    if (prog->n_stages >= OC_PROGRESS_MAX_STAGES) return OC_ERR_OOM;

    OcProgressStage *s = &prog->stages[prog->n_stages];
    memset(s, 0, sizeof(*s));
    copy_str(s->name, sizeof(s->name), name);
    s->total = total;
    s->completed = 0;
    s->done = false;
    s->failed = false;
    prog->n_stages++;

    if (!prog->running) {
        prog->running = true;
        prog->current_stage = 0;
        prog->start_time_ms = now_ms();
    }
    return OC_OK;
}

OcError oc_progress_update(OcProgress *prog, uint64_t completed)
{
    if (!prog || prog->n_stages == 0) return OC_ERR_INVALID_ARG;
    if (prog->cancelled) return OC_ERR_MODEL;
    OcProgressStage *s = &prog->stages[prog->current_stage];
    s->completed = completed;
    if (s->total > 0 && completed >= s->total) {
        s->done = true;
    }
    prog->elapsed_ms = now_ms() - prog->start_time_ms;
    return OC_OK;
}

OcError oc_progress_advance(OcProgress *prog)
{
    if (!prog) return OC_ERR_INVALID_ARG;
    if (prog->current_stage >= prog->n_stages) return OC_ERR_MODEL;
    prog->stages[prog->current_stage].done = true;
    prog->current_stage++;
    if (prog->current_stage >= prog->n_stages) {
        prog->running = false;
    }
    return OC_OK;
}

OcError oc_progress_complete(OcProgress *prog)
{
    if (!prog) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < prog->n_stages; i++)
        prog->stages[i].done = true;
    prog->running = false;
    prog->current_stage = prog->n_stages;
    prog->elapsed_ms = now_ms() - prog->start_time_ms;
    return OC_OK;
}

OcError oc_progress_fail(OcProgress *prog)
{
    if (!prog) return OC_ERR_INVALID_ARG;
    if (prog->current_stage < prog->n_stages)
        prog->stages[prog->current_stage].failed = true;
    prog->running = false;
    return OC_OK;
}

OcError oc_progress_cancel(OcProgress *prog)
{
    if (!prog) return OC_ERR_INVALID_ARG;
    prog->cancelled = true;
    prog->running = false;
    return OC_OK;
}

OcError oc_progress_get_current(const OcProgress *prog, const OcProgressStage **out)
{
    if (!prog || !out) return OC_ERR_INVALID_ARG;
    if (prog->current_stage >= prog->n_stages) return OC_ERR_MODEL;
    *out = &prog->stages[prog->current_stage];
    return OC_OK;
}

float oc_progress_percent(const OcProgress *prog)
{
    if (!prog || prog->n_stages == 0) return 0.0f;
    uint64_t total_total = 0;
    uint64_t total_completed = 0;
    for (uint32_t i = 0; i < prog->n_stages; i++) {
        total_total += prog->stages[i].total;
        total_completed += prog->stages[i].completed;
    }
    if (total_total == 0) return 0.0f;
    float pct = 100.0f * (float)total_completed / (float)total_total;
    if (pct > 100.0f) pct = 100.0f;
    return pct;
}

uint64_t oc_progress_elapsed_ms(const OcProgress *prog)
{
    if (!prog) return 0;
    if (prog->running)
        return now_ms() - prog->start_time_ms;
    return prog->elapsed_ms;
}

bool oc_progress_is_done(const OcProgress *prog)
{
    if (!prog || prog->n_stages == 0) return false;
    for (uint32_t i = 0; i < prog->n_stages; i++)
        if (!prog->stages[i].done) return false;
    return true;
}

bool oc_progress_is_running(const OcProgress *prog)
{
    return prog ? prog->running : false;
}

const char *oc_progress_stage_name(const OcProgress *prog)
{
    if (!prog || prog->current_stage >= prog->n_stages) return "";
    return prog->stages[prog->current_stage].name;
}

void oc_progress_free(OcProgress *prog)
{
    if (!prog) return;
    memset(prog, 0, sizeof(*prog));
}
