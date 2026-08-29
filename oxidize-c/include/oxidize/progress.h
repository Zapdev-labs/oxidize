/* progress.h — Progress tracking for long-running operations. */
#ifndef OXIDIZE_PROGRESS_H
#define OXIDIZE_PROGRESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_PROGRESS_MAX_STAGES 32
#define OC_PROGRESS_MAX_NAME 128

typedef struct {
    char name[OC_PROGRESS_MAX_NAME];
    uint64_t total;
    uint64_t completed;
    bool done;
    bool failed;
} OcProgressStage;

typedef struct {
    OcProgressStage stages[OC_PROGRESS_MAX_STAGES];
    uint32_t n_stages;
    uint32_t current_stage;
    uint64_t start_time_ms;
    uint64_t elapsed_ms;
    bool running;
    bool cancelled;
} OcProgress;

typedef void (*OcProgressCb)(const OcProgress *prog, void *user);

OcError oc_progress_init(OcProgress *prog);
OcError oc_progress_add_stage(OcProgress *prog, const char *name, uint64_t total);
OcError oc_progress_update(OcProgress *prog, uint64_t completed);
OcError oc_progress_advance(OcProgress *prog);
OcError oc_progress_complete(OcProgress *prog);
OcError oc_progress_fail(OcProgress *prog);
OcError oc_progress_cancel(OcProgress *prog);
OcError oc_progress_get_current(const OcProgress *prog, const OcProgressStage **out);
float oc_progress_percent(const OcProgress *prog);
uint64_t oc_progress_elapsed_ms(const OcProgress *prog);
bool oc_progress_is_done(const OcProgress *prog);
bool oc_progress_is_running(const OcProgress *prog);
const char *oc_progress_stage_name(const OcProgress *prog);
void oc_progress_free(OcProgress *prog);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PROGRESS_H */
