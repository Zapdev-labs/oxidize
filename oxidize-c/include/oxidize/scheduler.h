/* scheduler.h — Higher-level request scheduler for paged attention. */
#ifndef OXIDIZE_SCHEDULER_H
#define OXIDIZE_SCHEDULER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_SCHED_DEFAULT_MAX_BATCH_SIZE    32u
#define OC_SCHED_DEFAULT_MAX_TOKENS_TOTAL  8192u
#define OC_SCHED_INITIAL_CAPACITY          16u


typedef enum {
    OC_SCHED_PRIORITY_LOW    = 0,
    OC_SCHED_PRIORITY_NORMAL = 1,
    OC_SCHED_PRIORITY_HIGH   = 2,
} OcSchedPriority;


typedef enum {
    OC_SCHED_STATUS_PENDING   = 0,
    OC_SCHED_STATUS_RUNNING    = 1,
    OC_SCHED_STATUS_COMPLETED = 2,
    OC_SCHED_STATUS_CANCELLED = 3,
} OcSchedStatus;


typedef enum {
    OC_SCHED_PREEMPT_RECOMPUTE = 0,
    OC_SCHED_PREEMPT_SWAP      = 1,
} OcSchedPreemptMode;


/* A single inference request. The scheduler copies prompt_tokens (caller
 * retains ownership of the input array). */
typedef struct OcSchedRequest {
    uint64_t            id;
    uint32_t           *prompt_tokens;   /* owned copy */
    uint32_t            n_prompt;
    uint32_t            max_tokens;
    OcSchedPriority     priority;
    uint64_t            created_ms;
    OcSchedStatus       status;
} OcSchedRequest;


typedef struct OcSchedConfig {
    uint32_t            max_batch_size;             /* default 32            */
    uint32_t            max_tokens_total;            /* default 8192          */
    OcSchedPreemptMode  preempt_mode;               /* default RECOMPUTE     */
    bool                enable_continuous_batching;  /* default true          */
} OcSchedConfig;


typedef struct OcScheduler {
    OcSchedConfig       config;
    OcSchedRequest    **requests;       /* dynamic array of owned pointers   */
    uint32_t            capacity;       /* allocated slots                   */
    uint32_t            n_requests;     /* total in array                    */
    uint32_t            n_running;
    uint32_t            n_completed;
    uint64_t            next_id;        /* monotonic id generator            */
} OcScheduler;


/* Initialize config with defaults. Returns OC_ERR_INVALID_ARG if cfg NULL. */
OcError oc_sched_config_init(OcSchedConfig *cfg);


/* Initialize a request struct. Copies prompt_tokens (caller retains
 * ownership). Returns OC_ERR_INVALID_ARG on bad args, OC_ERR_OOM on
 * allocation failure. */
OcError oc_sched_request_init(OcSchedRequest *req, uint64_t id,
                               const uint32_t *prompt_tokens, uint32_t n_prompt,
                               uint32_t max_tokens, OcSchedPriority priority,
                               uint64_t created_ms);

/* Free owned fields of a request (does not free the struct itself). */
void oc_sched_request_free(OcSchedRequest *req);


/* Initialize a scheduler with the given config (or defaults if NULL). */
OcError oc_sched_init(OcScheduler *sched, const OcSchedConfig *cfg);

/* Add a request to the scheduler. The request is copied; the scheduler
 * owns the copy. `out_id` receives the assigned id (may be NULL).
 * Returns OC_ERR_INVALID_ARG on bad args, OC_ERR_OOM if full or alloc fail. */
OcError oc_sched_add_request(OcScheduler *sched, const OcSchedRequest *req,
                               uint64_t *out_id);

/* Cancel a request by id. Idempotent on already-cancelled requests.
 * Returns OC_ERR_INVALID_ARG for unknown id. */
OcError oc_sched_cancel_request(OcScheduler *sched, uint64_t id);

/* Get the next batch of request ids to process. Selects PENDING requests */
OcError oc_sched_next_batch(OcScheduler *sched, uint64_t *out_ids,
                             uint32_t max_batch, uint32_t *out_n);

/* Mark a request as completed. Idempotent on already-completed requests.
 * Returns OC_ERR_INVALID_ARG for unknown id. */
OcError oc_sched_complete_request(OcScheduler *sched, uint64_t id);


uint32_t oc_sched_n_pending(const OcScheduler *sched);
uint32_t oc_sched_n_running(const OcScheduler *sched);
uint32_t oc_sched_n_completed(const OcScheduler *sched);


/* Free the scheduler and all owned requests. Safe on NULL. */
void oc_sched_free(OcScheduler *sched);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SCHEDULER_H */
