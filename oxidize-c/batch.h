#ifndef OC_BATCH_H
#define OC_BATCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct oc_batch_scheduler oc_batch_scheduler;

typedef struct {
  size_t capacity;
  size_t queue_capacity;
  const int *gpu_ids;
  size_t gpu_count;
} oc_batch_config;

typedef enum {
  OC_SEQUENCE_PREFILL,
  OC_SEQUENCE_DECODE,
  OC_SEQUENCE_TERMINAL,
  OC_SEQUENCE_CANCELLED,
  OC_SEQUENCE_ERROR,
} oc_sequence_state;

typedef struct {
  uint64_t request_id;
  const uint32_t *prompt_tokens;
  size_t prompt_count;
  size_t max_tokens;
  uint64_t rng_seed;
} oc_sequence_request;

typedef struct {
  size_t slot;
  uint64_t request_id;
  uint32_t token;
  size_t position;
  uint64_t generation;
  oc_sequence_state phase;
  bool produces_output;
} oc_batch_lane;

bool oc_batch_config_valid(const oc_batch_config *config);
oc_batch_scheduler *oc_batch_scheduler_new(const oc_batch_config *config);
void oc_batch_scheduler_free(oc_batch_scheduler *scheduler);

bool oc_batch_scheduler_admit_sequence(oc_batch_scheduler *scheduler,
                                       const oc_sequence_request *request,
                                       size_t *slot);
bool oc_batch_scheduler_cancel(oc_batch_scheduler *scheduler, size_t slot);
bool oc_batch_scheduler_next_lanes(oc_batch_scheduler *scheduler, oc_batch_lane *lanes,
                                   size_t lane_capacity, size_t *lane_count);
bool oc_batch_scheduler_finish_lane(oc_batch_scheduler *scheduler,
                                    const oc_batch_lane *lane);
bool oc_batch_scheduler_record_output(oc_batch_scheduler *scheduler,
                                      const oc_batch_lane *lane, uint32_t token, bool eog);
bool oc_batch_scheduler_output_next(oc_batch_scheduler *scheduler, uint64_t *request_id,
                                    uint32_t *token);
bool oc_batch_scheduler_reclaim(oc_batch_scheduler *scheduler, size_t slot);
bool oc_batch_scheduler_next_random(oc_batch_scheduler *scheduler, size_t slot,
                                    uint64_t *value);

bool oc_batch_scheduler_admit(oc_batch_scheduler *scheduler, uint64_t sequence_id,
                              size_t *slot);
bool oc_batch_scheduler_enqueue_token(oc_batch_scheduler *scheduler, size_t slot,
                                      uint32_t token);
bool oc_batch_scheduler_next(oc_batch_scheduler *scheduler, uint64_t *sequence_id,
                             uint32_t *token);
bool oc_batch_scheduler_complete(oc_batch_scheduler *scheduler, size_t slot);

#endif
