#include "batch.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t request_id, rng_state, generation;
  uint32_t *prompt, *outputs;
  size_t prompt_count, prompt_cursor, position, generated, max_tokens;
  size_t output_head, output_count;
  uint32_t decode_token;
  oc_sequence_state state;
  bool active, in_flight;
} oc_sequence;

typedef struct {
  oc_sequence_request request;
  uint32_t *prompt;
} oc_pending_sequence;

struct oc_batch_scheduler {
  size_t capacity, queue_capacity, pending_head, pending_count, next_slot, next_output;
  uint64_t next_generation;
  oc_sequence *slots;
  oc_pending_sequence *pending;
};

static bool oc_size_product_ok(size_t left, size_t right) {
  return !left || right <= SIZE_MAX / left;
}

bool oc_batch_config_valid(const oc_batch_config *config) {
  if (!config || !config->capacity || config->queue_capacity < config->capacity ||
      !config->gpu_ids || !config->gpu_count ||
      !oc_size_product_ok(config->capacity, config->queue_capacity)) return false;
  for (size_t i = 0; i < config->gpu_count; ++i) {
    if (config->gpu_ids[i] < 0) return false;
    for (size_t j = 0; j < i; ++j)
      if (config->gpu_ids[i] == config->gpu_ids[j]) return false;
  }
  return true;
}

static void oc_sequence_clear(oc_sequence *sequence) {
  free(sequence->prompt);
  uint32_t *outputs = sequence->outputs;
  memset(sequence, 0, sizeof(*sequence));
  sequence->outputs = outputs;
}

static bool oc_copy_prompt(const uint32_t *source, size_t count, uint32_t **destination) {
  *destination = NULL;
  if (!count) return true;
  if (!source || count > SIZE_MAX / sizeof(**destination)) return false;
  uint32_t *copy = malloc(count * sizeof(*copy));
  if (!copy) return false;
  memcpy(copy, source, count * sizeof(*copy));
  *destination = copy;
  return true;
}

static bool oc_request_valid(const oc_sequence_request *request) {
  return request && request->max_tokens &&
         (!request->prompt_count || request->prompt_tokens);
}

static bool oc_id_exists(const oc_batch_scheduler *scheduler, uint64_t request_id) {
  for (size_t i = 0; i < scheduler->capacity; ++i)
    if (scheduler->slots[i].active && scheduler->slots[i].request_id == request_id) return true;
  for (size_t i = 0; i < scheduler->pending_count; ++i) {
    size_t index = (scheduler->pending_head + i) % scheduler->queue_capacity;
    if (scheduler->pending[index].request.request_id == request_id) return true;
  }
  return false;
}

static void oc_sequence_start(oc_batch_scheduler *scheduler, oc_sequence *sequence,
                              const oc_sequence_request *request, uint32_t *prompt) {
  uint32_t *outputs = sequence->outputs;
  memset(sequence, 0, sizeof(*sequence));
  sequence->outputs = outputs;
  sequence->request_id = request->request_id;
  sequence->prompt = prompt;
  sequence->prompt_count = request->prompt_count;
  sequence->max_tokens = request->max_tokens;
  sequence->rng_state = request->rng_seed ? request->rng_seed : UINT64_C(0x9e3779b97f4a7c15);
  sequence->generation = ++scheduler->next_generation;
  if (!sequence->generation) sequence->generation = ++scheduler->next_generation;
  sequence->state = request->prompt_count ? OC_SEQUENCE_PREFILL : OC_SEQUENCE_DECODE;
  sequence->active = true;
}

static bool oc_output_push(const oc_batch_scheduler *scheduler, oc_sequence *sequence,
                           uint32_t token) {
  if (sequence->output_count == scheduler->queue_capacity) return false;
  size_t tail = (sequence->output_head + sequence->output_count) % scheduler->queue_capacity;
  sequence->outputs[tail] = token;
  ++sequence->output_count;
  return true;
}

static bool oc_lane_matches(const oc_sequence *sequence, const oc_batch_lane *lane) {
  return lane && sequence->active && sequence->in_flight &&
         sequence->generation == lane->generation && sequence->request_id == lane->request_id;
}

static void oc_promote_pending(oc_batch_scheduler *scheduler) {
  while (scheduler->pending_count) {
    size_t slot = scheduler->capacity;
    for (size_t i = 0; i < scheduler->capacity; ++i)
      if (!scheduler->slots[i].active) { slot = i; break; }
    if (slot == scheduler->capacity) return;
    oc_pending_sequence *pending = &scheduler->pending[scheduler->pending_head];
    oc_sequence_start(scheduler, &scheduler->slots[slot], &pending->request, pending->prompt);
    pending->prompt = NULL;
    memset(&pending->request, 0, sizeof(pending->request));
    scheduler->pending_head = (scheduler->pending_head + 1) % scheduler->queue_capacity;
    --scheduler->pending_count;
  }
}

oc_batch_scheduler *oc_batch_scheduler_new(const oc_batch_config *config) {
  if (!oc_batch_config_valid(config)) return NULL;
  oc_batch_scheduler *scheduler = calloc(1, sizeof(*scheduler));
  if (!scheduler) return NULL;
  scheduler->slots = calloc(config->capacity, sizeof(*scheduler->slots));
  scheduler->pending = calloc(config->queue_capacity, sizeof(*scheduler->pending));
  uint32_t *outputs = calloc(config->capacity * config->queue_capacity, sizeof(*outputs));
  if (!scheduler->slots || !scheduler->pending || !outputs) {
    free(outputs); free(scheduler->pending); free(scheduler->slots); free(scheduler);
    return NULL;
  }
  for (size_t i = 0; i < config->capacity; ++i)
    scheduler->slots[i].outputs = outputs + i * config->queue_capacity;
  scheduler->capacity = config->capacity;
  scheduler->queue_capacity = config->queue_capacity;
  return scheduler;
}

void oc_batch_scheduler_free(oc_batch_scheduler *scheduler) {
  if (!scheduler) return;
  for (size_t i = 0; i < scheduler->capacity; ++i) free(scheduler->slots[i].prompt);
  for (size_t i = 0; i < scheduler->queue_capacity; ++i) free(scheduler->pending[i].prompt);
  free(scheduler->slots ? scheduler->slots[0].outputs : NULL);
  free(scheduler->pending); free(scheduler->slots); free(scheduler);
}

bool oc_batch_scheduler_admit_sequence(oc_batch_scheduler *scheduler,
                                       const oc_sequence_request *request, size_t *slot) {
  if (!scheduler || !slot || !oc_request_valid(request) || oc_id_exists(scheduler, request->request_id))
    return false;
  for (size_t i = 0; i < scheduler->capacity; ++i) {
    if (!scheduler->slots[i].active) {
      uint32_t *prompt = NULL;
      if (!oc_copy_prompt(request->prompt_tokens, request->prompt_count, &prompt)) return false;
      oc_sequence_start(scheduler, &scheduler->slots[i], request, prompt);
      *slot = i;
      return true;
    }
  }
  if (scheduler->pending_count == scheduler->queue_capacity) return false;
  size_t tail = (scheduler->pending_head + scheduler->pending_count) % scheduler->queue_capacity;
  if (!oc_copy_prompt(request->prompt_tokens, request->prompt_count, &scheduler->pending[tail].prompt))
    return false;
  scheduler->pending[tail].request = *request;
  scheduler->pending[tail].request.prompt_tokens = scheduler->pending[tail].prompt;
  ++scheduler->pending_count;
  *slot = SIZE_MAX;
  return true;
}

bool oc_batch_scheduler_cancel(oc_batch_scheduler *scheduler, size_t slot) {
  if (!scheduler || slot >= scheduler->capacity || !scheduler->slots[slot].active) return false;
  oc_sequence *sequence = &scheduler->slots[slot];
  sequence->state = OC_SEQUENCE_CANCELLED;
  sequence->in_flight = false;
  sequence->output_head = sequence->output_count = 0;
  return true;
}

bool oc_batch_scheduler_next_lanes(oc_batch_scheduler *scheduler, oc_batch_lane *lanes,
                                   size_t lane_capacity, size_t *lane_count) {
  if (!scheduler || !lanes || !lane_capacity || !lane_count) return false;
  oc_promote_pending(scheduler);
  *lane_count = 0;
  size_t start_slot = scheduler->next_slot;
  for (size_t checked = 0; checked < scheduler->capacity && *lane_count < lane_capacity; ++checked) {
    size_t slot = (start_slot + checked) % scheduler->capacity;
    oc_sequence *sequence = &scheduler->slots[slot];
    if (!sequence->active || sequence->in_flight ||
        (sequence->state != OC_SEQUENCE_PREFILL && sequence->state != OC_SEQUENCE_DECODE)) continue;
    oc_sequence_state phase = sequence->state;
    uint32_t token = phase == OC_SEQUENCE_PREFILL ? sequence->prompt[sequence->prompt_cursor++]
                                                   : sequence->decode_token;
    bool produces_output = phase == OC_SEQUENCE_DECODE ||
                           sequence->prompt_cursor == sequence->prompt_count;
    lanes[*lane_count] = (oc_batch_lane){slot, sequence->request_id, token, sequence->position++,
                                         sequence->generation, phase, produces_output};
    if (phase == OC_SEQUENCE_PREFILL && sequence->prompt_cursor == sequence->prompt_count) {
      sequence->state = OC_SEQUENCE_DECODE;
      sequence->decode_token = token;
    }
    sequence->in_flight = true;
    ++*lane_count;
    scheduler->next_slot = (slot + 1) % scheduler->capacity;
  }
  return *lane_count != 0;
}

bool oc_batch_scheduler_finish_lane(oc_batch_scheduler *scheduler, const oc_batch_lane *lane) {
  if (!scheduler || !lane || lane->slot >= scheduler->capacity) return false;
  oc_sequence *sequence = &scheduler->slots[lane->slot];
  if (!oc_lane_matches(sequence, lane)) return false;
  sequence->in_flight = false;
  return true;
}

bool oc_batch_scheduler_record_output(oc_batch_scheduler *scheduler,
                                      const oc_batch_lane *lane, uint32_t token, bool eog) {
  if (!scheduler || !lane || lane->slot >= scheduler->capacity) return false;
  oc_sequence *sequence = &scheduler->slots[lane->slot];
  if (!lane->produces_output || !oc_lane_matches(sequence, lane)) return false;
  sequence->decode_token = token;
  sequence->in_flight = false;
  ++sequence->generated;
  if (eog) {
    sequence->state = OC_SEQUENCE_TERMINAL;
    return true;
  }
  if (!oc_output_push(scheduler, sequence, token)) {
    sequence->state = OC_SEQUENCE_ERROR;
    return false;
  }
  if (sequence->generated == sequence->max_tokens) sequence->state = OC_SEQUENCE_TERMINAL;
  return true;
}

bool oc_batch_scheduler_output_next(oc_batch_scheduler *scheduler, uint64_t *request_id,
                                    uint32_t *token) {
  if (!scheduler || !request_id || !token) return false;
  for (size_t checked = 0; checked < scheduler->capacity; ++checked) {
    size_t slot = (scheduler->next_output + checked) % scheduler->capacity;
    oc_sequence *sequence = &scheduler->slots[slot];
    if (!sequence->active || !sequence->output_count) continue;
    *request_id = sequence->request_id;
    *token = sequence->outputs[sequence->output_head];
    sequence->output_head = (sequence->output_head + 1) % scheduler->queue_capacity;
    --sequence->output_count;
    scheduler->next_output = (slot + 1) % scheduler->capacity;
    return true;
  }
  return false;
}

bool oc_batch_scheduler_reclaim(oc_batch_scheduler *scheduler, size_t slot) {
  if (!scheduler || slot >= scheduler->capacity) return false;
  oc_sequence *sequence = &scheduler->slots[slot];
  if (!sequence->active || sequence->in_flight || sequence->output_count ||
      (sequence->state != OC_SEQUENCE_TERMINAL && sequence->state != OC_SEQUENCE_CANCELLED &&
       sequence->state != OC_SEQUENCE_ERROR)) return false;
  oc_sequence_clear(sequence);
  return true;
}

bool oc_batch_scheduler_next_random(oc_batch_scheduler *scheduler, size_t slot, uint64_t *value) {
  if (!scheduler || !value || slot >= scheduler->capacity || !scheduler->slots[slot].active) return false;
  uint64_t state = scheduler->slots[slot].rng_state;
  state ^= state << 13; state ^= state >> 7; state ^= state << 17;
  scheduler->slots[slot].rng_state = state;
  *value = state; return true;
}

bool oc_batch_scheduler_admit(oc_batch_scheduler *scheduler, uint64_t sequence_id, size_t *slot) {
  oc_sequence_request request = {sequence_id, NULL, 0, SIZE_MAX, sequence_id};
  if (!scheduler || !slot || oc_id_exists(scheduler, sequence_id)) return false;
  for (size_t i = 0; i < scheduler->capacity; ++i) if (!scheduler->slots[i].active) {
    oc_sequence_start(scheduler, &scheduler->slots[i], &request, NULL); *slot = i; return true;
  }
  return false;
}

bool oc_batch_scheduler_enqueue_token(oc_batch_scheduler *scheduler, size_t slot, uint32_t token) {
  return scheduler && slot < scheduler->capacity && scheduler->slots[slot].active &&
         oc_output_push(scheduler, &scheduler->slots[slot], token);
}

bool oc_batch_scheduler_next(oc_batch_scheduler *scheduler, uint64_t *sequence_id, uint32_t *token) {
  return oc_batch_scheduler_output_next(scheduler, sequence_id, token);
}

bool oc_batch_scheduler_complete(oc_batch_scheduler *scheduler, size_t slot) {
  if (!scheduler || slot >= scheduler->capacity || !scheduler->slots[slot].active ||
      scheduler->slots[slot].in_flight || scheduler->slots[slot].output_count) return false;
  scheduler->slots[slot].state = OC_SEQUENCE_TERMINAL;
  return oc_batch_scheduler_reclaim(scheduler, slot);
}
