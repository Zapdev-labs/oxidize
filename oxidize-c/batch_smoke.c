#include "oc.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  DEFAULT_CONTEXT = 128,
  DEFAULT_SLOTS = 2,
  BATCH_STEPS = 4,
  BENCH_WARMUP_STEPS = 2,
  MAX_BENCH_SLOTS = 64,
};

typedef struct {
  const char *model_path;
  size_t context;
  size_t slots;
  size_t bench_steps;
  bool bench_only;
} smoke_options;

static void usage(const char *program) {
  fprintf(stderr, "usage: %s --model <GGUF> [--ctx N] [--slots N] "
                  "[--bench-only --bench-steps N]\n", program);
}

static int parse_size(const char *text, size_t *value) {
  char *end = NULL;
  errno = 0;
  unsigned long long parsed = strtoull(text, &end, 10);
  if (errno || !text[0] || !end || *end || parsed > SIZE_MAX) return -1;
  *value = (size_t)parsed;
  return 0;
}

static int parse_options(int argc, char **argv, smoke_options *options) {
  *options = (smoke_options){.context = DEFAULT_CONTEXT, .slots = DEFAULT_SLOTS};
  for (int index = 1; index < argc; ++index) {
    const char *argument = argv[index];
    if (!strcmp(argument, "--model") && index + 1 < argc) {
      options->model_path = argv[++index];
    } else if (!strcmp(argument, "--ctx") && index + 1 < argc) {
      if (parse_size(argv[++index], &options->context) != 0) return -1;
    } else if (!strcmp(argument, "--slots") && index + 1 < argc) {
      if (parse_size(argv[++index], &options->slots) != 0) return -1;
    } else if (!strcmp(argument, "--bench-only")) {
      options->bench_only = true;
    } else if (!strcmp(argument, "--bench-steps") && index + 1 < argc) {
      if (parse_size(argv[++index], &options->bench_steps) != 0) return -1;
    } else {
      return -1;
    }
  }
  if (!options->model_path || options->context < BATCH_STEPS + 1 ||
      options->context > DEFAULT_CONTEXT || options->slots < DEFAULT_SLOTS)
    return -1;
  if (options->bench_only &&
      (!options->bench_steps || options->slots > MAX_BENCH_SLOTS ||
       options->bench_steps >= options->context - BENCH_WARMUP_STEPS))
    return -1;
  if (!options->bench_only && options->bench_steps) return -1;
  return 0;
}

static uint32_t greedy_id(const float *logits, size_t vocab_size) {
  uint32_t best = 0;
  for (size_t id = 1; id < vocab_size; ++id)
    if (logits[id] > logits[best]) best = (uint32_t)id;
  return best;
}

static bool batch_weight_present(const oc_weight *weight) {
  return weight && weight->rows && weight->cols && (weight->f32 || weight->data);
}

static void dump_batch_contract(const oc_model *model, size_t context) {
  if (!model) {
    puts("CONTRACT fail model=null");
    fflush(stdout);
    return;
  }
  printf("CONTRACT model gemma=%d kv_int8=%d kv_ctx=%zu ctx=%zu layers=%zu hidden=%zu "
         "heads=%zu kv_heads=%zu head_dim=%zu intermediate=%zu vocab=%zu kv_stride=%zu\n",
         model->gemma, model->cfg.kv_int8, model->kv_ctx, context, model->cfg.layer_count,
         model->cfg.hidden_size, model->cfg.n_heads, model->cfg.kv_heads,
         model->cfg.head_dim, model->cfg.intermediate_size, model->cfg.vocab_size,
         model->kv_stride);
  if (!model->layers) {
    puts("CONTRACT fail model=layers-null");
  } else if (!context) {
    puts("CONTRACT fail model=context-zero");
  } else if (context > model->kv_ctx) {
    puts("CONTRACT fail model=context-exceeds-kv");
  } else if (!model->cfg.hidden_size || !model->cfg.n_heads || !model->cfg.kv_heads ||
             !model->cfg.head_dim || !model->cfg.intermediate_size ||
             !model->cfg.vocab_size || !model->cfg.layer_count || !model->kv_stride) {
    puts("CONTRACT fail model=zero-global-geometry");
  } else if (model->cfg.head_dim < 32 || model->cfg.head_dim > 512 ||
             model->cfg.head_dim % 32) {
    puts("CONTRACT fail model=head-dim");
  } else if (model->cfg.n_heads % model->cfg.kv_heads) {
    puts("CONTRACT fail model=global-gqa");
  } else if (!model->gemma) {
    puts("CONTRACT fail model=not-gemma");
  } else if (model->cfg.kv_int8) {
    puts("CONTRACT fail model=kv-int8");
  } else {
    for (size_t index = 0; index < model->cfg.layer_count; ++index) {
      const oc_layer *layer = &model->layers[index];
      const size_t hd = layer->hd;
      const size_t kvh = layer->n_kv;
      const size_t q_len = layer->wo.cols;
      if (layer->is_gdn) {
        printf("CONTRACT fail layer=%zu reason=is-gdn\n", index);
      } else if (layer->q_bias) {
        printf("CONTRACT fail layer=%zu reason=q-bias\n", index);
      } else if (layer->k_bias) {
        printf("CONTRACT fail layer=%zu reason=k-bias\n", index);
      } else if (layer->v_bias) {
        printf("CONTRACT fail layer=%zu reason=v-bias\n", index);
      } else if (!layer->kv_ck) {
        printf("CONTRACT fail layer=%zu reason=kv-k-null\n", index);
      } else if (!layer->kv_cv) {
        printf("CONTRACT fail layer=%zu reason=kv-v-null\n", index);
      } else if (!layer->kv_cap) {
        printf("CONTRACT fail layer=%zu reason=kv-cap-zero\n", index);
      } else if (layer->kv_cap > model->kv_ctx) {
        printf("CONTRACT fail layer=%zu reason=kv-cap-exceeds-model cap=%zu\n", index,
               layer->kv_cap);
      } else if (!hd || !kvh) {
        printf("CONTRACT fail layer=%zu reason=zero-layer-geometry hd=%zu kvh=%zu\n", index,
               hd, kvh);
      } else if (hd < 32 || hd > 512 || hd % 32) {
        printf("CONTRACT fail layer=%zu reason=head-dim hd=%zu\n", index, hd);
      } else if (!q_len) {
        printf("CONTRACT fail layer=%zu reason=q-len-zero\n", index);
      } else if (q_len % hd) {
        printf("CONTRACT fail layer=%zu reason=q-len-divisibility q_len=%zu hd=%zu\n", index,
               q_len, hd);
      } else if (q_len / hd % kvh) {
        printf("CONTRACT fail layer=%zu reason=layer-gqa q_heads=%zu kv_heads=%zu\n", index,
               q_len / hd, kvh);
      } else if (hd > SIZE_MAX / kvh || hd * kvh > model->kv_stride) {
        printf("CONTRACT fail layer=%zu reason=kv-stride hd=%zu kvh=%zu stride=%zu\n", index,
               hd, kvh, model->kv_stride);
      } else if (!batch_weight_present(&layer->wq)) {
        printf("CONTRACT fail layer=%zu reason=wq-missing\n", index);
      } else if (!batch_weight_present(&layer->wk)) {
        printf("CONTRACT fail layer=%zu reason=wk-missing\n", index);
      } else if (!batch_weight_present(&layer->wo)) {
        printf("CONTRACT fail layer=%zu reason=wo-missing\n", index);
      } else if (!batch_weight_present(&layer->gate)) {
        printf("CONTRACT fail layer=%zu reason=gate-missing\n", index);
      } else if (!batch_weight_present(&layer->up)) {
        printf("CONTRACT fail layer=%zu reason=up-missing\n", index);
      } else if (!batch_weight_present(&layer->down)) {
        printf("CONTRACT fail layer=%zu reason=down-missing\n", index);
      } else if (layer->wq.cols != model->cfg.hidden_size) {
        printf("CONTRACT fail layer=%zu reason=wq-cols rows=%zu cols=%zu\n", index,
               layer->wq.rows, layer->wq.cols);
      } else if (layer->wq.rows != q_len && layer->wq.rows != 2 * q_len) {
        printf("CONTRACT fail layer=%zu reason=wq-rows rows=%zu q_len=%zu\n", index,
               layer->wq.rows, q_len);
      } else if (layer->wk.cols != model->cfg.hidden_size || layer->wk.rows != hd * kvh) {
        printf("CONTRACT fail layer=%zu reason=wk-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->wk.rows, layer->wk.cols, hd * kvh, model->cfg.hidden_size);
      } else if (!layer->v_from_k && !batch_weight_present(&layer->wv)) {
        printf("CONTRACT fail layer=%zu reason=wv-missing\n", index);
      } else if (!layer->v_from_k &&
                 (layer->wv.cols != model->cfg.hidden_size || layer->wv.rows != hd * kvh)) {
        printf("CONTRACT fail layer=%zu reason=wv-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->wv.rows, layer->wv.cols, hd * kvh, model->cfg.hidden_size);
      } else if (layer->wo.rows != model->cfg.hidden_size || layer->wo.cols != q_len) {
        printf("CONTRACT fail layer=%zu reason=wo-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->wo.rows, layer->wo.cols, model->cfg.hidden_size, q_len);
      } else if (layer->gate.rows != model->cfg.intermediate_size ||
                 layer->gate.cols != model->cfg.hidden_size) {
        printf("CONTRACT fail layer=%zu reason=gate-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->gate.rows, layer->gate.cols, model->cfg.intermediate_size,
               model->cfg.hidden_size);
      } else if (layer->up.rows != model->cfg.intermediate_size ||
                 layer->up.cols != model->cfg.hidden_size) {
        printf("CONTRACT fail layer=%zu reason=up-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->up.rows, layer->up.cols, model->cfg.intermediate_size,
               model->cfg.hidden_size);
      } else if (layer->down.rows != model->cfg.hidden_size ||
                 layer->down.cols != model->cfg.intermediate_size) {
        printf("CONTRACT fail layer=%zu reason=down-shape rows=%zu cols=%zu expected=%zux%zu\n",
               index, layer->down.rows, layer->down.cols, model->cfg.hidden_size,
               model->cfg.intermediate_size);
      } else {
        continue;
      }
      fflush(stdout);
      return;
    }
    puts("CONTRACT ok");
  }
  fflush(stdout);
}

static int configure_and_build(oc_model *model, size_t slots, size_t context,
                               const char *stage) {
  oc_cuda_memory_report report = {0};
  const int preflight = oc_cuda_batch_memory_preflight(model, 0, slots, context, &report);
  printf("CUDA stage=%s preflight=%d required=%zu free=%zu weights=%zu scratch=%zu slot_kv=%zu\n",
         stage, preflight, report.required_bytes, report.free_bytes, report.fp16_weight_bytes,
         report.scratch_bytes, report.slot_kv_bytes);
  fflush(stdout);
  if (preflight != 0) {
    dump_batch_contract(model, context);
    return -1;
  }

  const int configured = oc_cuda_configure_batch(model, 0, slots, context);
  printf("CUDA stage=%s configure=%d\n", stage, configured);
  fflush(stdout);
  if (configured != 0) return -1;

  const int built = oc_cuda_build(model);
  printf("CUDA stage=%s build=%d\n", stage, built);
  fflush(stdout);
  return built;
}

static int compare_scalar_and_one_lane(oc_model *model, uint32_t token, size_t context) {
  const size_t vocab_size = model->cfg.vocab_size;
  if (!vocab_size || vocab_size > SIZE_MAX / sizeof(float)) return -1;
  float *logits = calloc(vocab_size, sizeof(*logits));
  if (!logits) return -1;

  oc_cuda_release(model);
  oc_reset_state(model);
  oc_forward(model, &token, 1, 0, logits);
  const uint32_t cpu_q4_id = greedy_id(logits, vocab_size);
  printf("CPU_Q4 id=%u\n", cpu_q4_id);
  fflush(stdout);

  oc_cuda_release(model);
  oc_reset_state(model);
  if (configure_and_build(model, 1, context, "gpu-scalar") != 0) {
    free(logits);
    return -1;
  }
  oc_forward(model, &token, 1, 0, logits);
  const uint32_t gpu_scalar_id = greedy_id(logits, vocab_size);
  printf("GPU_SCALAR id=%u\n", gpu_scalar_id);
  fflush(stdout);

  oc_cuda_release(model);
  oc_reset_state(model);
  if (configure_and_build(model, 1, context, "gpu-batch") != 0) {
    free(logits);
    return -1;
  }
  const oc_cuda_decode_lane lane = {
      .token = token, .position = 0, .slot = 0, .want_token = 1};
  uint32_t batch_id = UINT32_MAX;
  if (oc_cuda_decode_batch(model, &lane, 1, &batch_id) != 0) {
    fprintf(stderr, "FAIL one-lane batch call\n");
    free(logits);
    return -1;
  }
  if (batch_id >= vocab_size) {
    fprintf(stderr, "FAIL one-lane batch invalid id=%u\n", batch_id);
    free(logits);
    return -1;
  }
  printf("PARITY cpu_q4=%u gpu_scalar=%u gpu_batch=%u\n", cpu_q4_id, gpu_scalar_id,
         batch_id);
  fflush(stdout);
  if (batch_id != gpu_scalar_id) {
    fprintf(stderr, "FAIL one-lane parity gpu_scalar=%u gpu_batch=%u\n", gpu_scalar_id,
            batch_id);
    free(logits);
    return -1;
  }
  free(logits);
  return 0;
}

static int run_two_lane_smoke(oc_model *model, const uint32_t *prompt, size_t prompt_count,
                              size_t slots, size_t context) {
  if (prompt_count < 2) return -1;
  oc_cuda_release(model);
  oc_reset_state(model);
  if (configure_and_build(model, slots, context, "two-lane") != 0) return -1;

  uint32_t next_ids[2] = {UINT32_MAX, UINT32_MAX};
  const oc_cuda_decode_lane warmup = {
      .token = prompt[1], .position = 0, .slot = 1, .want_token = 1};
  if (oc_cuda_decode_batch(model, &warmup, 1, &next_ids[1]) != 0 ||
      next_ids[1] >= model->cfg.vocab_size)
    return -1;
  oc_cuda_decode_lane lanes[2] = {
      {.token = prompt[0], .position = 0, .slot = 0, .want_token = 1},
      {.token = next_ids[1], .position = 1, .slot = 1, .want_token = 1},
  };
  for (size_t step = 0; step < BATCH_STEPS; ++step) {
    if (oc_cuda_decode_batch(model, lanes, 2, next_ids) != 0 ||
        next_ids[0] >= model->cfg.vocab_size || next_ids[1] >= model->cfg.vocab_size)
      return -1;
    printf("B2 step=%zu ids=%u,%u slots=%u,%u positions=%u,%u\n", step,
           next_ids[0], next_ids[1], lanes[0].slot, lanes[1].slot,
           lanes[0].position, lanes[1].position);
    lanes[0].token = next_ids[0];
    lanes[1].token = next_ids[1];
    ++lanes[0].position;
    ++lanes[1].position;
  }
  return 0;
}

static int run_batch_benchmark(oc_model *model, const uint32_t *prompt, size_t prompt_count,
                               size_t slots, size_t context, size_t steps) {
  oc_cuda_decode_lane lanes[MAX_BENCH_SLOTS];
  uint32_t next_ids[MAX_BENCH_SLOTS];
  if (!prompt_count || slots > MAX_BENCH_SLOTS) return -1;

  oc_cuda_release(model);
  oc_reset_state(model);
  if (configure_and_build(model, slots, context, "bench") != 0) return -1;
  for (size_t lane = 0; lane < slots; ++lane) {
    lanes[lane] = (oc_cuda_decode_lane){
        .token = prompt[lane % prompt_count], .position = 0, .slot = lane, .want_token = 1};
  }

  for (size_t step = 0; step < BENCH_WARMUP_STEPS; ++step) {
    if (oc_cuda_decode_batch(model, lanes, slots, next_ids) != 0) {
      fprintf(stderr, "FAIL bench warmup batch call\n");
      return -1;
    }
    for (size_t lane = 0; lane < slots; ++lane) {
      if (next_ids[lane] >= model->cfg.vocab_size) {
        fprintf(stderr, "FAIL bench warmup invalid id=%u\n", next_ids[lane]);
        return -1;
      }
      lanes[lane].token = next_ids[lane];
      ++lanes[lane].position;
    }
  }

  struct timespec started = {0}, ended = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) {
    fprintf(stderr, "FAIL bench clock start\n");
    return -1;
  }
  for (size_t step = 0; step < steps; ++step) {
    if (oc_cuda_decode_batch(model, lanes, slots, next_ids) != 0) {
      fprintf(stderr, "FAIL bench timed batch call\n");
      return -1;
    }
    for (size_t lane = 0; lane < slots; ++lane) {
      if (next_ids[lane] >= model->cfg.vocab_size) {
        fprintf(stderr, "FAIL bench timed invalid id=%u\n", next_ids[lane]);
        return -1;
      }
      lanes[lane].token = next_ids[lane];
      ++lanes[lane].position;
    }
  }
  if (clock_gettime(CLOCK_MONOTONIC, &ended) != 0) {
    fprintf(stderr, "FAIL bench clock end\n");
    return -1;
  }
  const double seconds = (double)(ended.tv_sec - started.tv_sec) +
                         (double)(ended.tv_nsec - started.tv_nsec) * 1e-9;
  const size_t generated_tokens = slots * steps;
  if (seconds <= 0.0) {
    fprintf(stderr, "FAIL bench nonpositive duration\n");
    return -1;
  }
  printf("BENCH lanes=%zu steps=%zu generated_tokens=%zu seconds=%.6f aggregate_tps=%.6f\n",
         slots, steps, generated_tokens, seconds, (double)generated_tokens / seconds);
  fflush(stdout);
  return 0;
}

static oc_model *load_model(const smoke_options *options) {
  if (!options->bench_only)
    return oc_model_load(options->model_path, options->context, 0);

  const char *saved_no_gpu = getenv("OC_NO_GPU");
  char *saved_no_gpu_copy = saved_no_gpu ? strdup(saved_no_gpu) : NULL;
  if ((saved_no_gpu && !saved_no_gpu_copy) || setenv("OC_NO_GPU", "1", 1) != 0) {
    free(saved_no_gpu_copy);
    return NULL;
  }
  oc_model *model = oc_model_load(options->model_path, options->context, 0);
  if (saved_no_gpu_copy) {
    setenv("OC_NO_GPU", saved_no_gpu_copy, 1);
    free(saved_no_gpu_copy);
  } else {
    unsetenv("OC_NO_GPU");
  }
  return model;
}

int main(int argc, char **argv) {
  smoke_options options;
  if (parse_options(argc, argv, &options) != 0) {
    usage(argv[0]);
    return 2;
  }

  oc_model *model = load_model(&options);
  if (!model) {
    fprintf(stderr, "FAIL model load\n");
    return 1;
  }

  oc_tokenizer *tokenizer = oc_tokenizer_load(model->g);
  const char *prompt_text = "Explain batched decoding in one sentence.";
  size_t prompt_count = 0;
  uint32_t *prompt = tokenizer ? oc_tokenize(tokenizer, prompt_text, true, &prompt_count) : NULL;
  int status = 1;
  if (!prompt || (!options.bench_only && prompt_count < 2)) {
    fprintf(stderr, "FAIL tokenization\n");
  } else if (options.bench_only &&
             run_batch_benchmark(model, prompt, prompt_count, options.slots,
                                 options.context, options.bench_steps) != 0) {
    fprintf(stderr, "FAIL batch benchmark\n");
  } else if (options.bench_only) {
    puts("PASS");
    status = 0;
  } else if (compare_scalar_and_one_lane(model, prompt[0], options.context) != 0) {
    fprintf(stderr, "FAIL scalar-batch parity\n");
  } else if (run_two_lane_smoke(model, prompt, prompt_count, options.slots,
                                options.context) != 0) {
    fprintf(stderr, "FAIL two-lane decode\n");
  } else {
    puts("PASS");
    status = 0;
  }

  free(prompt);
  oc_tokenizer_free(tokenizer);
  oc_model_free(model);
  return status;
}
