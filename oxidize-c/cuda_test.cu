#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "oc.h"
}

namespace {

struct TinyGemma {
  oc_model model = {};
  std::vector<float *> allocations;

  float *take(size_t n, float value = 0.0f) {
    float *data = static_cast<float *>(std::calloc(n, sizeof(float)));
    if (!data) std::abort();
    for (size_t i = 0; i < n; ++i) data[i] = value;
    allocations.push_back(data);
    return data;
  }

  static void dense(oc_weight *w, float *data, size_t rows, size_t cols) {
    w->quantized = false;
    w->quant = OC_F32;
    w->f32 = data;
    w->rows = rows;
    w->cols = cols;
  }

  TinyGemma() {
    constexpr size_t vocab = 8, hidden = 64, inter = 96, head_dim = 32;
    model.gemma = true;
    model.emb_scale = 1.0f;
    model.cfg.vocab_size = vocab;
    model.cfg.hidden_size = hidden;
    model.cfg.intermediate_size = inter;
    model.cfg.layer_count = 1;
    model.cfg.n_heads = 2;
    model.cfg.kv_heads = 1;
    model.cfg.head_dim = head_dim;
    model.cfg.rope_dim = head_dim;
    model.cfg.rms_eps = 1e-5f;
    model.cfg.rope_theta = 10000.0f;
    model.kv_ctx = 4;
    model.kv_stride = head_dim;
    model.layers = static_cast<oc_layer *>(std::calloc(1, sizeof(oc_layer)));
    oc_layer &l = model.layers[0];
    l.kv_slot = 0;
    l.hd = head_dim;
    l.n_kv = 1;
    l.n_rot = head_dim;
    l.theta = 10000.0f;
    l.attn_scale = 0.0f;
    l.v_from_k = true;
    l.v_rms = true;
    l.kv_cap = 3;
    l.attn_norm = take(hidden, 1.0f);
    l.ffn_norm = take(hidden, 1.0f);
    l.attn_post_norm = take(hidden, 1.0f);
    l.ffn_post_norm = take(hidden, 1.0f);
    l.kv_ck = take(l.kv_cap * head_dim, 0.0f);
    l.kv_cv = take(l.kv_cap * head_dim, 0.0f);
    model.final_norm = take(hidden, 1.0f);

    float *emb = take(vocab * hidden, 0.0f);
    for (size_t token = 0; token < vocab; ++token)
      for (size_t col = 0; col < hidden; ++col)
        emb[token * hidden + col] = 0.03f * static_cast<float>((token + 1) * ((col % 7) + 1));
    dense(&model.tok_emb, emb, vocab, hidden);
    float *head = take(vocab * hidden, 0.0f);
    for (size_t token = 0; token < vocab; ++token)
      for (size_t col = 0; col < hidden; ++col)
        head[token * hidden + col] = 0.011f * static_cast<float>(token + 2) *
                                    static_cast<float>(static_cast<int>(col % 11) - 5);
    dense(&model.lm_head, head, vocab, hidden);

    auto matrix = [&](size_t rows, size_t cols, float scale) {
      float *w = take(rows * cols, 0.0f);
      for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < cols; ++col)
          w[row * cols + col] = scale *
              static_cast<float>(static_cast<int>((row * 13 + col * 7) % 19) - 9);
      return w;
    };
    dense(&l.wq, matrix(2 * hidden, hidden, 0.007f), 2 * hidden, hidden);
    dense(&l.wk, matrix(head_dim, hidden, 0.009f), head_dim, hidden);
    dense(&l.wv, matrix(head_dim, hidden, 0.004f), head_dim, hidden);
    dense(&l.wo, matrix(hidden, hidden, 0.006f), hidden, hidden);
    dense(&l.gate, matrix(inter, hidden, 0.005f), inter, hidden);
    dense(&l.up, matrix(inter, hidden, 0.008f), inter, hidden);
    dense(&l.down, matrix(hidden, inter, 0.004f), hidden, inter);
  }

  ~TinyGemma() {
    oc_cuda_release(&model);
    std::free(model.layers);
    for (float *data : allocations) std::free(data);
  }
};

uint32_t scalar_greedy(TinyGemma *fixture, uint32_t token, uint32_t position) {
  float logits[8] = {};
  const float *embed = fixture->model.tok_emb.f32 + static_cast<size_t>(token) * 64;
  oc_cuda_forward(&fixture->model, embed, position, 1, logits, nullptr);
  uint32_t best = 0;
  for (uint32_t id = 1; id < 8; ++id)
    if (logits[id] > logits[best]) best = id;
  return best;
}

bool configure(TinyGemma *fixture) {
  return oc_cuda_configure_batch(&fixture->model, 0, 4, 4) == 0 &&
         oc_cuda_build(&fixture->model) == 0;
}

int test_batch_configuration_rejects_unsupported_attention_contracts() {
  TinyGemma fixture;
  oc_layer &layer = fixture.model.layers[0];
  float *bias = fixture.take(1, 0.25f);

  layer.q_bias = bias;
  layer.q_bias_n = 1;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted q bias\n");
    return 1;
  }
  layer.q_bias = nullptr;
  layer.q_bias_n = 0;
  layer.k_bias = bias;
  layer.k_bias_n = 1;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted k bias\n");
    return 1;
  }
  layer.k_bias = nullptr;
  layer.k_bias_n = 0;
  layer.v_bias = bias;
  layer.v_bias_n = 1;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted v bias\n");
    return 1;
  }
  layer.v_bias = nullptr;
  layer.v_bias_n = 0;
  layer.hd = 16;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted head_dim < 32\n");
    return 1;
  }
  layer.hd = 32;
  layer.hd = 33;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted non-warp head_dim\n");
    return 1;
  }
  layer.hd = 544;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted head_dim > 512\n");
    return 1;
  }
  layer.hd = 32;
  layer.n_kv = 0;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted zero KV heads\n");
    return 1;
  }
  layer.n_kv = 1;
  fixture.model.cfg.kv_heads = 0;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted zero global KV heads\n");
    return 1;
  }
  fixture.model.cfg.kv_heads = 1;
  layer.n_kv = 3;
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, 4) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted non-divisible GQA\n");
    return 1;
  }
  return 0;
}

int test_batch_preflight_reports_tied_head_once_and_rejects_oversized_context() {
  TinyGemma fixture;
  fixture.model.tied = true;
  fixture.model.lm_head = fixture.model.tok_emb;
  oc_cuda_memory_report report = {};
  if (oc_cuda_batch_memory_preflight(&fixture.model, 0, 4, 4, &report) != 0 ||
      report.fp16_weight_bytes == 0 || report.scratch_bytes == 0 ||
      report.slot_kv_bytes == 0 || report.required_bytes > report.free_bytes) {
    std::fprintf(stderr, "cuda_test: batch preflight omitted required memory\n");
    return 1;
  }
  if (oc_cuda_configure_batch(&fixture.model, 0, 4, fixture.model.kv_ctx + 1) != -1) {
    std::fprintf(stderr, "cuda_test: batch configure accepted context beyond global cache\n");
    return 1;
  }
  return 0;
}

int test_batch_one_matches_scalar_fixture() {
  TinyGemma scalar;
  TinyGemma batch;
  if (!configure(&scalar) || !configure(&batch)) return 1;
  const uint32_t expected = scalar_greedy(&scalar, 3, 0);
  const oc_cuda_decode_lane lane = {3, 0, 0, 1};
  uint32_t actual = UINT32_MAX;
  if (oc_cuda_decode_batch(&batch.model, &lane, 1, &actual) != 0 || actual != expected) {
    std::fprintf(stderr, "cuda_test: B=1 did not match scalar fixture (%u vs %u)\n", actual, expected);
    return 1;
  }
  return 0;
}

int test_fixture_full_logits_match_host_output_projection() {
  TinyGemma fixture;
  if (!configure(&fixture)) return 1;
  float logits[8] = {};
  float normed[64] = {};
  const uint32_t token = 3;
  const float *embed = fixture.model.tok_emb.f32 + static_cast<size_t>(token) * 64;
  oc_cuda_forward(&fixture.model, embed, 0, 1, logits, normed);
  for (size_t row = 0; row < 8; ++row) {
    float reference = 0.0f;
    for (size_t col = 0; col < 64; ++col)
      reference += fixture.model.lm_head.f32[row * 64 + col] * normed[col];
    if (!std::isfinite(logits[row]) || std::fabs(logits[row] - reference) > 2e-3f) {
      std::fprintf(stderr, "cuda_test: full logit %zu diverged from host reference\n", row);
      return 1;
    }
  }
  return 0;
}

int test_batch_two_lanes_are_isolated_and_padding_is_inert() {
  TinyGemma batched;
  TinyGemma slot_zero;
  TinyGemma slot_one;
  TinyGemma clean_padding;
  if (!configure(&batched) || !configure(&slot_zero) || !configure(&slot_one) ||
      !configure(&clean_padding)) return 1;
  const oc_cuda_decode_lane lanes[3] = {{2, 0, 0, 1}, {5, 1, 1, 1}, {7, 0, 2, 1}};
  uint32_t ids[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  if (oc_cuda_decode_batch(&batched.model, lanes, 3, ids) != 0) return 1;
  const uint32_t expected_zero = scalar_greedy(&slot_zero, 2, 0);
  const uint32_t expected_one = scalar_greedy(&slot_one, 5, 1);
  if (ids[0] != expected_zero || ids[1] != expected_one) {
    std::fprintf(stderr, "cuda_test: B=2 lanes contaminated each other\n");
    return 1;
  }
  const oc_cuda_decode_lane padded_slot = {6, 1, 3, 1};
  uint32_t padded_actual = UINT32_MAX, padded_expected = UINT32_MAX;
  if (oc_cuda_decode_batch(&batched.model, &padded_slot, 1, &padded_actual) != 0 ||
      oc_cuda_decode_batch(&clean_padding.model, &padded_slot, 1, &padded_expected) != 0 ||
      padded_actual != padded_expected) {
    std::fprintf(stderr, "cuda_test: padded bucket lane wrote a KV slot\n");
    return 1;
  }
  return 0;
}

int test_batch_slot_histories_wrap_without_cross_slot_contamination() {
  TinyGemma batched;
  TinyGemma scalar_zero;
  TinyGemma scalar_one;
  if (!configure(&batched) || !configure(&scalar_zero) || !configure(&scalar_one)) return 1;
  for (uint32_t position = 0; position < 4; ++position) {
    const oc_cuda_decode_lane lanes[2] = {
        {static_cast<uint32_t>(1 + position), position, 0, 1},
        {static_cast<uint32_t>(5 - position), position, 1, 1}};
    uint32_t ids[2] = {UINT32_MAX, UINT32_MAX};
    const uint32_t expected_zero = scalar_greedy(&scalar_zero, lanes[0].token, position);
    const uint32_t expected_one = scalar_greedy(&scalar_one, lanes[1].token, position);
    if (oc_cuda_decode_batch(&batched.model, lanes, 2, ids) != 0 ||
        ids[0] != expected_zero || ids[1] != expected_one) {
      std::fprintf(stderr, "cuda_test: multi-step GQA slot history mismatch at position %u\n", position);
      return 1;
    }
  }
  return 0;
}

int test_invalid_lanes_are_rejected() {
  TinyGemma fixture;
  if (!configure(&fixture)) return 1;
  uint32_t id = UINT32_MAX;
  const oc_cuda_decode_lane invalid_slot = {1, 0, 4, 1};
  const oc_cuda_decode_lane invalid_token = {8, 0, 0, 1};
  const oc_cuda_decode_lane invalid_position = {1, 4, 0, 1};
  if (oc_cuda_decode_batch(&fixture.model, nullptr, 1, &id) != -1 ||
      oc_cuda_decode_batch(&fixture.model, &invalid_slot, 1, &id) != -1 ||
      oc_cuda_decode_batch(&fixture.model, &invalid_token, 1, &id) != -1 ||
      oc_cuda_decode_batch(&fixture.model, &invalid_position, 1, &id) != -1 ||
      oc_cuda_decode_batch(&fixture.model, &invalid_slot, 0, &id) != -1) {
    std::fprintf(stderr, "cuda_test: invalid batch lane was accepted\n");
    return 1;
  }
  fixture.model.cfg.kv_int8 = 1;
  const oc_cuda_decode_lane valid = {1, 0, 0, 1};
  if (oc_cuda_decode_batch(&fixture.model, &valid, 1, &id) != -1) {
    std::fprintf(stderr, "cuda_test: Gemma batch decode accepted int8 KV\n");
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("cuda_test: SKIP no CUDA device\n");
    return 0;
  }
  if (test_batch_one_matches_scalar_fixture() != 0 ||
      test_fixture_full_logits_match_host_output_projection() != 0 ||
      test_batch_two_lanes_are_isolated_and_padding_is_inert() != 0 ||
      test_batch_slot_histories_wrap_without_cross_slot_contamination() != 0 ||
      test_batch_configuration_rejects_unsupported_attention_contracts() != 0 ||
      test_batch_preflight_reports_tied_head_once_and_rejects_oversized_context() != 0 ||
      test_invalid_lanes_are_rejected() != 0)
    return 1;
  std::printf("cuda_test: PASS real Gemma batch decode fixture\n");
  return 0;
}
