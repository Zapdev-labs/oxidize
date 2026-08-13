#include "oxidize/rotor_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define OXIDIZE_ROTOR_AVX512 1
#endif

namespace oxidize {
namespace {

// Deterministic PRNG for rotor construction (no runtime dependency).
float lcg_unit(uint64_t& state) {
  state = state * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<float>(state >> 40) / static_cast<float>(1u << 24);
}

// Random 3D rotation matrix from a unit axis + angle (rotor sandwich product
// on vectors is exactly a rotation, so we store the equivalent 3x3 matrix).
void make_rotor(uint64_t& state, float* m) {
  float ax = lcg_unit(state) * 2.0f - 1.0f;
  float ay = lcg_unit(state) * 2.0f - 1.0f;
  float az = lcg_unit(state) * 2.0f - 1.0f;
  const float n = std::sqrt(ax * ax + ay * ay + az * az) + 1.0e-9f;
  ax /= n;
  ay /= n;
  az /= n;
  const float angle = lcg_unit(state) * 6.28318530718f;
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  const float t = 1.0f - c;
  m[0] = t * ax * ax + c;
  m[1] = t * ax * ay - s * az;
  m[2] = t * ax * az + s * ay;
  m[3] = t * ax * ay + s * az;
  m[4] = t * ay * ay + c;
  m[5] = t * ay * az - s * ax;
  m[6] = t * ax * az - s * ay;
  m[7] = t * ay * az + s * ax;
  m[8] = t * az * az + c;
}

}

struct RotorQuantCache::Page {
  size_t layer = 0;
  size_t head = 0;
  size_t tokens = 0;
  std::vector<float> key_scales;    // tokens * blocks_per_row
  std::vector<uint8_t> key_codes;   // packed nibbles
  std::vector<float> value_scales;
  std::vector<uint8_t> value_codes;
};

float RotorQuantStats::compression_ratio_vs_f32() const {
  const size_t bytes = key_bytes + value_bytes + metadata_bytes;
  return bytes == 0 ? 1.0f
                    : static_cast<float>(f32_baseline_bytes) /
                          static_cast<float>(bytes);
}

RotorQuantCache::RotorQuantCache(RotorQuantConfig config) : config_(config) {
  if (config_.head_dim == 0 || config_.block_size == 0) {
    throw std::invalid_argument("RotorQuantCache: zero dimension");
  }
  uint64_t state = config_.seed;
  const size_t groups = config_.head_dim / 3;
  rotors_.resize(groups * 9);
  for (size_t g = 0; g < groups; ++g) {
    make_rotor(state, rotors_.data() + g * 9);
  }
}

RotorQuantCache::~RotorQuantCache() = default;
RotorQuantCache::RotorQuantCache(RotorQuantCache&&) noexcept = default;
RotorQuantCache& RotorQuantCache::operator=(RotorQuantCache&&) noexcept = default;

std::vector<float> RotorQuantCache::rotate(const std::vector<float>& v) const {
  if (v.size() != config_.head_dim) {
    throw std::invalid_argument("RotorQuantCache: rotate shape mismatch");
  }
  std::vector<float> out(v.size());
  const size_t groups = config_.head_dim / 3;
  for (size_t g = 0; g < groups; ++g) {
    const float* m = rotors_.data() + g * 9;
    const float x = v[g * 3];
    const float y = v[g * 3 + 1];
    const float z = v[g * 3 + 2];
    out[g * 3] = m[0] * x + m[1] * y + m[2] * z;
    out[g * 3 + 1] = m[3] * x + m[4] * y + m[5] * z;
    out[g * 3 + 2] = m[6] * x + m[7] * y + m[8] * z;
  }
  // Trailing dims not divisible by 3 pass through unrotated.
  for (size_t i = groups * 3; i < config_.head_dim; ++i) {
    out[i] = v[i];
  }
  return out;
}

std::vector<float> RotorQuantCache::unrotate(const std::vector<float>& v) const {
  if (v.size() != config_.head_dim) {
    throw std::invalid_argument("RotorQuantCache: unrotate shape mismatch");
  }
  std::vector<float> out(v.size());
  const size_t groups = config_.head_dim / 3;
  for (size_t g = 0; g < groups; ++g) {
    const float* m = rotors_.data() + g * 9;  // transpose = inverse
    const float x = v[g * 3];
    const float y = v[g * 3 + 1];
    const float z = v[g * 3 + 2];
    out[g * 3] = m[0] * x + m[3] * y + m[6] * z;
    out[g * 3 + 1] = m[1] * x + m[4] * y + m[7] * z;
    out[g * 3 + 2] = m[2] * x + m[5] * y + m[8] * z;
  }
  for (size_t i = groups * 3; i < config_.head_dim; ++i) {
    out[i] = v[i];
  }
  return out;
}

void RotorQuantCache::store_page(size_t layer, size_t kv_head,
                                 const std::vector<float>& keys,
                                 const std::vector<float>& values,
                                 size_t tokens) {
  if (tokens == 0 || keys.size() != tokens * config_.head_dim ||
      values.size() != tokens * config_.head_dim) {
    throw std::invalid_argument("RotorQuantCache: page shape mismatch");
  }
  auto page = std::make_unique<Page>();
  page->layer = layer;
  page->head = kv_head;
  page->tokens = tokens;
  const size_t blocks_per_row =
      (config_.head_dim + config_.block_size - 1) / config_.block_size;
  const size_t code_bytes = (config_.head_dim + 1) / 2;

  auto quantize = [&](const std::vector<float>& src, std::vector<float>& scales,
                      std::vector<uint8_t>& codes) {
    scales.assign(tokens * blocks_per_row, 0.0f);
    codes.assign(tokens * code_bytes, 0);
    std::vector<float> row(config_.head_dim);
    std::vector<float> rotated(config_.head_dim);
    const size_t groups = config_.head_dim / 3;
    for (size_t t = 0; t < tokens; ++t) {
      const float* in = src.data() + t * config_.head_dim;
      for (size_t g = 0; g < groups; ++g) {
        const float* m = rotors_.data() + g * 9;
        const float x = in[g * 3];
        const float y = in[g * 3 + 1];
        const float z = in[g * 3 + 2];
        rotated[g * 3] = m[0] * x + m[1] * y + m[2] * z;
        rotated[g * 3 + 1] = m[3] * x + m[4] * y + m[5] * z;
        rotated[g * 3 + 2] = m[6] * x + m[7] * y + m[8] * z;
      }
      for (size_t i = groups * 3; i < config_.head_dim; ++i) {
        rotated[i] = in[i];
      }
      for (size_t b = 0; b < blocks_per_row; ++b) {
        const size_t start = b * config_.block_size;
        const size_t end = std::min(start + config_.block_size, config_.head_dim);
        float max_abs = 0.0f;
        for (size_t i = start; i < end; ++i) {
          max_abs = std::max(max_abs, std::fabs(rotated[i]));
        }
        const float scale = max_abs > 0.0f ? max_abs / 7.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[t * blocks_per_row + b] = scale;
        for (size_t i = start; i < end; ++i) {
          const float v = rotated[i] * inv;
          const int q = std::clamp(
              static_cast<int>(v + (v >= 0.0f ? 0.5f : -0.5f)), -7, 7) + 7;
          uint8_t& byte = codes[t * code_bytes + i / 2];
          if ((i & 1u) == 0) {
            byte = static_cast<uint8_t>((byte & 0xF0u) | q);
          } else {
            byte = static_cast<uint8_t>((byte & 0x0Fu) | (q << 4));
          }
        }
      }
    }
  };
  quantize(keys, page->key_scales, page->key_codes);
  quantize(values, page->value_scales, page->value_codes);
  pages_.push_back(std::move(page));
}

std::vector<float> RotorQuantCache::logits(size_t layer, size_t kv_head,
                                           const std::vector<float>& query) const {
  // Rotation is orthogonal, so q . k == (Rq) . (Rk): rotate the query once and
  // the per-token work is a plain int4 dot product.
  const std::vector<float> rq = rotate(query);
  const size_t blocks_per_row =
      (config_.head_dim + config_.block_size - 1) / config_.block_size;
  const size_t code_bytes = (config_.head_dim + 1) / 2;
  std::vector<float> result;
#if defined(OXIDIZE_ROTOR_AVX512)
  if (config_.block_size == 32 && config_.head_dim % 32 == 0) {
    // Low nibble of byte j holds dim 2j, high nibble dim 2j+1: deinterleave
    // the rotated query once so each 16-byte block decodes straight into FMAs.
    const size_t half = config_.head_dim / 2;
    std::vector<float> rqe(half), rqo(half);
    for (size_t i = 0; i < half; ++i) {
      rqe[i] = rq[2 * i];
      rqo[i] = rq[2 * i + 1];
    }
    const __m128i lo4 = _mm_set1_epi8(0x0F);
    const __m512 seven = _mm512_set1_ps(7.0f);
    for (const auto& page : pages_) {
      if (page->layer != layer || page->head != kv_head) {
        continue;
      }
      for (size_t t = 0; t < page->tokens; ++t) {
        const uint8_t* codes = page->key_codes.data() + t * code_bytes;
        const float* scales = page->key_scales.data() + t * blocks_per_row;
        float logit = 0.0f;
        for (size_t b = 0; b < blocks_per_row; ++b) {
          const __m128i bytes = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(codes + b * 16));
          const __m512 lo = _mm512_sub_ps(
              _mm512_cvtepi32_ps(
                  _mm512_cvtepu8_epi32(_mm_and_si128(bytes, lo4))),
              seven);
          const __m512 hi = _mm512_sub_ps(
              _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                  _mm_and_si128(_mm_srli_epi16(bytes, 4), lo4))),
              seven);
          const __m512 sum = _mm512_fmadd_ps(
              hi, _mm512_loadu_ps(rqo.data() + b * 16),
              _mm512_mul_ps(lo, _mm512_loadu_ps(rqe.data() + b * 16)));
          logit += scales[b] * _mm512_reduce_add_ps(sum);
        }
        result.push_back(logit);
      }
    }
    return result;
  }
#endif
  for (const auto& page : pages_) {
    if (page->layer != layer || page->head != kv_head) {
      continue;
    }
    for (size_t t = 0; t < page->tokens; ++t) {
      const uint8_t* codes = page->key_codes.data() + t * code_bytes;
      const float* scales = page->key_scales.data() + t * blocks_per_row;
      float logit = 0.0f;
      for (size_t b = 0; b < blocks_per_row; ++b) {
        const size_t start = b * config_.block_size;
        const size_t end = std::min(start + config_.block_size, config_.head_dim);
        float sum = 0.0f;
        for (size_t i = start; i < end; ++i) {
          const uint8_t byte = codes[i / 2];
          const int q = (i & 1u) == 0 ? (byte & 0x0F) : (byte >> 4);
          sum += rq[i] * static_cast<float>(q - 7);
        }
        logit += sum * scales[b];
      }
      result.push_back(logit);
    }
  }
  return result;
}

std::vector<float> RotorQuantCache::attention(size_t layer, size_t kv_head,
                                              const std::vector<float>& query) const {
  std::vector<float> scores = logits(layer, kv_head, query);
  if (scores.empty()) {
    return std::vector<float>(config_.head_dim, 0.0f);
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
  float max_score = -std::numeric_limits<float>::infinity();
  for (float& s : scores) {
    s *= scale;
    max_score = std::max(max_score, s);
  }
  float sum = 0.0f;
  for (float& s : scores) {
    s = std::exp(s - max_score);
    sum += s;
  }
  for (float& s : scores) {
    s /= sum;
  }
  const size_t blocks_per_row =
      (config_.head_dim + config_.block_size - 1) / config_.block_size;
  const size_t code_bytes = (config_.head_dim + 1) / 2;
  std::vector<float> acc(config_.head_dim, 0.0f);
  size_t score_index = 0;
#if defined(OXIDIZE_ROTOR_AVX512)
  if (config_.block_size == 32 && config_.head_dim % 32 == 0) {
    // Accumulate raw codes per nibble lane; fold the -7 offset in once per
    // block at the end via the running weight sum.
    const size_t half = config_.head_dim / 2;
    std::vector<float> acc_e(half, 0.0f), acc_o(half, 0.0f);
    std::vector<float> wsum(blocks_per_row, 0.0f);
    const __m128i lo4 = _mm_set1_epi8(0x0F);
    for (const auto& page : pages_) {
      if (page->layer != layer || page->head != kv_head) {
        continue;
      }
      for (size_t t = 0; t < page->tokens; ++t) {
        const float weight = scores[score_index++];
        if (weight < 1.0e-12f) {
          continue;
        }
        const uint8_t* codes = page->value_codes.data() + t * code_bytes;
        const float* scales = page->value_scales.data() + t * blocks_per_row;
        for (size_t b = 0; b < blocks_per_row; ++b) {
          const float ws = weight * scales[b];
          wsum[b] += ws;
          const __m512 vws = _mm512_set1_ps(ws);
          const __m128i bytes = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(codes + b * 16));
          const __m512 lo = _mm512_cvtepi32_ps(
              _mm512_cvtepu8_epi32(_mm_and_si128(bytes, lo4)));
          const __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
              _mm_and_si128(_mm_srli_epi16(bytes, 4), lo4)));
          float* pe = acc_e.data() + b * 16;
          float* po = acc_o.data() + b * 16;
          _mm512_storeu_ps(pe, _mm512_fmadd_ps(vws, lo, _mm512_loadu_ps(pe)));
          _mm512_storeu_ps(po, _mm512_fmadd_ps(vws, hi, _mm512_loadu_ps(po)));
        }
      }
    }
    for (size_t b = 0; b < blocks_per_row; ++b) {
      for (size_t l = 0; l < 16; ++l) {
        acc[b * 32 + 2 * l] = acc_e[b * 16 + l] - 7.0f * wsum[b];
        acc[b * 32 + 2 * l + 1] = acc_o[b * 16 + l] - 7.0f * wsum[b];
      }
    }
    return unrotate(acc);
  }
#endif
  for (const auto& page : pages_) {
    if (page->layer != layer || page->head != kv_head) {
      continue;
    }
    for (size_t t = 0; t < page->tokens; ++t) {
      const float weight = scores[score_index++];
      const uint8_t* codes = page->value_codes.data() + t * code_bytes;
      const float* scales = page->value_scales.data() + t * blocks_per_row;
      for (size_t b = 0; b < blocks_per_row; ++b) {
        const size_t start = b * config_.block_size;
        const size_t end = std::min(start + config_.block_size, config_.head_dim);
        const float ws = weight * scales[b];
        for (size_t i = start; i < end; ++i) {
          const uint8_t byte = codes[i / 2];
          const int q = (i & 1u) == 0 ? (byte & 0x0F) : (byte >> 4);
          acc[i] += ws * static_cast<float>(q - 7);
        }
      }
    }
  }
  // Accumulate in rotated space; one inverse rotation at the end.
  return unrotate(acc);
}

size_t RotorQuantCache::page_count() const { return pages_.size(); }

bool RotorQuantCache::page_view(size_t index, PageView* view) const {
  if (index >= pages_.size() || view == nullptr) {
    return false;
  }
  const Page& page = *pages_[index];
  view->layer = page.layer;
  view->kv_head = page.head;
  view->tokens = page.tokens;
  view->key_codes = page.key_codes.data();
  view->key_scales = page.key_scales.data();
  view->value_codes = page.value_codes.data();
  view->value_scales = page.value_scales.data();
  return true;
}

RotorQuantStats RotorQuantCache::stats() const {
  RotorQuantStats stats;
  for (const auto& page : pages_) {
    stats.token_count += page->tokens;
    stats.key_bytes += page->key_codes.size();
    stats.value_bytes += page->value_codes.size();
    stats.metadata_bytes += (page->key_scales.size() + page->value_scales.size()) *
                            sizeof(float);
  }
  stats.f32_baseline_bytes = stats.token_count * config_.head_dim * 2 * sizeof(float);
  const float coords = static_cast<float>(2 * stats.token_count * config_.head_dim);
  if (coords > 0.0f) {
    stats.total_bits_per_coord =
        static_cast<float>(
            (stats.key_bytes + stats.value_bytes + stats.metadata_bytes) * 8) /
        coords;
  }
  return stats;
}

}
