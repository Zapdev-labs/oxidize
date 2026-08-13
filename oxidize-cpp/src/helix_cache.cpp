#include "oxidize/helix_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define OXIDIZE_HELIX_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace oxidize {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kPhiStep = 2.0f * kPi / 16.0f;

struct PageKey {
  size_t layer = 0;
  size_t head = 0;
  size_t page = 0;
};

struct ColdKeyTile {
  std::vector<float> mu_phi;
  std::vector<float> log_rho_min;
  std::vector<float> log_rho_step;
  std::vector<uint8_t> active_mask;
  // One byte per (token, pair): radius code in the high nibble, phase code in
  // the low nibble. Inactive coords are 0.
  std::vector<uint8_t> codes;
  // Decode caches derived from the fields above; rebuilt at store time and
  // excluded from compression stats. rho for code k is a * r^k, computed with
  // masked multiplies from r^1, r^2, r^4, r^8 (no gather needed).
  std::vector<float> rho_lut;  // pairs * 16: exp(log_rho_min + step * code)
  std::vector<float> rho_a;    // exp(log_rho_min)
  std::vector<float> rho_r1;   // exp(step), and its 2nd/4th/8th powers
  std::vector<float> rho_r2;
  std::vector<float> rho_r4;
  std::vector<float> rho_r8;
  std::vector<float> cos_mu;
  std::vector<float> sin_mu;
};

struct ColdValueTile {
  std::vector<float> scales;
  std::vector<uint8_t> codes;  // 3-bit codes + 8 padding bytes for wide loads
};

constexpr size_t kValueCodePadding = 8;

struct PromotionState {
  uint32_t uncertainty_counter = 0;
  float recent_max_overlap = 0.0f;
  uint64_t access_count = 0;
};

size_t packed_bits_bytes(size_t bit_count) { return (bit_count + 7) / 8; }

void set_bit(std::vector<uint8_t>& data, size_t index, bool value) {
  const size_t byte = index / 8;
  const uint8_t mask = static_cast<uint8_t>(1u << (index % 8));
  if (value) {
    data[byte] = static_cast<uint8_t>(data[byte] | mask);
  } else {
    data[byte] = static_cast<uint8_t>(data[byte] & ~mask);
  }
}

bool get_bit(const std::vector<uint8_t>& data, size_t index) {
  return (data[index / 8] & static_cast<uint8_t>(1u << (index % 8))) != 0;
}

float wrap_angle(float value) {
  while (value <= -kPi) value += 2.0f * kPi;
  while (value > kPi) value -= 2.0f * kPi;
  return value;
}

float rope_frequency(size_t pair, size_t head_dim, float theta) {
  return std::pow(theta, -2.0f * static_cast<float>(pair) /
                             static_cast<float>(head_dim));
}

void hadamard8(const float* src, float* dst) {
  const float a0 = src[0] + src[1];
  const float a1 = src[0] - src[1];
  const float a2 = src[2] + src[3];
  const float a3 = src[2] - src[3];
  const float a4 = src[4] + src[5];
  const float a5 = src[4] - src[5];
  const float a6 = src[6] + src[7];
  const float a7 = src[6] - src[7];
  const float b0 = a0 + a2;
  const float b1 = a1 + a3;
  const float b2 = a0 - a2;
  const float b3 = a1 - a3;
  const float b4 = a4 + a6;
  const float b5 = a5 + a7;
  const float b6 = a4 - a6;
  const float b7 = a5 - a7;
  dst[0] = b0 + b4;
  dst[1] = b1 + b5;
  dst[2] = b2 + b6;
  dst[3] = b3 + b7;
  dst[4] = b0 - b4;
  dst[5] = b1 - b5;
  dst[6] = b2 - b6;
  dst[7] = b3 - b7;
}

bool same_key(const PageKey& a, size_t layer, size_t head, size_t page) {
  return a.layer == layer && a.head == head && a.page == page;
}

// cos/sin of (code - 8) * pi/8 for the 16 phase codes.
struct PhaseLut {
  alignas(64) float c[16];
  alignas(64) float s[16];
  PhaseLut() {
    for (int k = 0; k < 16; ++k) {
      c[k] = std::cos((k - 8) * kPhiStep);
      s[k] = std::sin((k - 8) * kPhiStep);
    }
  }
};
const PhaseLut kPhase;

#if defined(OXIDIZE_HELIX_AVX512)

// Natural log, x > 0. ~1e-7 relative error; quantization steps are far
// coarser, so this cannot move any code.
inline __m512 v_log_ps(__m512 x) {
  const __m512i xi = _mm512_castps_si512(x);
  __m512 e = _mm512_cvtepi32_ps(
      _mm512_sub_epi32(_mm512_srli_epi32(xi, 23), _mm512_set1_epi32(127)));
  __m512 m = _mm512_castsi512_ps(_mm512_or_epi32(
      _mm512_and_epi32(xi, _mm512_set1_epi32(0x007FFFFF)),
      _mm512_set1_epi32(0x3F800000)));
  const __mmask16 gt = _mm512_cmp_ps_mask(m, _mm512_set1_ps(1.41421356f),
                                          _CMP_GT_OQ);
  m = _mm512_mask_mul_ps(m, gt, m, _mm512_set1_ps(0.5f));
  e = _mm512_mask_add_ps(e, gt, e, _mm512_set1_ps(1.0f));
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 z = _mm512_div_ps(_mm512_sub_ps(m, one), _mm512_add_ps(m, one));
  const __m512 z2 = _mm512_mul_ps(z, z);
  __m512 p = _mm512_set1_ps(1.0f / 9.0f);
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 7.0f));
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 5.0f));
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 3.0f));
  p = _mm512_fmadd_ps(p, z2, one);
  p = _mm512_mul_ps(p, _mm512_add_ps(z, z));
  return _mm512_fmadd_ps(e, _mm512_set1_ps(0.69314718056f), p);
}

// atan2 with ~1e-5 rad error — two orders of magnitude below the pi/8 phase
// quantization step.
inline __m512 v_atan2_ps(__m512 y, __m512 x) {
  const __m512 ax = _mm512_abs_ps(x);
  const __m512 ay = _mm512_abs_ps(y);
  const __m512 hi = _mm512_max_ps(ax, ay);
  const __m512 lo = _mm512_min_ps(ax, ay);
  const __mmask16 nz = _mm512_cmp_ps_mask(hi, _mm512_setzero_ps(), _CMP_GT_OQ);
  const __m512 a = _mm512_maskz_div_ps(nz, lo, hi);
  const __m512 a2 = _mm512_mul_ps(a, a);
  __m512 r = _mm512_set1_ps(-0.0117212f);
  r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(0.05265332f));
  r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(-0.11643287f));
  r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(0.19354346f));
  r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(-0.33262347f));
  r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(0.99997726f));
  r = _mm512_mul_ps(r, a);
  const __mmask16 swap = _mm512_cmp_ps_mask(ay, ax, _CMP_GT_OQ);
  r = _mm512_mask_sub_ps(r, swap, _mm512_set1_ps(kPi * 0.5f), r);
  const __mmask16 xneg =
      _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
  r = _mm512_mask_sub_ps(r, xneg, _mm512_set1_ps(kPi), r);
  const __mmask16 yneg =
      _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ);
  return _mm512_mask_sub_ps(r, yneg, _mm512_setzero_ps(), r);
}

// exp with ~2e-7 relative error for softmax weights.
inline __m512 v_exp_ps(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
  const __m512 n =
      _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)),
                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 f = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
  f = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), f);
  __m512 p = _mm512_set1_ps(1.9875691500e-4f);
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3981999507e-3f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(8.3334519073e-3f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.1665795894e-2f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.6666665459e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.0000001201e-1f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));  // 1 + f + f^2*P(f)
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
  const __m512i pow2 = _mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(pow2));
}

#endif

// Softmax over raw scores scaled by `scale`, in place.
void softmax_inplace(std::vector<float>& scores, float scale) {
  const size_t n = scores.size();
#if defined(OXIDIZE_HELIX_AVX512)
  if (n >= 64) {
    float* d = scores.data();
    const size_t vn = n & ~size_t(15);
    const __m512 vscale = _mm512_set1_ps(scale);
    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < vn; i += 16) {
      const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(d + i), vscale);
      _mm512_storeu_ps(d + i, v);
      vmax = _mm512_max_ps(vmax, v);
    }
    float max_score = _mm512_reduce_max_ps(vmax);
    for (size_t i = vn; i < n; ++i) {
      d[i] *= scale;
      max_score = std::max(max_score, d[i]);
    }
    const __m512 vm = _mm512_set1_ps(max_score);
    __m512 vsum = _mm512_setzero_ps();
    for (size_t i = 0; i < vn; i += 16) {
      const __m512 e = v_exp_ps(_mm512_sub_ps(_mm512_loadu_ps(d + i), vm));
      _mm512_storeu_ps(d + i, e);
      vsum = _mm512_add_ps(vsum, e);
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (size_t i = vn; i < n; ++i) {
      d[i] = std::exp(d[i] - max_score);
      sum += d[i];
    }
    const __m512 vinv = _mm512_set1_ps(1.0f / sum);
    for (size_t i = 0; i < vn; i += 16) {
      _mm512_storeu_ps(d + i, _mm512_mul_ps(_mm512_loadu_ps(d + i), vinv));
    }
    for (size_t i = vn; i < n; ++i) {
      d[i] /= sum;
    }
    return;
  }
#endif
  float max_score = -std::numeric_limits<float>::infinity();
  for (float& score : scores) {
    score *= scale;
    max_score = std::max(max_score, score);
  }
  float sum = 0.0f;
  for (float& score : scores) {
    score = std::exp(score - max_score);
    sum += score;
  }
  for (float& score : scores) {
    score /= sum;
  }
}

}

struct HelixCache::Impl {
  PageKey key;
  HelixPageTier tier = HelixPageTier::Cold;
  std::vector<size_t> positions;
  ColdKeyTile cold_key;
  ColdValueTile cold_value;
  std::vector<float> hot_keys;
  std::vector<float> hot_values;
  PromotionState promotion;
};

float HelixCacheStats::compression_ratio_vs_f32() const {
  const size_t bytes = key_bytes + value_bytes + hot_bytes + metadata_bytes;
  return bytes == 0 ? 1.0f
                    : static_cast<float>(f32_baseline_bytes) /
                          static_cast<float>(bytes);
}

HelixCache::HelixCache(HelixCacheConfig config) : config_(config) {
  if (config_.page_size == 0) {
    throw std::invalid_argument("HelixCache: page_size must be non-zero");
  }
  if (config_.head_dim == 0 || config_.head_dim % 2 != 0 || config_.head_dim % 8 != 0) {
    throw std::invalid_argument("HelixCache: head_dim must be a non-zero multiple of 8");
  }
  if (config_.key_radius_bits != 4 || config_.key_phase_bits != 4 ||
      config_.value_bits != 3) {
    throw std::invalid_argument("HelixCache: only 4-bit keys and 3-bit values are supported");
  }
}

HelixCache::~HelixCache() = default;
HelixCache::HelixCache(HelixCache&&) noexcept = default;
HelixCache& HelixCache::operator=(HelixCache&&) noexcept = default;

void HelixCache::store_hot_page(size_t layer, size_t kv_head, size_t page_id,
                                const std::vector<float>& pre_rope_keys,
                                const std::vector<float>& values,
                                const std::vector<size_t>& positions) {
  const size_t token_count = positions.size();
  if (token_count == 0 || token_count > config_.page_size) {
    throw std::invalid_argument("HelixCache: invalid hot token count");
  }
  if (pre_rope_keys.size() != token_count * config_.head_dim ||
      values.size() != token_count * config_.head_dim) {
    throw std::invalid_argument("HelixCache: hot page shape mismatch");
  }
  auto page = std::make_unique<Impl>();
  page->key = PageKey{layer, kv_head, page_id};
  page->tier = HelixPageTier::Hot;
  page->positions = positions;
  page->hot_keys = pre_rope_keys;
  page->hot_values = values;
  pages_.push_back(std::move(page));
}

void HelixCache::store_cold_page(size_t layer, size_t kv_head, size_t page_id,
                                 const std::vector<float>& pre_rope_keys,
                                 const std::vector<float>& values,
                                 const std::vector<size_t>& positions) {
  const size_t tokens = positions.size();
  if (tokens == 0 || tokens > config_.page_size) {
    throw std::invalid_argument("HelixCache: invalid cold token count");
  }
  if (pre_rope_keys.size() != tokens * config_.head_dim ||
      values.size() != tokens * config_.head_dim) {
    throw std::invalid_argument("HelixCache: cold page shape mismatch");
  }
  auto page = std::make_unique<Impl>();
  page->key = PageKey{layer, kv_head, page_id};
  page->tier = HelixPageTier::Cold;
  page->positions = positions;
  const size_t pairs = config_.head_dim / 2;
  const size_t codes = tokens * pairs;
  page->cold_key.mu_phi.assign(pairs, 0.0f);
  page->cold_key.log_rho_min.assign(pairs, 0.0f);
  page->cold_key.log_rho_step.assign(pairs, 0.0f);
  page->cold_key.active_mask.assign(packed_bits_bytes(codes), 0);
  page->cold_key.codes.assign(codes, 0);
  page->cold_key.rho_lut.assign(pairs * 16, 0.0f);
  page->cold_key.rho_a.assign(pairs, 0.0f);
  page->cold_key.rho_r1.assign(pairs, 1.0f);
  page->cold_key.rho_r2.assign(pairs, 1.0f);
  page->cold_key.rho_r4.assign(pairs, 1.0f);
  page->cold_key.rho_r8.assign(pairs, 1.0f);
  page->cold_key.cos_mu.assign(pairs, 1.0f);
  page->cold_key.sin_mu.assign(pairs, 0.0f);

  // Pass 1: polar-decompose every pair once into scratch and accumulate
  // per-pair stats. sin/cos of phi are y/rho and x/rho — no trig needed.
  std::vector<float> sc_log_rho(codes, 0.0f);
  std::vector<float> sc_phi(codes, 0.0f);
  std::vector<float> sin_sum(pairs, 0.0f);
  std::vector<float> cos_sum(pairs, 0.0f);
  std::vector<float> min_log(pairs, std::numeric_limits<float>::infinity());
  std::vector<float> max_log(pairs, -std::numeric_limits<float>::infinity());
#if defined(OXIDIZE_HELIX_AVX512)
  if (pairs % 16 == 0) {
    const __m512i even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16,
                                               18, 20, 22, 24, 26, 28, 30);
    const __m512i odd_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17,
                                              19, 21, 23, 25, 27, 29, 31);
    const __m512 thr = _mm512_set1_ps(config_.inactive_threshold);
    const __m512 eps = _mm512_set1_ps(1.0e-12f);
    for (size_t t = 0; t < tokens; ++t) {
      const float* row = pre_rope_keys.data() + t * config_.head_dim;
      for (size_t k = 0; k < pairs; k += 16) {
        const __m512 lo = _mm512_loadu_ps(row + 2 * k);
        const __m512 hi = _mm512_loadu_ps(row + 2 * k + 16);
        const __m512 x = _mm512_permutex2var_ps(lo, even_idx, hi);
        const __m512 y = _mm512_permutex2var_ps(lo, odd_idx, hi);
        const __m512 rho =
            _mm512_sqrt_ps(_mm512_fmadd_ps(y, y, _mm512_mul_ps(x, x)));
        const __mmask16 m = _mm512_cmp_ps_mask(rho, thr, _CMP_GE_OQ);
        const size_t idx = t * pairs + k;
        std::memcpy(page->cold_key.active_mask.data() + idx / 8, &m,
                    sizeof(uint16_t));
        const __m512 inv = _mm512_maskz_div_ps(
            _mm512_cmp_ps_mask(rho, _mm512_setzero_ps(), _CMP_GT_OQ) & m,
            _mm512_set1_ps(1.0f), rho);
        _mm512_storeu_ps(sin_sum.data() + k,
                         _mm512_fmadd_ps(y, inv,
                                         _mm512_loadu_ps(sin_sum.data() + k)));
        _mm512_storeu_ps(cos_sum.data() + k,
                         _mm512_fmadd_ps(x, inv,
                                         _mm512_loadu_ps(cos_sum.data() + k)));
        const __m512 lr = v_log_ps(_mm512_add_ps(rho, eps));
        _mm512_storeu_ps(sc_log_rho.data() + idx, lr);
        _mm512_storeu_ps(sc_phi.data() + idx, v_atan2_ps(y, x));
        __m512 vmin = _mm512_loadu_ps(min_log.data() + k);
        __m512 vmax = _mm512_loadu_ps(max_log.data() + k);
        vmin = _mm512_mask_min_ps(vmin, m, vmin, lr);
        vmax = _mm512_mask_max_ps(vmax, m, vmax, lr);
        _mm512_storeu_ps(min_log.data() + k, vmin);
        _mm512_storeu_ps(max_log.data() + k, vmax);
      }
    }
  } else
#endif
  {
    for (size_t t = 0; t < tokens; ++t) {
      const float* row = pre_rope_keys.data() + t * config_.head_dim;
      for (size_t p = 0; p < pairs; ++p) {
        const float x = row[2 * p];
        const float y = row[2 * p + 1];
        const float rho = std::sqrt(x * x + y * y);
        if (rho < config_.inactive_threshold) {
          continue;
        }
        const size_t idx = t * pairs + p;
        set_bit(page->cold_key.active_mask, idx, true);
        sc_phi[idx] = std::atan2(y, x);
        sc_log_rho[idx] = std::log(rho + 1.0e-12f);
        if (rho > 0.0f) {
          sin_sum[p] += y / rho;
          cos_sum[p] += x / rho;
        }
        min_log[p] = std::min(min_log[p], sc_log_rho[idx]);
        max_log[p] = std::max(max_log[p], sc_log_rho[idx]);
      }
    }
  }

  std::vector<float> inv_step(pairs, 0.0f);
  for (size_t p = 0; p < pairs; ++p) {
    if (!std::isfinite(min_log[p])) {
      min_log[p] = 0.0f;  // pair fully inactive; codes stay 0 via mask
      continue;
    }
    const float mu = std::atan2(sin_sum[p], cos_sum[p]);
    const float step =
        (max_log[p] > min_log[p]) ? (max_log[p] - min_log[p]) / 15.0f : 0.0f;
    page->cold_key.mu_phi[p] = mu;
    page->cold_key.log_rho_min[p] = min_log[p];
    page->cold_key.log_rho_step[p] = step;
    page->cold_key.cos_mu[p] = std::cos(mu);
    page->cold_key.sin_mu[p] = std::sin(mu);
    const float a = std::exp(min_log[p]);
    const float r1 = std::exp(step);
    page->cold_key.rho_a[p] = a;
    page->cold_key.rho_r1[p] = r1;
    page->cold_key.rho_r2[p] = r1 * r1;
    page->cold_key.rho_r4[p] = page->cold_key.rho_r2[p] * page->cold_key.rho_r2[p];
    page->cold_key.rho_r8[p] = page->cold_key.rho_r4[p] * page->cold_key.rho_r4[p];
    for (int k = 0; k < 16; ++k) {
      page->cold_key.rho_lut[p * 16 + k] = std::exp(min_log[p] + step * k);
    }
    inv_step[p] = step == 0.0f ? 0.0f : 1.0f / step;
  }

  // Pass 2: encode codes from scratch. No transcendentals left.
#if defined(OXIDIZE_HELIX_AVX512)
  if (pairs % 16 == 0) {
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 inv_phi = _mm512_set1_ps(1.0f / kPhiStep);
    const __m512 inv_2pi = _mm512_set1_ps(1.0f / (2.0f * kPi));
    const __m512 two_pi = _mm512_set1_ps(2.0f * kPi);
    const __m512i vzero = _mm512_setzero_si512();
    const __m512i v15 = _mm512_set1_epi32(15);
    for (size_t t = 0; t < tokens; ++t) {
      const size_t base = t * pairs;
      for (size_t k = 0; k < pairs; k += 16) {
        const size_t idx = base + k;
        uint16_t bits;
        std::memcpy(&bits, page->cold_key.active_mask.data() + idx / 8,
                    sizeof(bits));
        const __mmask16 m = bits;
        const __m512 lr = _mm512_loadu_ps(sc_log_rho.data() + idx);
        const __m512 vmin = _mm512_loadu_ps(min_log.data() + k);
        const __m512 vinv = _mm512_loadu_ps(inv_step.data() + k);
        __m512i rc = _mm512_cvttps_epi32(_mm512_add_ps(
            _mm512_mul_ps(_mm512_sub_ps(lr, vmin), vinv), half));
        rc = _mm512_min_epi32(_mm512_max_epi32(rc, vzero), v15);
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(sc_phi.data() + idx),
                                 _mm512_loadu_ps(page->cold_key.mu_phi.data() + k));
        const __m512 n = _mm512_roundscale_ps(
            _mm512_mul_ps(d, inv_2pi),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        d = _mm512_fnmadd_ps(n, two_pi, d);
        __m512i pc = _mm512_add_epi32(
            _mm512_cvtps_epi32(_mm512_mul_ps(d, inv_phi)),
            _mm512_set1_epi32(8));
        pc = _mm512_min_epi32(_mm512_max_epi32(pc, vzero), v15);
        const __m512i byte = _mm512_maskz_mov_epi32(
            m, _mm512_or_epi32(_mm512_slli_epi32(rc, 4), pc));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(page->cold_key.codes.data() + idx),
            _mm512_cvtepi32_epi8(byte));
      }
    }
  } else
#endif
  {
    for (size_t t = 0; t < tokens; ++t) {
      for (size_t p = 0; p < pairs; ++p) {
        const size_t idx = t * pairs + p;
        if (!get_bit(page->cold_key.active_mask, idx)) {
          continue;
        }
        const int rho_code = std::clamp(
            static_cast<int>((sc_log_rho[idx] - min_log[p]) * inv_step[p] +
                             0.5f),
            0, 15);
        const float delta = wrap_angle(sc_phi[idx] - page->cold_key.mu_phi[p]);
        const int phi_code = std::clamp(
            static_cast<int>(std::lround(delta / kPhiStep)) + 8, 0, 15);
        page->cold_key.codes[idx] =
            static_cast<uint8_t>((rho_code << 4) | phi_code);
      }
    }
  }

  const size_t groups = config_.head_dim / 8;
  page->cold_value.scales.assign(groups, 0.0f);
  page->cold_value.codes.assign(
      packed_bits_bytes(tokens * config_.head_dim * 3) + kValueCodePadding, 0);
  std::vector<float> transformed(tokens * 8);
  for (size_t g = 0; g < groups; ++g) {
    float max_abs = 0.0f;
    for (size_t t = 0; t < tokens; ++t) {
      hadamard8(values.data() + t * config_.head_dim + g * 8,
                transformed.data() + t * 8);
      for (size_t i = 0; i < 8; ++i) {
        max_abs = std::max(max_abs, std::fabs(transformed[t * 8 + i]));
      }
    }
    const float scale = max_abs > 0.0f ? max_abs / 3.0f : 0.0f;
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    page->cold_value.scales[g] = scale;
    for (size_t t = 0; t < tokens; ++t) {
      // 8 codes x 3 bits = 24 bits, always byte-aligned (head_dim % 8 == 0).
      uint32_t word = 0;
      for (size_t i = 0; i < 8; ++i) {
        const float v = transformed[t * 8 + i] * inv_scale;
        const int q = std::clamp(
            static_cast<int>(v + (v >= 0.0f ? 0.5f : -0.5f)), -3, 3);
        word |= static_cast<uint32_t>(q + 3) << (3 * i);
      }
      const size_t byte = (t * config_.head_dim + g * 8) * 3 / 8;
      page->cold_value.codes[byte] = static_cast<uint8_t>(word);
      page->cold_value.codes[byte + 1] = static_cast<uint8_t>(word >> 8);
      page->cold_value.codes[byte + 2] = static_cast<uint8_t>(word >> 16);
    }
  }
  pages_.push_back(std::move(page));
}

std::vector<float> HelixCache::logits(size_t layer, size_t kv_head,
                                      const std::vector<float>& query_pre_rope,
                                      size_t query_position,
                                      float rope_theta) const {
  if (query_pre_rope.size() != config_.head_dim) {
    throw std::invalid_argument("HelixCache: query shape mismatch");
  }
  const size_t pairs = config_.head_dim / 2;
  // Per-pair RoPE step rotators: advancing one token multiplies the rotated
  // query by (cos f, sin f), so the inner loop needs no trig at all.
  std::vector<float> freq(pairs), step_c(pairs), step_s(pairs);
  for (size_t p = 0; p < pairs; ++p) {
    freq[p] = rope_frequency(p, config_.head_dim, rope_theta);
    step_c[p] = std::cos(freq[p]);
    step_s[p] = std::sin(freq[p]);
  }
  std::vector<float> qx(pairs), qy(pairs);
  auto seed_query = [&](const Impl& page, float rel, bool cold) {
    for (size_t p = 0; p < pairs; ++p) {
      const float theta =
          freq[p] * rel - (cold ? page.cold_key.mu_phi[p] : 0.0f);
      const float c = std::cos(theta);
      const float s = std::sin(theta);
      const float x = query_pre_rope[2 * p];
      const float y = query_pre_rope[2 * p + 1];
      qx[p] = x * c - y * s;
      qy[p] = x * s + y * c;
    }
  };
  std::vector<float> result;
  result.reserve(pages_.size() * config_.page_size);
  for (const auto& page_ptr : pages_) {
    const Impl& page = *page_ptr;
    if (page.key.layer != layer || page.key.head != kv_head) {
      continue;
    }
    const size_t tokens = page.positions.size();
    const bool cold = page.tier == HelixPageTier::Cold;
    // Rotate the query to relative position pos[0] (and by -mu for cold pages
    // so the key side reduces to its 16-entry phase table).
    seed_query(page, static_cast<float>(page.positions[0]) -
                         static_cast<float>(query_position),
               cold);

#if defined(OXIDIZE_HELIX_AVX512)
    if (cold && pairs % 16 == 0 && pairs <= 128) {
      const size_t chunks = pairs / 16;
      __m512 vqx[8], vqy[8], vcf[8], vsf[8];
      for (size_t k = 0; k < chunks; ++k) {
        vqx[k] = _mm512_loadu_ps(qx.data() + k * 16);
        vqy[k] = _mm512_loadu_ps(qy.data() + k * 16);
        vcf[k] = _mm512_loadu_ps(step_c.data() + k * 16);
        vsf[k] = _mm512_loadu_ps(step_s.data() + k * 16);
      }
      const __m512 phase_c = _mm512_load_ps(kPhase.c);
      const __m512 phase_s = _mm512_load_ps(kPhase.s);
      const __m512i lo4 = _mm512_set1_epi32(15);
      const __m512i b4 = _mm512_set1_epi32(0x10);
      const __m512i b5 = _mm512_set1_epi32(0x20);
      const __m512i b6 = _mm512_set1_epi32(0x40);
      const __m512i b7 = _mm512_set1_epi32(0x80);
      const uint8_t* am = page.cold_key.active_mask.data();
      const uint8_t* cb = page.cold_key.codes.data();
      const ColdKeyTile& ck = page.cold_key;
      for (size_t t = 0; t < tokens; ++t) {
        if (t > 0) {
          if (page.positions[t] == page.positions[t - 1] + 1) {
            for (size_t k = 0; k < chunks; ++k) {
              const __m512 nx = _mm512_fmsub_ps(
                  vqx[k], vcf[k], _mm512_mul_ps(vqy[k], vsf[k]));
              vqy[k] = _mm512_fmadd_ps(vqx[k], vsf[k],
                                       _mm512_mul_ps(vqy[k], vcf[k]));
              vqx[k] = nx;
            }
          } else {
            seed_query(page, static_cast<float>(page.positions[t]) -
                                 static_cast<float>(query_position),
                       true);
            for (size_t k = 0; k < chunks; ++k) {
              vqx[k] = _mm512_loadu_ps(qx.data() + k * 16);
              vqy[k] = _mm512_loadu_ps(qy.data() + k * 16);
            }
          }
        }
        __m512 acc = _mm512_setzero_ps();
        const size_t base = t * pairs;
        for (size_t k = 0; k < chunks; ++k) {
          uint16_t bits;
          std::memcpy(&bits, am + (base + k * 16) / 8, sizeof(bits));
          const __mmask16 m = bits;
          const __m512i ci = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(cb + base + k * 16)));
          const __m512i phi_idx = _mm512_and_epi32(ci, lo4);
          // rho = a * r^code via 4 masked multiplies on the code bits.
          __m512 rho =
              _mm512_maskz_loadu_ps(m, ck.rho_a.data() + k * 16);
          rho = _mm512_mask_mul_ps(rho, _mm512_test_epi32_mask(ci, b4), rho,
                                   _mm512_loadu_ps(ck.rho_r1.data() + k * 16));
          rho = _mm512_mask_mul_ps(rho, _mm512_test_epi32_mask(ci, b5), rho,
                                   _mm512_loadu_ps(ck.rho_r2.data() + k * 16));
          rho = _mm512_mask_mul_ps(rho, _mm512_test_epi32_mask(ci, b6), rho,
                                   _mm512_loadu_ps(ck.rho_r4.data() + k * 16));
          rho = _mm512_mask_mul_ps(rho, _mm512_test_epi32_mask(ci, b7), rho,
                                   _mm512_loadu_ps(ck.rho_r8.data() + k * 16));
          const __m512 cc = _mm512_permutexvar_ps(phi_idx, phase_c);
          const __m512 ss = _mm512_permutexvar_ps(phi_idx, phase_s);
          const __m512 proj =
              _mm512_fmadd_ps(vqy[k], ss, _mm512_mul_ps(vqx[k], cc));
          acc = _mm512_fmadd_ps(rho, proj, acc);
        }
        result.push_back(_mm512_reduce_add_ps(acc));
      }
      continue;
    }
#endif
    for (size_t t = 0; t < tokens; ++t) {
      if (t > 0) {
        if (page.positions[t] == page.positions[t - 1] + 1) {
          for (size_t p = 0; p < pairs; ++p) {
            const float nx = qx[p] * step_c[p] - qy[p] * step_s[p];
            qy[p] = qx[p] * step_s[p] + qy[p] * step_c[p];
            qx[p] = nx;
          }
        } else {
          seed_query(page, static_cast<float>(page.positions[t]) -
                               static_cast<float>(query_position),
                     cold);
        }
      }
      float logit = 0.0f;
      if (cold) {
        const size_t base = t * pairs;
        for (size_t p = 0; p < pairs; ++p) {
          const size_t idx = base + p;
          if (!get_bit(page.cold_key.active_mask, idx)) {
            continue;
          }
          const uint8_t code = page.cold_key.codes[idx];
          const float rho = page.cold_key.rho_lut[p * 16 + (code >> 4)];
          logit += rho * (qx[p] * kPhase.c[code & 15] +
                          qy[p] * kPhase.s[code & 15]);
        }
      } else {
        const float* row = page.hot_keys.data() + t * config_.head_dim;
        for (size_t p = 0; p < pairs; ++p) {
          logit += qx[p] * row[2 * p] + qy[p] * row[2 * p + 1];
        }
      }
      result.push_back(logit);
    }
  }
  return result;
}

std::vector<float> HelixCache::attention(size_t layer, size_t kv_head,
                                         const std::vector<float>& query_pre_rope,
                                         size_t query_position,
                                         float rope_theta) {
  std::vector<float> scores = logits(layer, kv_head, query_pre_rope, query_position,
                                     rope_theta);
  if (scores.empty()) {
    return std::vector<float>(config_.head_dim, 0.0f);
  }
  softmax_inplace(scores,
                  1.0f / std::sqrt(static_cast<float>(config_.head_dim)));
  std::vector<float> out(config_.head_dim, 0.0f);
  const size_t groups = config_.head_dim / 8;
  std::vector<float> acc(config_.head_dim);
  size_t score_index = 0;
  for (auto& page_ptr : pages_) {
    Impl& page = *page_ptr;
    if (page.key.layer != layer || page.key.head != kv_head) {
      continue;
    }
    page.promotion.access_count += 1;
    const size_t tokens = page.positions.size();
    if (page.tier == HelixPageTier::Hot) {
      for (size_t t = 0; t < tokens; ++t) {
        const float weight = scores[score_index++];
        for (size_t i = 0; i < config_.head_dim; ++i) {
          out[i] += weight * page.hot_values[t * config_.head_dim + i];
        }
      }
      continue;
    }
    // The Hadamard transform is linear, so accumulate weighted codes per page
    // and apply the inverse transform + scale once at the end of the page.
    std::fill(acc.begin(), acc.end(), 0.0f);
    float sum_w = 0.0f;
    const uint8_t* codes = page.cold_value.codes.data();
    for (size_t t = 0; t < tokens; ++t) {
      const float weight = scores[score_index++];
      if (weight < 1.0e-12f) {
        continue;
      }
      const float w8 = weight * 0.125f;
      sum_w += w8;
      const size_t row_byte = t * config_.head_dim * 3 / 8;
#if defined(OXIDIZE_HELIX_AVX512) && defined(__BMI2__)
      const __m512 vw = _mm512_set1_ps(w8);
      for (size_t g = 0; g + 1 < groups; g += 2) {
        // pdep expands 8x3-bit codes into 8 bytes; two groups fill a zmm.
        uint64_t two_words;
        std::memcpy(&two_words, codes + row_byte + g * 3, sizeof(two_words));
        const uint64_t lo8 =
            _pdep_u64(two_words & 0xFFFFFFu, 0x0707070707070707ull);
        const uint64_t hi8 =
            _pdep_u64((two_words >> 24) & 0xFFFFFFu, 0x0707070707070707ull);
        const __m512 f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
            _mm_set_epi64x(static_cast<long long>(hi8),
                           static_cast<long long>(lo8))));
        float* accg = acc.data() + g * 8;
        _mm512_storeu_ps(accg,
                         _mm512_fmadd_ps(vw, f, _mm512_loadu_ps(accg)));
      }
      if (groups % 2 != 0) {
        const size_t g = groups - 1;
        uint32_t word;
        std::memcpy(&word, codes + row_byte + g * 3, sizeof(word));
        for (size_t i = 0; i < 8; ++i) {
          acc[g * 8 + i] += w8 * static_cast<float>((word >> (3 * i)) & 7u);
        }
      }
#elif defined(__AVX2__)
      const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
      const __m256i mask7 = _mm256_set1_epi32(7);
      const __m256 vw = _mm256_set1_ps(w8);
      for (size_t g = 0; g < groups; ++g) {
        uint32_t word;
        std::memcpy(&word, codes + row_byte + g * 3, sizeof(word));
        const __m256i q = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(word)), shifts),
            mask7);
        const __m256 f = _mm256_cvtepi32_ps(q);
        float* accg = acc.data() + g * 8;
        _mm256_storeu_ps(accg,
                         _mm256_fmadd_ps(vw, f, _mm256_loadu_ps(accg)));
      }
#else
      for (size_t g = 0; g < groups; ++g) {
        uint32_t word;
        std::memcpy(&word, codes + row_byte + g * 3, sizeof(word));
        for (size_t i = 0; i < 8; ++i) {
          acc[g * 8 + i] += w8 * static_cast<float>((word >> (3 * i)) & 7u);
        }
      }
#endif
    }
    float centered[8];
    float inverse[8];
    for (size_t g = 0; g < groups; ++g) {
      const float s = page.cold_value.scales[g];
      for (size_t i = 0; i < 8; ++i) {
        centered[i] = acc[g * 8 + i] - 3.0f * sum_w;
      }
      hadamard8(centered, inverse);
      for (size_t i = 0; i < 8; ++i) {
        out[g * 8 + i] += s * inverse[i];
      }
    }
  }
  return out;
}

size_t HelixCache::page_count() const { return pages_.size(); }

bool HelixCache::cold_page_view(size_t index, HelixColdPageView* view) const {
  if (index >= pages_.size() || view == nullptr ||
      pages_[index]->tier != HelixPageTier::Cold) {
    return false;
  }
  const Impl& page = *pages_[index];
  view->layer = page.key.layer;
  view->kv_head = page.key.head;
  view->tokens = page.positions.size();
  view->positions = page.positions.data();
  view->key_codes = page.cold_key.codes.data();
  view->active_mask = page.cold_key.active_mask.data();
  view->mu_phi = page.cold_key.mu_phi.data();
  view->log_rho_min = page.cold_key.log_rho_min.data();
  view->log_rho_step = page.cold_key.log_rho_step.data();
  view->value_codes = page.cold_value.codes.data();
  view->value_scales = page.cold_value.scales.data();
  return true;
}

void HelixCache::bump_uncertainty(size_t layer, size_t kv_head, size_t page_id,
                                  float interval_overlap) {
  for (auto& page : pages_) {
    if (same_key(page->key, layer, kv_head, page_id)) {
      page->promotion.recent_max_overlap =
          std::max(page->promotion.recent_max_overlap, interval_overlap);
      if (interval_overlap >= config_.promotion_epsilon) {
        page->promotion.uncertainty_counter += 1;
      }
      return;
    }
  }
  throw std::invalid_argument("HelixCache: page not found");
}

bool HelixCache::should_promote(size_t layer, size_t kv_head, size_t page_id) const {
  for (const auto& page : pages_) {
    if (same_key(page->key, layer, kv_head, page_id)) {
      return page->tier == HelixPageTier::Cold &&
             page->promotion.uncertainty_counter >= config_.promotion_budget;
    }
  }
  throw std::invalid_argument("HelixCache: page not found");
}

HelixCacheStats HelixCache::stats() const {
  HelixCacheStats stats;
  for (const auto& page_ptr : pages_) {
    const Impl& page = *page_ptr;
    const size_t tokens = page.positions.size();
    stats.token_count += tokens;
    stats.page_metadata_bytes += page.positions.size() * sizeof(size_t);
    if (page.tier == HelixPageTier::Hot) {
      stats.hot_pages += 1;
      stats.hot_bytes += page.hot_keys.size() * sizeof(float);
      stats.hot_bytes += page.hot_values.size() * sizeof(float);
      continue;
    }
    stats.cold_pages += 1;
    stats.key_metadata_bytes += page.cold_key.mu_phi.size() * sizeof(float);
    stats.key_metadata_bytes += page.cold_key.log_rho_min.size() * sizeof(float);
    stats.key_metadata_bytes += page.cold_key.log_rho_step.size() * sizeof(float);
    stats.value_metadata_bytes += page.cold_value.scales.size() * sizeof(float);
    stats.key_bytes += page.cold_key.active_mask.size();
    stats.key_bytes += page.cold_key.codes.size();
    stats.value_bytes += page.cold_value.codes.size() - kValueCodePadding;
  }
  stats.metadata_bytes =
      stats.key_metadata_bytes + stats.value_metadata_bytes + stats.page_metadata_bytes;
  // Keys and values each contribute head_dim coordinates per token.
  const float key_coords = static_cast<float>(stats.token_count * config_.head_dim);
  const float total_coords = 2.0f * key_coords;
  stats.f32_baseline_bytes = stats.token_count * config_.head_dim * 2 * sizeof(float);
  if (key_coords > 0.0f) {
    stats.key_bits_per_coord =
        static_cast<float>((stats.key_bytes + stats.key_metadata_bytes) * 8) /
        key_coords;
    stats.value_bits_per_coord =
        static_cast<float>((stats.value_bytes + stats.value_metadata_bytes) * 8) /
        key_coords;
    stats.total_bits_per_coord =
        static_cast<float>((stats.key_bytes + stats.value_bytes + stats.hot_bytes +
                            stats.metadata_bytes) * 8) /
        total_coords;
  }
  return stats;
}

}
