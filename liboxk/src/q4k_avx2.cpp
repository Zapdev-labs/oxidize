#include "oxk_common.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define OXK_X86 1
#endif

namespace oxk {

#ifdef OXK_X86

static inline int32_t hsum_i32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b1110));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01));
    return _mm_cvtsi128_si32(sum128);
}

struct Q4Block {
    float d_w;
    float dmin_w;
    __m256i scale_v[8];
    int32_t mins[8];
    __m256i q4_lo[4];
    __m256i q4_hi[4];
};

static inline Q4Block decode_q4_block(const uint8_t *w_ptr) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    Q4Block b{};
    b.d_w = f16_le_to_f32(w_ptr[0], w_ptr[1]);
    b.dmin_w = f16_le_to_f32(w_ptr[2], w_ptr[3]);
    const uint8_t *scales = w_ptr + 4;
    const uint8_t *qs = w_ptr + 16;

    for (size_t gp = 0; gp < 4; ++gp) {
        const size_t g1 = gp * 2;
        const size_t g2 = g1 + 1;
        uint8_t s1, ms1, s2, ms2;
        get_scale_min_k4(g1, scales, s1, ms1);
        get_scale_min_k4(g2, scales, s2, ms2);
        b.scale_v[g1] = _mm256_set1_epi16(static_cast<int16_t>(s1));
        b.scale_v[g2] = _mm256_set1_epi16(static_cast<int16_t>(s2));
        b.mins[g1] = static_cast<int32_t>(ms1);
        b.mins[g2] = static_cast<int32_t>(ms2);
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qs + gp * 32));
        b.q4_lo[gp] = _mm256_and_si256(packed, mask);
        b.q4_hi[gp] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    }
    return b;
}

static inline void load_q8_block(const uint8_t *q8_ptr, float &d_q8, __m256i q8v[8],
                                 int32_t bs[8]) {
    d_q8 = read_f32_le(q8_ptr);
    const uint8_t *q8 = q8_ptr + 4;
    const uint8_t *bsums = q8_ptr + 4 + QK_K;
    for (size_t i = 0; i < 8; ++i) {
        q8v[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8 + i * 32));
    }
    for (size_t g = 0; g < 8; ++g) {
        bs[g] = static_cast<int32_t>(read_i16_le(bsums + g * 4)) +
                static_cast<int32_t>(read_i16_le(bsums + g * 4 + 2));
    }
}

static inline float row_dot_decoded(const Q4Block &b, float d_q8, const __m256i q8v[8],
                                    const int32_t bs[8]) {
    __m256i vec_pos = _mm256_setzero_si256();
    int32_t min_acc = 0;
    for (size_t gp = 0; gp < 4; ++gp) {
        const size_t g1 = gp * 2;
        const size_t g2 = g1 + 1;
        const __m256i p16_low = _mm256_maddubs_epi16(b.q4_lo[gp], q8v[g1]);
        const __m256i p16_high = _mm256_maddubs_epi16(b.q4_hi[gp], q8v[g2]);
        const __m256i p32_low = _mm256_madd_epi16(p16_low, b.scale_v[g1]);
        const __m256i p32_high = _mm256_madd_epi16(p16_high, b.scale_v[g2]);
        vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));
        min_acc += b.mins[g1] * bs[g1];
        min_acc += b.mins[g2] * bs[g2];
    }
    const int32_t pos_acc = hsum_i32(vec_pos);
    return b.d_w * d_q8 * static_cast<float>(pos_acc) -
           b.dmin_w * d_q8 * static_cast<float>(min_acc);
}

float q4k_q8k_row_dot_avx2(const uint8_t *row, size_t blocks_per_row, const uint8_t *q8k) {
    float acc = 0.0f;
    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint8_t *w_ptr = row + block_idx * BLOCK_Q4_K_SIZE;
        const Q4Block b = decode_q4_block(w_ptr);
        float d_q8;
        __m256i q8v[8];
        int32_t bs[8];
        load_q8_block(q8k + block_idx * BLOCK_Q8_K_BYTES, d_q8, q8v, bs);
        acc += row_dot_decoded(b, d_q8, q8v, bs);
    }
    return acc;
}

void q4k_q8k_row_dot_x4_avx2(const uint8_t *rows_base, size_t row_bytes,
                              size_t blocks_per_row, const uint8_t *q8k, float out[4]) {
    float acc[4] = {0, 0, 0, 0};
    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        float d_q8;
        __m256i q8v[8];
        int32_t bs[8];
        load_q8_block(q8k + block_idx * BLOCK_Q8_K_BYTES, d_q8, q8v, bs);
        for (size_t r = 0; r < 4; ++r) {
            const uint8_t *w_block = rows_base + r * row_bytes + block_idx * BLOCK_Q4_K_SIZE;
            const Q4Block b = decode_q4_block(w_block);
            acc[r] += row_dot_decoded(b, d_q8, q8v, bs);
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        out[i] = acc[i];
    }
}

void q4k_q8k_row_dot_x8_avx2(const uint8_t *rows_base, size_t row_bytes,
                              size_t blocks_per_row, const uint8_t *q8k, float out[8]) {
    float acc[8] = {};
    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        float d_q8;
        __m256i q8v[8];
        int32_t bs[8];
        load_q8_block(q8k + block_idx * BLOCK_Q8_K_BYTES, d_q8, q8v, bs);
        for (size_t r = 0; r < 8; ++r) {
            const uint8_t *w_block = rows_base + r * row_bytes + block_idx * BLOCK_Q4_K_SIZE;
            const Q4Block b = decode_q4_block(w_block);
            acc[r] += row_dot_decoded(b, d_q8, q8v, bs);
        }
    }
    for (size_t i = 0; i < 8; ++i) {
        out[i] = acc[i];
    }
}

#endif // OXK_X86

} // namespace oxk
