#include "oxk_common.hpp"

namespace oxk {

float q4k_q8k_row_dot_scalar(const uint8_t *row, size_t blocks_per_row,
                             const uint8_t *q8k) {
    float acc = 0.0f;
    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint8_t *w = row + block_idx * BLOCK_Q4_K_SIZE;
        const uint8_t *q8b = q8k + block_idx * BLOCK_Q8_K_BYTES;
        const float d_w = f16_le_to_f32(w[0], w[1]);
        const float dmin_w = f16_le_to_f32(w[2], w[3]);
        const float d_q8 = read_f32_le(q8b);
        const uint8_t *scales = w + 4;
        const uint8_t *qs = w + 16;
        const uint8_t *q8 = q8b + 4;
        const uint8_t *bsums = q8b + 4 + QK_K;

        int32_t pos = 0;
        int32_t min_acc = 0;
        for (size_t gp = 0; gp < 4; ++gp) {
            const size_t g1 = gp * 2;
            const size_t g2 = g1 + 1;
            uint8_t s1, ms1, s2, ms2;
            get_scale_min_k4(g1, scales, s1, ms1);
            get_scale_min_k4(g2, scales, s2, ms2);
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            for (size_t i = 0; i < 32; ++i) {
                const uint8_t byte = qs[gp * 32 + i];
                sum1 += static_cast<int32_t>(byte & 0x0fu) *
                        static_cast<int32_t>(static_cast<int8_t>(q8[g1 * 32 + i]));
                sum2 += static_cast<int32_t>(byte >> 4) *
                        static_cast<int32_t>(static_cast<int8_t>(q8[g2 * 32 + i]));
            }
            pos += static_cast<int32_t>(s1) * sum1 + static_cast<int32_t>(s2) * sum2;
            const int32_t bs1 = static_cast<int32_t>(read_i16_le(bsums + g1 * 4)) +
                                static_cast<int32_t>(read_i16_le(bsums + g1 * 4 + 2));
            const int32_t bs2 = static_cast<int32_t>(read_i16_le(bsums + g2 * 4)) +
                                static_cast<int32_t>(read_i16_le(bsums + g2 * 4 + 2));
            min_acc += static_cast<int32_t>(ms1) * bs1;
            min_acc += static_cast<int32_t>(ms2) * bs2;
        }
        acc += d_w * d_q8 * static_cast<float>(pos) -
               dmin_w * d_q8 * static_cast<float>(min_acc);
    }
    return acc;
}

} // namespace oxk
