#include "oxk_common.hpp"

#include <algorithm>
#include <cmath>

namespace oxk {

void quantize_q8k_into(const float *vector, size_t n_blocks, uint8_t *out) {
    for (size_t b = 0; b < n_blocks; ++b) {
        const float *block_in = vector + b * QK_K;
        uint8_t *block_out = out + b * BLOCK_Q8_K_BYTES;

        float amax = 0.0f;
        float max_val = 0.0f;
        for (size_t i = 0; i < QK_K; ++i) {
            const float av = std::fabs(block_in[i]);
            if (av > amax) {
                amax = av;
                max_val = block_in[i];
            }
        }
        if (amax == 0.0f) {
            write_f32_le(block_out, 0.0f);
            std::memset(block_out + 4, 0, BLOCK_Q8_K_BYTES - 4);
            continue;
        }
        const float iscale = -128.0f / max_val;
        const float d = 1.0f / iscale;
        write_f32_le(block_out, d);
        for (size_t i = 0; i < QK_K; ++i) {
            const int32_t q = static_cast<int32_t>(std::round(iscale * block_in[i]));
            const int32_t clamped = std::max(-128, std::min(127, q));
            block_out[4 + i] = static_cast<uint8_t>(static_cast<int8_t>(clamped));
        }
        const size_t bsums_off = 4 + QK_K;
        for (size_t g = 0; g < QK_K / 16; ++g) {
            int32_t sum = 0;
            for (size_t i = 0; i < 16; ++i) {
                sum += static_cast<int32_t>(static_cast<int8_t>(block_out[4 + g * 16 + i]));
            }
            const int16_t sum16 = static_cast<int16_t>(
                std::max(static_cast<int32_t>(-32768),
                         std::min(static_cast<int32_t>(32767), sum)));
            block_out[bsums_off + g * 2] = static_cast<uint8_t>(sum16 & 0xff);
            block_out[bsums_off + g * 2 + 1] = static_cast<uint8_t>((sum16 >> 8) & 0xff);
        }
    }
}

} // namespace oxk
