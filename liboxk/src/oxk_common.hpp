#pragma once

#include <cstdint>
#include <cstring>

namespace oxk {

constexpr size_t QK_K = 256;
constexpr size_t BLOCK_Q4_K_SIZE = 144;
constexpr size_t BLOCK_Q8_K_BYTES = 292;

inline uint16_t read_u16_le(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline int16_t read_i16_le(const uint8_t *p) {
    return static_cast<int16_t>(read_u16_le(p));
}

inline float read_f32_le(const uint8_t *p) {
    float v;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

inline void write_f32_le(uint8_t *p, float v) {
    std::memcpy(p, &v, sizeof(float));
}

inline float f16_le_to_f32(uint8_t b0, uint8_t b1) {
    const uint16_t bits = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
    const uint32_t sign = (bits >> 15) & 1u;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t frac = bits & 0x03ffu;
    uint32_t f32_bits;
    if (exp == 0) {
        if (frac == 0) {
            f32_bits = sign << 31;
        } else {
            uint32_t frac_norm = frac;
            int32_t e = -14;
            while ((frac_norm & 0x0400u) == 0) {
                frac_norm <<= 1;
                --e;
            }
            frac_norm &= 0x03ffu;
            f32_bits = (sign << 31) | (static_cast<uint32_t>(e + 127) << 23) |
                       (frac_norm << 13);
        }
    } else if (exp == 0x1fu) {
        f32_bits = (sign << 31) | (0xffu << 23) | (frac << 13);
    } else {
        f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &f32_bits, sizeof(float));
    return out;
}

inline void get_scale_min_k4(size_t j, const uint8_t *scales, uint8_t &scale,
                             uint8_t &min_val) {
    if (j < 4) {
        scale = scales[j] & 63u;
        min_val = scales[j + 4] & 63u;
    } else {
        scale = (scales[j + 4] & 0x0fu) | ((scales[j - 4] >> 6) << 4);
        min_val = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

} // namespace oxk
