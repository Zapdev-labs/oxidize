#include "../include/oxk.h"

#include "oxk_common.hpp"

namespace oxk {
void cpu_detect_init();
bool has_avx2();
void quantize_q8k_into(const float *vector, size_t n_blocks, uint8_t *out);
void gemv_q4k_range(const uint8_t *rows, size_t n_rows, size_t blocks_per_row,
                    const uint8_t *q8k, float *out);
int gemv_q4k_f32(const uint8_t *weight_rows, size_t n_rows, size_t in_dim, const float *vector,
                 float *output);
int gemv_quantized(uint32_t quant_type, const uint8_t *qbytes, size_t qbytes_len, size_t rows,
                   size_t cols, const float *vector, float *output);
float dot_f32(const float *a, const float *b, size_t n);
int rms_norm_f32(const float *x, const float *weight, size_t n, float eps, float *out);
} // namespace oxk

extern "C" {

int oxk_init(void) {
    oxk::cpu_detect_init();
    return 0;
}

const char *oxk_version(void) {
    return "0.1.0";
}

int oxk_has_avx2(void) {
    return oxk::has_avx2() ? 1 : 0;
}

int oxk_quantize_q8k(const float *vector, size_t n_blocks, uint8_t *out) {
    if (vector == nullptr || out == nullptr) {
        return -1;
    }
    oxk::quantize_q8k_into(vector, n_blocks, out);
    return 0;
}

int oxk_gemv_q4k_range(const uint8_t *rows, size_t n_rows, size_t blocks_per_row,
                       const uint8_t *q8k, float *out) {
    if (n_rows == 0 || blocks_per_row == 0 || rows == nullptr || q8k == nullptr ||
        out == nullptr) {
        return -1;
    }
    oxk::gemv_q4k_range(rows, n_rows, blocks_per_row, q8k, out);
    return 0;
}

int oxk_gemv_q4k_f32(const uint8_t *weight_rows, size_t n_rows, size_t in_dim,
                     const float *vector, float *output) {
    return oxk::gemv_q4k_f32(weight_rows, n_rows, in_dim, vector, output);
}

int oxk_gemv_quantized(uint32_t quant_type, const uint8_t *qbytes, size_t qbytes_len,
                       size_t rows, size_t cols, const float *vector, float *output) {
    return oxk::gemv_quantized(quant_type, qbytes, qbytes_len, rows, cols, vector, output);
}

float oxk_dot_f32(const float *a, const float *b, size_t n) {
    if (a == nullptr || b == nullptr || n == 0) {
        return 0.0f;
    }
    return oxk::dot_f32(a, b, n);
}

int oxk_rms_norm_f32(const float *x, const float *weight, size_t n, float eps, float *out) {
    if (x == nullptr || weight == nullptr || out == nullptr || n == 0) {
        return -1;
    }
    return oxk::rms_norm_f32(x, weight, n, eps, out);
}

} // extern "C"
