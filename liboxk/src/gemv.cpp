#include "oxk_common.hpp"

#include <vector>

namespace oxk {

bool has_avx2();
float q4k_q8k_row_dot_scalar(const uint8_t *row, size_t blocks_per_row, const uint8_t *q8k);
void quantize_q8k_into(const float *vector, size_t n_blocks, uint8_t *out);

#ifdef OXK_X86
float q4k_q8k_row_dot_avx2(const uint8_t *row, size_t blocks_per_row, const uint8_t *q8k);
void q4k_q8k_row_dot_x4_avx2(const uint8_t *rows_base, size_t row_bytes, size_t blocks_per_row,
                             const uint8_t *q8k, float out[4]);
void q4k_q8k_row_dot_x8_avx2(const uint8_t *rows_base, size_t row_bytes, size_t blocks_per_row,
                             const uint8_t *q8k, float out[8]);
#endif

void gemv_q4k_range(const uint8_t *rows, size_t n_rows, size_t blocks_per_row,
                    const uint8_t *q8k, float *out) {
    const size_t row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;

#ifdef OXK_X86
    if (has_avx2()) {
        size_t r = 0;
        while (r + 8 <= n_rows) {
            float octet[8];
            q4k_q8k_row_dot_x8_avx2(rows + r * row_bytes, row_bytes, blocks_per_row, q8k, octet);
            for (size_t i = 0; i < 8; ++i) {
                out[r + i] = octet[i];
            }
            r += 8;
        }
        while (r + 4 <= n_rows) {
            float quad[4];
            q4k_q8k_row_dot_x4_avx2(rows + r * row_bytes, row_bytes, blocks_per_row, q8k, quad);
            for (size_t i = 0; i < 4; ++i) {
                out[r + i] = quad[i];
            }
            r += 4;
        }
        while (r < n_rows) {
            out[r] = q4k_q8k_row_dot_avx2(rows + r * row_bytes, blocks_per_row, q8k);
            ++r;
        }
        return;
    }
#endif

    for (size_t r = 0; r < n_rows; ++r) {
        out[r] = q4k_q8k_row_dot_scalar(rows + r * row_bytes, blocks_per_row, q8k);
    }
}

int gemv_q4k_f32(const uint8_t *weight_rows, size_t n_rows, size_t in_dim, const float *vector,
                 float *output) {
    if (n_rows == 0 || in_dim == 0 || in_dim % QK_K != 0) {
        return -1;
    }
    const size_t blocks_per_row = in_dim / QK_K;
    std::vector<uint8_t> q8k(blocks_per_row * BLOCK_Q8_K_BYTES);
    quantize_q8k_into(vector, blocks_per_row, q8k.data());
    gemv_q4k_range(weight_rows, n_rows, blocks_per_row, q8k.data(), output);
    return 0;
}

/* Q8_0 block: 2-byte scale (f16) + 32 int8 values = 34 bytes, 32 values per block */
constexpr size_t BLOCK_Q8_0_SIZE = 34;

static float gemv_q8_0_row(const uint8_t *row, size_t cols, const float *vector) {
    const size_t n_blocks = cols / 32;
    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t *block = row + b * BLOCK_Q8_0_SIZE;
        const float d = f16_le_to_f32(block[0], block[1]);
        const uint8_t *qs = block + 2;
        for (size_t i = 0; i < 32; ++i) {
            acc += d * static_cast<float>(static_cast<int8_t>(qs[i])) * vector[b * 32 + i];
        }
    }
    return acc;
}

int gemv_quantized(uint32_t quant_type, const uint8_t *qbytes, size_t qbytes_len, size_t rows,
                   size_t cols, const float *vector, float *output) {
    if (rows == 0 || cols == 0 || qbytes == nullptr || vector == nullptr || output == nullptr) {
        return -1;
    }
    switch (quant_type) {
    case 11: // Q4_K_S
    case 12: // Q4_K_M
        return gemv_q4k_f32(qbytes, rows, cols, vector, output);
    case 6: { // Q8_0
        const size_t bytes_per_row = qbytes_len / rows;
        if (bytes_per_row * rows != qbytes_len) {
            return -1;
        }
        for (size_t r = 0; r < rows; ++r) {
            output[r] = gemv_q8_0_row(qbytes + r * bytes_per_row, cols, vector);
        }
        return 0;
    }
    default:
        return -1;
    }
}

} // namespace oxk
