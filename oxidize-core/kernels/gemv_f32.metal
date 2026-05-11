#include <metal_stdlib>
using namespace metal;

kernel void gemv_f32_kernel(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }

    float sum = 0.0f;
    uint row_offset = gid * cols;
    for (uint col = 0; col < cols; ++col) {
        sum += matrix[row_offset + col] * vector[col];
    }
    output[gid] = sum;
}

kernel void gemv_q8_0_f32_kernel(
    device const uchar* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) {
        return;
    }

    constexpr uint qk = 32;
    constexpr uint block_size = 2 + qk;
    const uint blocks_per_row = cols / qk;
    const uint row_offset = gid * blocks_per_row * block_size;

    float sum = 0.0f;
    for (uint block = 0; block < blocks_per_row; ++block) {
        const uint block_offset = row_offset + block * block_size;
        const ushort scale_bits = (ushort(matrix[block_offset + 1]) << 8) | ushort(matrix[block_offset]);
        const half scale_half = as_type<half>(scale_bits);
        const float scale = float(scale_half);

        const uint vector_offset = block * qk;
        for (uint i = 0; i < qk; ++i) {
            const int8_t q = int8_t(matrix[block_offset + 2 + i]);
            sum += float(q) * scale * vector[vector_offset + i];
        }
    }

    output[gid] = sum;
}
