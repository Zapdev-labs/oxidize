/* common.glsl — shared declarations for the oxidize-c compute shaders. */
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer UV4 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer FV4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer IV4 { ivec4 v[]; };

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64 { uint64_t v[]; };

float bf16lo(uint w) { return uintBitsToFloat(w << 16); }
float bf16hi(uint w) { return uintBitsToFloat(w & 0xFFFF0000u); }

const float NEG_INF = uintBitsToFloat(0xFF800000u);

/* Four E2M1 codes, one per byte (0..15), to four int8 holding twice the
 * value: magnitudes {0,1,2,3,4,6,8,12}, bit 3 the sign. */
uint e2m1x4(uint b)
{
    uint m = b & 0x07070707u;
    uint t = m & 0x03030303u & (((m >> 2) & 0x01010101u) * 0xFFu);
    uint v = m + t + ((m & (m >> 1) & (m >> 2) & 0x01010101u) << 1);
    uint neg = (b >> 3) & (m | (m >> 1) | (m >> 2)) & 0x01010101u;
    return (v ^ (neg * 0xFFu)) + neg;
}

/* Half of a UE4M3 scale byte (the half undoes the doubled E2M1 integers). */
float ue4m3h(uint b)
{
    b &= 0x7Fu;
    if (b == 0x7Fu) return 0.0;
    uint e = b >> 3, m = b & 7u;
    return e == 0u ? float(m) * (1.0 / 1024.0) : uintBitsToFloat(((e + 119u) << 23) | (m << 20));
}

vec4 ue4m3h4(uint w)
{
    return vec4(ue4m3h(w & 0xFFu), ue4m3h((w >> 8) & 0xFFu), ue4m3h((w >> 16) & 0xFFu), ue4m3h(w >> 24));
}

/* Two nibble words of one 16-value sub-block (block words 2s and 2s+1) to
 * its 16 int8 in natural order. */
uvec4 e2m1_sub(uint w0, uint w1)
{
    return uvec4(e2m1x4(w0 & 0x0F0F0F0Fu), e2m1x4(w1 & 0x0F0F0F0Fu),
                 e2m1x4((w0 >> 4) & 0x0F0F0F0Fu), e2m1x4((w1 >> 4) & 0x0F0F0F0Fu));
}

int dot4(uint a, uint b) { return dotPacked4x8EXT(int(a), int(b)); }

int dot16(uvec4 a, uvec4 b) { return dot4(a.x, b.x) + dot4(a.y, b.y) + dot4(a.z, b.z) + dot4(a.w, b.w); }

float tanh_s(float x)
{
    float e = exp(-2.0 * abs(x));
    return sign(x) * (1.0 - e) / (1.0 + e);
}

float sigmoid_s(float x) { return 1.0 / (1.0 + exp(-x)); }

/* Inkling "situ": soft-capped SiLU gate times soft-capped up. */
float situ(float g, float u, float beta, float lbeta)
{
    float ga = beta * tanh_s(g / beta) * sigmoid_s(g);
    return ga * (lbeta > 0.0 ? lbeta * tanh_s(u / lbeta) : u);
}

#define POST_NONE    0u
#define POST_SITU    1u
#define POST_SOFTCAP 2u
#define POST_SITU_Q  3u

/* Push block of the NVFP4 matmul kernels (nvfp4_mm, nvfp4_mv).
 *   y[c][r] = gscale[mat] * sum_k W[mat][r][k] * A[cidx[c]][k]
 * Weights are planar NVFP4 (oc_nvfp4_planar): per matrix, rows * nb blocks
 * of 32 nibble bytes, then rows * nb scale words. Columns c run over
 * [first, first + count): one item of a grouped list (items != 0: ivec4
 * {mat, first, count, 0} per workgroup row y) or the dense range [0, n).
 * With POST_SITU the rows are [0, ff) gate rows paired with [ff, 2ff) up
 * rows and y receives situ(gate, up) in [c][ff]. */
#define MM_PUSH \
    layout(push_constant) uniform P { \
        uint64_t w, gscale, aq, ad, items, cidx, y, mat_bytes; \
        uint rows, cols, n, y_stride; \
        uint aq_stride, ad_stride, post, ff; \
        float beta, lbeta, cap, pad; \
        uint64_t yd; \
    } p;
