#ifndef OXIDIZE_ATTN_KERNELS_H
#define OXIDIZE_ATTN_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float oc_attn_dot_f32(const float *a, const float *b, size_t n);
void oc_attn_axpy_f32(float *y, const float *x, float alpha, size_t n);
void oc_attn_add_f32(float *y, const float *x, size_t n);
void oc_attn_scale_f32(float *y, float scale, size_t n);
void oc_attn_rms_apply_f32(const float *x, const float *weight, float inv,
                           float *out, size_t n);

float oc_attn_dot_q8(const float *a, const int8_t *b, size_t n);
void oc_attn_axpy_q8(float *y, const int8_t *x, float alpha, size_t n);

#ifdef __cplusplus
}
#endif

#endif
