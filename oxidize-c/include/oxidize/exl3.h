#ifndef OXIDIZE_EXL3_H
#define OXIDIZE_EXL3_H
#include "oxidize/error.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Packed little-endian bytes, input-tile-major then output-tile-major.
 * Scales/bias are owned by the overlay; this view owns no storage. */
typedef struct OcExl3Matrix {
    size_t input_dim, output_dim;
    const uint8_t *packed;
    size_t packed_size;
    unsigned bits_twice, codebook;
    const float *suh, *svh, *bias;
} OcExl3Matrix;
float oc_exl3_codebook_value(unsigned codebook, uint16_t idx);
OcError oc_exl3_validate(const OcExl3Matrix *m);
OcError oc_exl3_decode_tile(const OcExl3Matrix *m, size_t input_tile,
                          size_t output_tile, float out[256]);
/* Strides count floats. The process-global parallel pool must not be called
 * concurrently or recursively, as required by oc_parallel_for. */
OcError oc_exl3_matmul(const OcExl3Matrix *m, const float *in,
                      size_t in_stride, float *out, size_t out_stride, size_t batch);
OcError oc_exl3_matvec(const OcExl3Matrix *m, const float *in, float *out);
#ifdef __cplusplus
}
#endif
#endif
