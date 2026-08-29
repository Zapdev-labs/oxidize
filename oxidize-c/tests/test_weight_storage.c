/* test_weight_storage.c — Weight storage tests. */
#include "framework.h"
#include "oxidize/weight_storage.h"
#include <stdlib.h>
#include <string.h>

Test(wstore, init_empty)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    cr_assert_eq(w.type, OC_WEIGHT_F32);
    cr_assert(oc_weight_storage_is_empty(&w));
    oc_weight_storage_free(&w);
}

Test(wstore, f32_storage)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    float *data = malloc(3 * sizeof(float));
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
    cr_assert_eq(oc_weight_storage_f32(&w, data, 3), OC_OK);
    cr_assert_eq(w.type, OC_WEIGHT_F32);
    cr_assert(!oc_weight_storage_is_empty(&w));
    size_t len;
    const float *f = oc_weight_storage_f32_data(&w, &len);
    cr_assert_not_null(f);
    cr_assert_eq(len, 3);
    cr_assert_float_eq(f[0], 1.0f, 0.001f);
    oc_weight_storage_free(&w);
}

Test(wstore, f32_output_dim)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    float *data = malloc(12 * sizeof(float));
    memset(data, 0, 12 * sizeof(float));
    oc_weight_storage_f32(&w, data, 12);
    /* 12 elements / 4 input_dim = 3 output rows */
    cr_assert_eq(oc_weight_storage_output_dim(&w, 4), 3);
    oc_weight_storage_free(&w);
}

Test(wstore, quantized_storage)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    uint8_t *data = malloc(144);
    memset(data, 0x42, 144);
    cr_assert_eq(oc_weight_storage_quantized(&w, OC_QUANT_Q4_K_M, data, 144), OC_OK);
    cr_assert_eq(w.type, OC_WEIGHT_QUANTIZED);
    cr_assert_eq(w.qtype, OC_QUANT_Q4_K_M);
    cr_assert(!oc_weight_storage_is_empty(&w));
    size_t sz;
    const uint8_t *q = oc_weight_storage_quant_bytes(&w, &sz);
    cr_assert_not_null(q);
    cr_assert_eq(sz, 144);
    oc_weight_storage_free(&w);
}

Test(wstore, quantized_qtype)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    uint8_t *data = malloc(34);
    oc_weight_storage_quantized(&w, OC_QUANT_Q8_0, data, 34);
    cr_assert_eq(oc_weight_storage_qtype(&w), OC_QUANT_Q8_0);
    oc_weight_storage_free(&w);
}

Test(wstore, f32_qtype_unknown)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    float *data = malloc(4);
    oc_weight_storage_f32(&w, data, 1);
    cr_assert_eq(oc_weight_storage_qtype(&w), OC_QUANT_UNKNOWN);
    oc_weight_storage_free(&w);
}

Test(wstore, quant_bytes_null_for_f32)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    float *data = malloc(4);
    oc_weight_storage_f32(&w, data, 1);
    size_t sz;
    cr_assert_null(oc_weight_storage_quant_bytes(&w, &sz));
    oc_weight_storage_free(&w);
}

Test(wstore, f32_data_null_for_quantized)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    uint8_t *data = malloc(34);
    oc_weight_storage_quantized(&w, OC_QUANT_Q8_0, data, 34);
    size_t len;
    cr_assert_null(oc_weight_storage_f32_data(&w, &len));
    oc_weight_storage_free(&w);
}

Test(wstore, mmap_storage)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    /* Simulate mmap data. */
    static const uint8_t fake_mmap[256];
    cr_assert_eq(oc_weight_storage_mmap(&w, OC_QUANT_Q4_K_M,
                                         fake_mmap, 16, 144), OC_OK);
    cr_assert_eq(w.type, OC_WEIGHT_MMAP_QUANTIZED);
    cr_assert_eq(w.qtype, OC_QUANT_Q4_K_M);
    cr_assert(!oc_weight_storage_is_empty(&w));
    size_t sz;
    const uint8_t *q = oc_weight_storage_quant_bytes(&w, &sz);
    cr_assert_not_null(q);
    cr_assert_eq(sz, 144);
    cr_assert_eq(q, fake_mmap + 16);  /* points into mmap at offset */
    oc_weight_storage_free(&w);
    /* mmap data should NOT be freed. */
    cr_assert_not_null(fake_mmap);
}

Test(wstore, mmap_output_dim)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    static const uint8_t fake_mmap[4096];
    /* Q8_0: block_width=32, block_size=34, input_dim=256 -> 8 blocks -> 272 bytes/row */
    oc_weight_storage_mmap(&w, OC_QUANT_Q8_0, fake_mmap, 0, 272);
    cr_assert_eq(oc_weight_storage_output_dim(&w, 256), 1);
    oc_weight_storage_free(&w);
}

Test(wstore, quantized_output_dim)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    /* Q8_0: block_width=32, block_size=34, input_dim=256 -> 8 blocks -> 272 bytes/row */
    uint8_t *data = malloc(272 * 3);
    memset(data, 0, 272 * 3);
    oc_weight_storage_quantized(&w, OC_QUANT_Q8_0, data, 272 * 3);
    cr_assert_eq(oc_weight_storage_output_dim(&w, 256), 3);
    oc_weight_storage_free(&w);
}

Test(wstore, empty_after_init)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    cr_assert(oc_weight_storage_is_empty(&w));
    cr_assert_eq(oc_weight_storage_output_dim(&w, 1), 0);
    oc_weight_storage_free(&w);
}

Test(wstore, free_null)
{
    oc_weight_storage_free(NULL);
}

Test(wstore, reassign_f32_to_quant)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    float *fdata = malloc(8);
    oc_weight_storage_f32(&w, fdata, 8);
    /* Now reassign to quantized — old f32 data should be freed. */
    uint8_t *qdata = malloc(34);
    cr_assert_eq(oc_weight_storage_quantized(&w, OC_QUANT_Q8_0, qdata, 34), OC_OK);
    cr_assert_eq(w.type, OC_WEIGHT_QUANTIZED);
    cr_assert_eq(w.qtype, OC_QUANT_Q8_0);
    oc_weight_storage_free(&w);
}

Test(wstore, quantized_bad_qtype)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    uint8_t *data = malloc(34);
    cr_assert_neq(oc_weight_storage_quantized(&w, OC_QUANT_UNKNOWN, data, 34), OC_OK);
    free(data);
    oc_weight_storage_free(&w);
}

Test(wstore, mmap_bad_args)
{
    OcWeightStorage w;
    oc_weight_storage_init(&w);
    cr_assert_neq(oc_weight_storage_mmap(&w, OC_QUANT_Q8_0, NULL, 0, 34), OC_OK);
    static const uint8_t fake[64];
    cr_assert_neq(oc_weight_storage_mmap(&w, OC_QUANT_UNKNOWN, fake, 0, 34), OC_OK);
    oc_weight_storage_free(&w);
}
