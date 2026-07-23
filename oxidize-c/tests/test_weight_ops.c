/* test_weight_ops.c — Weight matrix operations tests. */
#include <criterion/criterion.h>
#include "oxidize/weight_ops.h"
#include "oxidize/weight_storage.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Test(wops, gemv_f32_basic)
{
    /* W = [[1,2],[3,4]], x = [5,6] -> y = [17, 39] */
    float w[] = {1,2,3,4};
    float x[] = {5,6};
    float y[2];
    cr_assert_eq(oc_gemv_f32(w, 2, 2, x, y), OC_OK);
    cr_assert_float_eq(y[0], 17.0f, 0.001f);
    cr_assert_float_eq(y[1], 39.0f, 0.001f);
}

Test(wops, gemv_f32_identity)
{
    float w[] = {1,0,0,1};
    float x[] = {3,4};
    float y[2];
    oc_gemv_f32(w, 2, 2, x, y);
    cr_assert_float_eq(y[0], 3.0f, 0.001f);
    cr_assert_float_eq(y[1], 4.0f, 0.001f);
}

Test(wops, gemv_f32_null)
{
    cr_assert_neq(oc_gemv_f32(NULL, 1, 1, NULL, NULL), OC_OK);
}

Test(wops, gemv_f32_zero_dim)
{
    float w[] = {1};
    float x[] = {1};
    float y[1];
    cr_assert_neq(oc_gemv_f32(w, 0, 1, x, y), OC_OK);
}

Test(wops, gemv_weight_f32)
{
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    float *data = malloc(6 * sizeof(float));
    /* W = [[1,2,3],[4,5,6]] 2x3 */
    data[0]=1; data[1]=2; data[2]=3; data[3]=4; data[4]=5; data[5]=6;
    oc_weight_storage_f32(&ws, data, 6);
    float x[] = {1,0,1};  /* pick columns 0 and 2 */
    float y[2];
    cr_assert_eq(oc_gemv_weight(&ws, 2, 3, x, y), OC_OK);
    cr_assert_float_eq(y[0], 4.0f, 0.001f);   /* 1*1 + 2*0 + 3*1 */
    cr_assert_float_eq(y[1], 10.0f, 0.001f);  /* 4*1 + 5*0 + 6*1 */
    oc_weight_storage_free(&ws);
}

Test(wops, gemv_weight_null)
{
    float x[1], y[1];
    cr_assert_neq(oc_gemv_weight(NULL, 1, 1, x, y), OC_OK);
}

Test(wops, gemv_expert_f32)
{
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    /* 2 experts, each 2x2: expert0=[[1,2],[3,4]], expert1=[[5,6],[7,8]] */
    float *data = malloc(8 * sizeof(float));
    data[0]=1; data[1]=2; data[2]=3; data[3]=4;
    data[4]=5; data[5]=6; data[6]=7; data[7]=8;
    oc_weight_storage_f32(&ws, data, 8);
    float x[] = {1, 1};
    float y[2];
    /* Expert 0: [1+2, 3+4] = [3, 7] */
    cr_assert_eq(oc_gemv_expert_weight(&ws, 0, 2, 2, 2, x, y), OC_OK);
    cr_assert_float_eq(y[0], 3.0f, 0.001f);
    cr_assert_float_eq(y[1], 7.0f, 0.001f);
    /* Expert 1: [5+6, 7+8] = [11, 15] */
    cr_assert_eq(oc_gemv_expert_weight(&ws, 1, 2, 2, 2, x, y), OC_OK);
    cr_assert_float_eq(y[0], 11.0f, 0.001f);
    cr_assert_float_eq(y[1], 15.0f, 0.001f);
    oc_weight_storage_free(&ws);
}

Test(wops, gemv_expert_out_of_range)
{
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    float *data = malloc(8 * sizeof(float));
    oc_weight_storage_f32(&ws, data, 8);
    float x[] = {1, 1};
    float y[2];
    cr_assert_neq(oc_gemv_expert_weight(&ws, 5, 2, 2, 2, x, y), OC_OK);
    oc_weight_storage_free(&ws);
}

Test(wops, gemv_fused)
{
    OcWeightStorage ws1, ws2;
    oc_weight_storage_init(&ws1);
    oc_weight_storage_init(&ws2);
    float *d1 = malloc(4 * sizeof(float));
    d1[0]=1; d1[1]=2; d1[2]=3; d1[3]=4;
    oc_weight_storage_f32(&ws1, d1, 4);
    float *d2 = malloc(4 * sizeof(float));
    d2[0]=5; d2[1]=6; d2[2]=7; d2[3]=8;
    oc_weight_storage_f32(&ws2, d2, 4);

    float x[] = {1, 1};
    float y1[2], y2[2];
    OcGemvPart parts[2] = {
        {&ws1, 2, y1},
        {&ws2, 2, y2},
    };
    cr_assert_eq(oc_gemv_weight_fused(parts, 2, 2, x), OC_OK);
    cr_assert_float_eq(y1[0], 3.0f, 0.001f);
    cr_assert_float_eq(y1[1], 7.0f, 0.001f);
    cr_assert_float_eq(y2[0], 11.0f, 0.001f);
    cr_assert_float_eq(y2[1], 15.0f, 0.001f);
    oc_weight_storage_free(&ws1);
    oc_weight_storage_free(&ws2);
}

Test(wops, gemv_fused_skip_zero_rows)
{
    OcWeightStorage ws1;
    oc_weight_storage_init(&ws1);
    float *d1 = malloc(4 * sizeof(float));
    d1[0]=1; d1[1]=2; d1[2]=3; d1[3]=4;
    oc_weight_storage_f32(&ws1, d1, 4);

    float x[] = {1, 1};
    float y1[2] = {0, 0};
    float y2[2] = {99, 99};
    OcGemvPart parts[2] = {
        {&ws1, 2, y1},
        {NULL, 0, y2},  /* rows=0 -> skip */
    };
    cr_assert_eq(oc_gemv_weight_fused(parts, 2, 2, x), OC_OK);
    cr_assert_float_eq(y1[0], 3.0f, 0.001f);
    cr_assert_float_eq(y2[0], 99.0f, 0.001f);  /* unchanged */
    oc_weight_storage_free(&ws1);
}

Test(wops, gemm_weight_f32_batch)
{
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    /* W = [[1,2],[3,4]] 2x2 */
    float *data = malloc(4 * sizeof(float));
    data[0]=1; data[1]=2; data[2]=3; data[3]=4;
    oc_weight_storage_f32(&ws, data, 4);
    /* X = [[1,0],[0,1],[1,1]] 3x2 */
    float inputs[] = {1,0, 0,1, 1,1};
    float outputs[6];
    cr_assert_eq(oc_gemm_weight(&ws, 2, 2, inputs, outputs, 3), OC_OK);
    /* batch 0: [1,3], batch 1: [2,4], batch 2: [3,7] */
    cr_assert_float_eq(outputs[0], 1.0f, 0.001f);
    cr_assert_float_eq(outputs[1], 3.0f, 0.001f);
    cr_assert_float_eq(outputs[2], 2.0f, 0.001f);
    cr_assert_float_eq(outputs[3], 4.0f, 0.001f);
    cr_assert_float_eq(outputs[4], 3.0f, 0.001f);
    cr_assert_float_eq(outputs[5], 7.0f, 0.001f);
    oc_weight_storage_free(&ws);
}

Test(wops, gemm_weight_batch1_uses_gemv)
{
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    float *data = malloc(4 * sizeof(float));
    data[0]=1; data[1]=2; data[2]=3; data[3]=4;
    oc_weight_storage_f32(&ws, data, 4);
    float inputs[] = {1, 1};
    float outputs[2];
    cr_assert_eq(oc_gemm_weight(&ws, 2, 2, inputs, outputs, 1), OC_OK);
    cr_assert_float_eq(outputs[0], 3.0f, 0.001f);
    cr_assert_float_eq(outputs[1], 7.0f, 0.001f);
    oc_weight_storage_free(&ws);
}

Test(wops, add_repeating_bias)
{
    float buf[] = {1, 2, 3, 4, 5, 6};
    float bias[] = {10, 20};
    oc_add_repeating_bias(buf, 6, bias, 2);
    /* bias repeats: [10,20,10,20,10,20] */
    cr_assert_float_eq(buf[0], 11.0f, 0.001f);
    cr_assert_float_eq(buf[1], 22.0f, 0.001f);
    cr_assert_float_eq(buf[2], 13.0f, 0.001f);
    cr_assert_float_eq(buf[3], 24.0f, 0.001f);
    cr_assert_float_eq(buf[4], 15.0f, 0.001f);
    cr_assert_float_eq(buf[5], 26.0f, 0.001f);
}

Test(wops, add_repeating_bias_empty)
{
    float buf[] = {1, 2};
    float bias[] = {0};
    oc_add_repeating_bias(buf, 2, bias, 0);
    /* No change with zero-length bias. */
    cr_assert_float_eq(buf[0], 1.0f, 0.001f);
    cr_assert_float_eq(buf[1], 2.0f, 0.001f);
}

Test(wops, add_repeating_bias_null)
{
    float buf[] = {1};
    oc_add_repeating_bias(buf, 1, NULL, 0);
    cr_assert_float_eq(buf[0], 1.0f, 0.001f);
}

Test(wops, gemv_weight_q8_0)
{
    /* Create a Q8_0 weight matrix: 2 rows x 32 cols.
     * Q8_0 block: [d_f16(2 bytes), 32 x int8] = 34 bytes per block.
     * One block per row (32 cols). */
    OcWeightStorage ws;
    oc_weight_storage_init(&ws);
    uint8_t *data = malloc(68); /* 2 rows x 34 bytes */
    /* Row 0: d=1.0, values = 1..32 as int8 -> sum = 528 */
    data[0] = 0x00; data[1] = 0x3C;  /* f16 = 1.0 */
    for (int i = 0; i < 32; i++) data[2 + i] = (uint8_t)(i + 1);
    /* Row 1: d=2.0, values = -16..15 as int8 -> sum = -16 * 2 = -32 */
    data[34] = 0x00; data[35] = 0x40;  /* f16 = 2.0 */
    for (int i = 0; i < 32; i++) data[36 + i] = (uint8_t)(i - 16);
    oc_weight_storage_quantized(&ws, OC_QUANT_Q8_0, data, 68);

    float input[32];
    for (int i = 0; i < 32; i++) input[i] = 1.0f;
    float output[2];
    cr_assert_eq(oc_gemv_weight(&ws, 2, 32, input, output), OC_OK);
    /* Row 0: sum(1..32) * 1.0 = 528 */
    cr_assert_float_eq(output[0], 528.0f, 1.0f);
    /* Row 1: sum(-16..15) * 2.0 = -16 * 2 = -32 */
    cr_assert_float_eq(output[1], -32.0f, 1.0f);
    oc_weight_storage_free(&ws);
}
