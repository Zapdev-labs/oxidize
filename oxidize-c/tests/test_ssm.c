/* test_ssm.c — State-Space Model (SSM) recurrent state tests. */
#include "framework.h"
#include "oxidize/ssm.h"
#include <string.h>

Test(ssm, conv_ring_init)
{
    OcSsmConvRing ring;
    cr_assert_eq(oc_ssm_conv_ring_init(&ring, 4, 8), OC_OK);
    cr_assert_eq(ring.capacity, 4);
    cr_assert_eq(ring.dim, 8);
    cr_assert_eq(ring.head, 0);
    cr_assert_eq(ring.len, 0);
    oc_ssm_conv_ring_free(&ring);
}

OC_TEST_NULL_SAFE(ssm, conv_ring_init_null,
        cr_assert_neq(oc_ssm_conv_ring_init(NULL, 4, 8), OC_OK);)

Test(ssm, conv_ring_push)
{
    OcSsmConvRing ring;
    oc_ssm_conv_ring_init(&ring, 4, 4);
    float frame[4] = {1, 2, 3, 4};
    cr_assert_eq(oc_ssm_conv_ring_push(&ring, frame, 4), OC_OK);
    cr_assert_eq(oc_ssm_conv_ring_len(&ring), 1);
    oc_ssm_conv_ring_free(&ring);
}

Test(ssm, conv_ring_push_wraparound)
{
    OcSsmConvRing ring;
    oc_ssm_conv_ring_init(&ring, 2, 2);
    float f1[2] = {1, 2};
    float f2[2] = {3, 4};
    float f3[2] = {5, 6};
    oc_ssm_conv_ring_push(&ring, f1, 2);
    oc_ssm_conv_ring_push(&ring, f2, 2);
    oc_ssm_conv_ring_push(&ring, f3, 2);
    cr_assert_eq(oc_ssm_conv_ring_len(&ring), 2);
    const float *past;
    size_t past_len;
    cr_assert_eq(oc_ssm_conv_ring_past(&ring, 1, &past, &past_len), OC_OK);
    cr_assert_eq(past_len, 2);
    cr_assert_float_eq(past[0], 5.0f, 0.001f);
    cr_assert_float_eq(past[1], 6.0f, 0.001f);
    oc_ssm_conv_ring_free(&ring);
}

Test(ssm, conv_ring_past_invalid)
{
    OcSsmConvRing ring;
    oc_ssm_conv_ring_init(&ring, 4, 2);
    const float *past;
    size_t past_len;
    cr_assert_neq(oc_ssm_conv_ring_past(&ring, 1, &past, &past_len), OC_OK);
    oc_ssm_conv_ring_free(&ring);
}

Test(ssm, conv_ring_push_dim_mismatch)
{
    OcSsmConvRing ring;
    oc_ssm_conv_ring_init(&ring, 4, 4);
    float frame[2] = {1, 2};
    cr_assert_neq(oc_ssm_conv_ring_push(&ring, frame, 2), OC_OK);
    oc_ssm_conv_ring_free(&ring);
}

Test(ssm, conv_ring_checksum)
{
    OcSsmConvRing ring;
    oc_ssm_conv_ring_init(&ring, 4, 2);
    float f[2] = {1.0f, 2.0f};
    oc_ssm_conv_ring_push(&ring, f, 2);
    double cs = oc_ssm_conv_ring_checksum(&ring);
    cr_assert(cs > 0.0);
    oc_ssm_conv_ring_free(&ring);
}

Test(ssm, engine_init)
{
    OcSsmEngine engine;
    cr_assert_eq(oc_ssm_engine_init(&engine, 2, 64, 16, 64), OC_OK);
    cr_assert_eq(engine.n_layers, 2);
    cr_assert_eq(engine.state_dim, 64);
    cr_assert_eq(engine.ssm_pos, 0);
    cr_assert_eq(engine.n_checkpoints, 0);
    cr_assert_not_null(engine.ssm_states);
    cr_assert_not_null(engine.conv_buffers);
    oc_ssm_engine_free(&engine);
}

OC_TEST_NULL_SAFE(ssm, engine_init_null,
        cr_assert_neq(oc_ssm_engine_init(NULL, 1, 1, 1, 1), OC_OK);)

Test(ssm, engine_init_bad_args)
{
    OcSsmEngine engine;
    cr_assert_neq(oc_ssm_engine_init(&engine, 0, 64, 16, 64), OC_OK);
    cr_assert_neq(oc_ssm_engine_init(&engine, 2, 0, 16, 64), OC_OK);
}

Test(ssm, advance)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    cr_assert_eq(oc_ssm_advance(&engine, 5), OC_OK);
    cr_assert_eq(oc_ssm_position(&engine), 5);
    oc_ssm_engine_free(&engine);
}

Test(ssm, push_checkpoint)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_advance(&engine, 3);
    cr_assert_eq(oc_ssm_push_checkpoint(&engine, 3), OC_OK);
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), 1);
    oc_ssm_engine_free(&engine);
}

Test(ssm, push_duplicate_checkpoint)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_advance(&engine, 3);
    oc_ssm_push_checkpoint(&engine, 3);
    oc_ssm_push_checkpoint(&engine, 3);
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), 1);
    oc_ssm_engine_free(&engine);
}

Test(ssm, push_max_checkpoints)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_advance(&engine, 1);
    oc_ssm_push_checkpoint(&engine, 1);
    oc_ssm_advance(&engine, 1);
    oc_ssm_push_checkpoint(&engine, 2);
    oc_ssm_advance(&engine, 1);
    oc_ssm_push_checkpoint(&engine, 3);
    /* Should only keep 2 (OC_SSM_MAX_CHECKPOINTS). */
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), OC_SSM_MAX_CHECKPOINTS);
    oc_ssm_engine_free(&engine);
}

Test(ssm, rollback)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_advance(&engine, 5);
    oc_ssm_push_checkpoint(&engine, 5);
    oc_ssm_advance(&engine, 3);
    cr_assert_eq(oc_ssm_position(&engine), 8);
    cr_assert_eq(oc_ssm_rollback(&engine, 5), OC_OK);
    cr_assert_eq(oc_ssm_position(&engine), 5);
    oc_ssm_engine_free(&engine);
}

Test(ssm, rollback_not_found)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    cr_assert_neq(oc_ssm_rollback(&engine, 99), OC_OK);
    oc_ssm_engine_free(&engine);
}

Test(ssm, clear_checkpoints)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_push_checkpoint(&engine, 0);
    oc_ssm_push_checkpoint(&engine, 1);
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), 2);
    oc_ssm_clear_checkpoints(&engine);
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), 0);
    oc_ssm_engine_free(&engine);
}

Test(ssm, reset)
{
    OcSsmEngine engine;
    oc_ssm_engine_init(&engine, 1, 32, 8, 32);
    oc_ssm_advance(&engine, 10);
    oc_ssm_push_checkpoint(&engine, 5);
    oc_ssm_reset(&engine);
    cr_assert_eq(oc_ssm_position(&engine), 0);
    cr_assert_eq(oc_ssm_n_checkpoints(&engine), 0);
    oc_ssm_engine_free(&engine);
}

OC_TEST_NULL_SAFE(ssm, free_null,
        oc_ssm_engine_free(NULL);)
