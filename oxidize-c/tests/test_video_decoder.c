/* test_video_decoder.c — OcVideoFrameList + repetitive decoder tests.
 *
 * Unique suite name "video_decoder".
 */
#include "framework.h"
#include <string.h>

#include "oxidize/video_decoder.h"

/* ─── list lifecycle ────────────────────────────────────────────────── */

Test(video_decoder, list_init_zero_capacity)
{
    OcVideoFrameList list;
    cr_assert_eq(oc_video_frame_list_init(&list, 0), OC_OK, "");
    cr_assert_eq(list.count, 0u, "");
    cr_assert_eq(list.capacity, 0u, "");
    cr_assert_null(list.frames, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, list_init_with_capacity)
{
    OcVideoFrameList list;
    cr_assert_eq(oc_video_frame_list_init(&list, 4), OC_OK, "");
    cr_assert_eq(list.count, 0u, "");
    cr_assert_eq(list.capacity, 4u, "");
    cr_assert_not_null(list.frames, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, list_init_null)
{
    cr_assert_eq(oc_video_frame_list_init(NULL, 4), OC_ERR_INVALID_ARG, "");
}

Test(video_decoder, list_free_null_is_safe)
{
    oc_video_frame_list_free(NULL);
    cr_assert(true, "");
}

Test(video_decoder, list_free_double_is_safe)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 2);
    oc_video_frame_list_free(&list);
    oc_video_frame_list_free(&list);
    cr_assert(true, "");
}

/* ─── add ──────────────────────────────────────────────────────────── */

Test(video_decoder, add_frame_basic)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    float data[6] = {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f}; /* 2x1x3 */
    cr_assert_eq(oc_video_frame_list_add(&list, 2, 1, data), OC_OK, "");
    cr_assert_eq(list.count, 1u, "");
    cr_assert_eq(list.frames[0].width, 2u, "");
    cr_assert_eq(list.frames[0].height, 1u, "");
    cr_assert_eq(list.frames[0].channels, 3u, "");
    cr_assert_not_null(list.frames[0].data, "");
    cr_assert_float_eq(list.frames[0].data[0], 0.f, 1e-6, "");
    cr_assert_float_eq(list.frames[0].data[5], 0.5f, 1e-6, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, add_frame_null_args)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    cr_assert_eq(oc_video_frame_list_add(NULL, 1, 1, (const float *)"x"),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_frame_list_add(&list, 0, 1, (const float *)"x"),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_frame_list_add(&list, 1, 0, (const float *)"x"),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_frame_list_add(&list, 1, 1, NULL),
                 OC_ERR_INVALID_ARG, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, add_grows_capacity)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    float data[3] = {0.f, 0.5f, 1.f};
    for (int i = 0; i < 10; ++i) {
        cr_assert_eq(oc_video_frame_list_add(&list, 1, 1, data), OC_OK, "");
    }
    cr_assert_eq(list.count, 10u, "");
    cr_assert_geq(list.capacity, 10u, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, add_raw_takes_ownership)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    OcVideoFrame f;
    f.width = 1; f.height = 1; f.channels = 3;
    f.data = (float *)malloc(sizeof(float) * 3);
    f.data[0] = 0.25f; f.data[1] = 0.5f; f.data[2] = 0.75f;
    cr_assert_eq(oc_video_frame_list_add_raw(&list, &f), OC_OK, "");
    cr_assert_eq(list.count, 1u, "");
    cr_assert_null(f.data, "caller struct zeroed");
    cr_assert_eq(f.width, 0u, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, add_raw_bad_channels)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    OcVideoFrame f;
    f.width = 1; f.height = 1; f.channels = 4;
    f.data = (float *)malloc(sizeof(float) * 4);
    cr_assert_eq(oc_video_frame_list_add_raw(&list, &f), OC_ERR_INVALID_ARG, "");
    free(f.data);
    oc_video_frame_list_free(&list);
}

/* ─── repetitive decoder ──────────────────────────────────────────── */

Test(video_decoder, repetitive_basic)
{
    OcVideoFrameList out;
    float frame[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; /* 2x1x3 */
    cr_assert_eq(oc_video_decoder_repetitive(&out, 2, 1, frame, 5), OC_OK, "");
    cr_assert_eq(out.count, 5u, "");
    for (size_t i = 0; i < out.count; ++i) {
        cr_assert_eq(out.frames[i].width, 2u, "");
        cr_assert_eq(out.frames[i].height, 1u, "");
        cr_assert_eq(out.frames[i].channels, 3u, "");
        cr_assert_float_eq(out.frames[i].data[0], 1.f, 1e-6, "");
        cr_assert_float_eq(out.frames[i].data[5], 6.f, 1e-6, "");
    }
    oc_video_frame_list_free(&out);
}

Test(video_decoder, repetitive_null_args)
{
    OcVideoFrameList out;
    float frame[3] = {0.f, 0.f, 0.f};
    cr_assert_eq(oc_video_decoder_repetitive(NULL, 1, 1, frame, 1),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_decoder_repetitive(&out, 1, 1, NULL, 1),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_decoder_repetitive(&out, 0, 1, frame, 1),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_decoder_repetitive(&out, 1, 0, frame, 1),
                 OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_decoder_repetitive(&out, 1, 1, frame, 0),
                 OC_ERR_INVALID_ARG, "");
}

/* ─── get ─────────────────────────────────────────────────────────── */

Test(video_decoder, get_by_index)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    float data[3] = {0.1f, 0.2f, 0.3f};
    oc_video_frame_list_add(&list, 1, 1, data);
    const OcVideoFrame *out = NULL;
    cr_assert_eq(oc_video_frame_get(&list, 0, &out), OC_OK, "");
    cr_assert_not_null(out, "");
    cr_assert_eq(out->width, 1u, "");
    cr_assert_float_eq(out->data[2], 0.3f, 1e-6, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, get_out_of_range)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    float data[3] = {0.f, 0.f, 0.f};
    oc_video_frame_list_add(&list, 1, 1, data);
    const OcVideoFrame *out = NULL;
    cr_assert_eq(oc_video_frame_get(&list, 99, &out), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_frame_get(NULL, 0, &out), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_video_frame_get(&list, 0, NULL), OC_ERR_INVALID_ARG, "");
    oc_video_frame_list_free(&list);
}

/* ─── size helpers ────────────────────────────────────────────────── */

Test(video_decoder, frame_size_bytes)
{
    /* 2x3 RGB floats = 18 floats = 72 bytes */
    cr_assert_eq(oc_video_frame_size_bytes(2, 3), 72u, "");
    cr_assert_eq(oc_video_frame_size_bytes(0, 0), 0u, "");
}

Test(video_decoder, list_size_bytes)
{
    OcVideoFrameList list;
    oc_video_frame_list_init(&list, 1);
    float data[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; /* 2x1x3 = 24 bytes */
    oc_video_frame_list_add(&list, 2, 1, data);
    oc_video_frame_list_add(&list, 2, 1, data);
    cr_assert_eq(oc_video_frame_list_size_bytes(&list), 48u, "");
    oc_video_frame_list_free(&list);
}

Test(video_decoder, list_size_bytes_null)
{
    cr_assert_eq(oc_video_frame_list_size_bytes(NULL), 0u, "");
}
