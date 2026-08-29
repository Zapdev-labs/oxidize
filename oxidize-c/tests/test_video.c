/* test_video.c — video multimodal tests. */
#include "framework.h"
#include "oxidize/video.h"
#include <string.h>

Test(video, sampler_init)
{
    OcVideoFrameSampler s;
    oc_video_sampler_init(&s, OC_VIDEO_FRAME_CONFIG_DEFAULT);
    cr_assert_eq(s.cfg.strategy, OC_FRAME_SAMPLE_UNIFORM);
    cr_assert_eq(s.cfg.n_frames, 8);
    cr_assert_eq(s.cfg.max_frames, 32);
    cr_assert(!s.has_prev_hist);
}

Test(video, sampler_plan_uniform)
{
    OcVideoFrameSampler s;
    oc_video_sampler_init(&s, OC_VIDEO_FRAME_CONFIG_DEFAULT);
    uint32_t indices[32];
    size_t count = 0;
    OcError e = oc_video_sampler_plan(&s, 100, indices, &count);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(count, 8, "should select 8 frames");
    cr_assert_eq(indices[0], 0);
    cr_assert_eq(indices[7], 87); /* 7 * 100 / 8 = 87 */
}

Test(video, sampler_plan_zero_frames)
{
    OcVideoFrameSampler s;
    oc_video_sampler_init(&s, OC_VIDEO_FRAME_CONFIG_DEFAULT);
    uint32_t indices[8];
    size_t count = 99;
    cr_assert_eq(oc_video_sampler_plan(&s, 0, indices, &count), OC_OK);
    cr_assert_eq(count, 0);
}

Test(video, sampler_plan_more_frames_than_total)
{
    OcVideoFrameSampler s;
    oc_video_sampler_init(&s, OC_VIDEO_FRAME_CONFIG_DEFAULT);
    uint32_t indices[32];
    size_t count = 0;
    /* 5 frames requested, only 3 available */
    OcVideoFrameConfig cfg = OC_VIDEO_FRAME_CONFIG_DEFAULT;
    cfg.n_frames = 5;
    cfg.max_frames = 32;
    oc_video_sampler_init(&s, cfg);
    cr_assert_eq(oc_video_sampler_plan(&s, 3, indices, &count), OC_OK);
    cr_assert_eq(count, 3, "should select all 3 frames");
}

Test(video, histogram_basic)
{
    OcVideoFrame frame = {
        .rgb = NULL,
        .width = 0,
        .height = 0,
        .timestamp_ms = 0,
        .frame_idx = 0,
    };
    uint32_t hist[64];
    /* NULL frame should produce all zeros. */
    oc_video_compute_histogram(&frame, hist);
    for (int i = 0; i < 64; i++)
        cr_assert_eq(hist[i], 0);
}

Test(video, histogram_diff_identical)
{
    uint32_t a[64] = {0};
    uint32_t b[64] = {0};
    for (int i = 0; i < 64; i++) { a[i] = 100; b[i] = 100; }
    cr_assert_float_eq(oc_video_histogram_diff(a, b), 0.0f, 1e-6f);
}

Test(video, histogram_diff_different)
{
    uint32_t a[64] = {0};
    uint32_t b[64] = {0};
    a[0] = 100;
    b[1] = 100;
    float diff = oc_video_histogram_diff(a, b);
    cr_assert(diff > 0.0f, "different histograms should have diff > 0");
}

Test(video, aggregate_mean)
{
    /* 3 frames, dim=4. */
    float embeddings[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
    };
    OcVideoEmbedding emb;
    OcError e = oc_video_aggregate(embeddings, 3, 4, OC_TEMPORAL_MEAN, &emb);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(emb.dim, 4);
    cr_assert_eq(emb.n_frames, 3);
    cr_assert_float_eq(emb.data[0], 2.0f, 0.01f);
    cr_assert_float_eq(emb.data[1], 3.0f, 0.01f);
    cr_assert_float_eq(emb.data[3], 5.0f, 0.01f);
    oc_video_embedding_free(&emb);
}

Test(video, aggregate_max)
{
    float embeddings[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    OcVideoEmbedding emb;
    cr_assert_eq(oc_video_aggregate(embeddings, 2, 4, OC_TEMPORAL_MAX, &emb), OC_OK);
    cr_assert_eq(emb.dim, 4);
    cr_assert_float_eq(emb.data[0], 5.0f, 0.01f);
    cr_assert_float_eq(emb.data[3], 8.0f, 0.01f);
    oc_video_embedding_free(&emb);
}

Test(video, aggregate_concat)
{
    float embeddings[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    OcVideoEmbedding emb;
    cr_assert_eq(oc_video_aggregate(embeddings, 2, 4, OC_TEMPORAL_CONCAT, &emb), OC_OK);
    cr_assert_eq(emb.dim, 8);
    cr_assert_float_eq(emb.data[0], 1.0f, 0.01f);
    cr_assert_float_eq(emb.data[7], 8.0f, 0.01f);
    oc_video_embedding_free(&emb);
}

Test(video, aggregate_attn)
{
    float embeddings[8] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    OcVideoEmbedding emb;
    cr_assert_eq(oc_video_aggregate(embeddings, 2, 4, OC_TEMPORAL_ATTN, &emb), OC_OK);
    cr_assert_eq(emb.dim, 4);
    /* Attention pooling should produce a weighted average. */
    cr_assert(emb.data[0] > 0.0f);
    cr_assert(emb.data[1] > 0.0f);
    oc_video_embedding_free(&emb);
}

OC_TEST_NULL_SAFE(video, aggregate_null,
        cr_assert_neq(oc_video_aggregate(NULL, 3, 4, OC_TEMPORAL_MEAN, NULL), OC_OK);)

Test(video, prompt_create)
{
    OcVideoEmbedding emb = { .data = NULL, .dim = 0, .n_frames = 0, .method = OC_TEMPORAL_MEAN };
    OcVideoPrompt p;
    cr_assert_eq(oc_video_prompt_create(&emb, "describe this video", &p), OC_OK);
    cr_assert_eq(p.video_emb, &emb);
    cr_assert_str_eq(p.text_prompt, "describe this video");
    oc_video_prompt_free(&p);
}

Test(video, sampler_reset)
{
    OcVideoFrameSampler s;
    oc_video_sampler_init(&s, OC_VIDEO_FRAME_CONFIG_DEFAULT);
    s.total_frames_seen = 100;
    s.frames_selected = 50;
    s.has_prev_hist = true;
    oc_video_sampler_reset(&s);
    cr_assert_eq(s.total_frames_seen, 0);
    cr_assert_eq(s.frames_selected, 0);
    cr_assert(!s.has_prev_hist);
}
