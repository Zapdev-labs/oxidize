/* test_video_error.c — video pipeline error type tests. */
#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/video_error.h"

/* Unique suite name "video_error" to avoid collision with test_video.c's
 * "video" suite. */


Test(video_error, message_none)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_NONE), "no error");
}

Test(video_error, message_decode)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_DECODE),
                     "frame/video decoding failed");
}

Test(video_error, message_no_frames)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_NO_FRAMES),
                     "decoder produced no frames");
}

Test(video_error, message_invalid_config)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_INVALID_CONFIG),
                     "invalid video configuration");
}

Test(video_error, message_unsupported_format)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_UNSUPPORTED_FORMAT),
                     "unsupported video format");
}

Test(video_error, message_timeout)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_TIMEOUT),
                     "frame extraction timed out");
}

Test(video_error, message_oom)
{
    cr_assert_str_eq(oc_video_error_message(OC_VIDEO_ERR_OOM),
                     "out of memory");
}

Test(video_error, message_unknown_code)
{
    cr_assert_str_eq(oc_video_error_message((OcVideoError)999), "unknown");
}


Test(video_error, name_none)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_NONE),
                     "OC_VIDEO_ERR_NONE");
}

Test(video_error, name_decode)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_DECODE),
                     "OC_VIDEO_ERR_DECODE");
}

Test(video_error, name_no_frames)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_NO_FRAMES),
                     "OC_VIDEO_ERR_NO_FRAMES");
}

Test(video_error, name_invalid_config)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_INVALID_CONFIG),
                     "OC_VIDEO_ERR_INVALID_CONFIG");
}

Test(video_error, name_unsupported_format)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_UNSUPPORTED_FORMAT),
                     "OC_VIDEO_ERR_UNSUPPORTED_FORMAT");
}

Test(video_error, name_timeout)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_TIMEOUT),
                     "OC_VIDEO_ERR_TIMEOUT");
}

Test(video_error, name_oom)
{
    cr_assert_str_eq(oc_video_error_name(OC_VIDEO_ERR_OOM),
                     "OC_VIDEO_ERR_OOM");
}

Test(video_error, name_unknown_code)
{
    cr_assert_str_eq(oc_video_error_name((OcVideoError)999), "unknown");
}


Test(video_error, recoverable_none_is_false)
{
    cr_assert(!oc_video_error_is_recoverable(OC_VIDEO_ERR_NONE));
}

Test(video_error, recoverable_decode_is_true)
{
    cr_assert(oc_video_error_is_recoverable(OC_VIDEO_ERR_DECODE));
}

Test(video_error, recoverable_no_frames_is_false)
{
    cr_assert(!oc_video_error_is_recoverable(OC_VIDEO_ERR_NO_FRAMES));
}

Test(video_error, recoverable_invalid_config_is_false)
{
    cr_assert(!oc_video_error_is_recoverable(OC_VIDEO_ERR_INVALID_CONFIG));
}

Test(video_error, recoverable_unsupported_format_is_false)
{
    cr_assert(!oc_video_error_is_recoverable(OC_VIDEO_ERR_UNSUPPORTED_FORMAT));
}

Test(video_error, recoverable_timeout_is_true)
{
    cr_assert(oc_video_error_is_recoverable(OC_VIDEO_ERR_TIMEOUT));
}

Test(video_error, recoverable_oom_is_false)
{
    cr_assert(!oc_video_error_is_recoverable(OC_VIDEO_ERR_OOM));
}

Test(video_error, recoverable_unknown_code_is_false)
{
    cr_assert(!oc_video_error_is_recoverable((OcVideoError)999));
}


Test(video_error, to_oc_none)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_NONE), OC_OK);
}

Test(video_error, to_oc_decode)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_DECODE), OC_ERR_FORMAT);
}

Test(video_error, to_oc_no_frames)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_NO_FRAMES), OC_ERR_FORMAT);
}

Test(video_error, to_oc_invalid_config)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_INVALID_CONFIG),
                 OC_ERR_INVALID_ARG);
}

Test(video_error, to_oc_unsupported_format)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_UNSUPPORTED_FORMAT),
                 OC_ERR_FORMAT);
}

Test(video_error, to_oc_timeout)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_TIMEOUT), OC_ERR_IO);
}

Test(video_error, to_oc_oom)
{
    cr_assert_eq(oc_video_error_to_oc(OC_VIDEO_ERR_OOM), OC_ERR_OOM);
}

Test(video_error, to_oc_unknown_code)
{
    cr_assert_eq(oc_video_error_to_oc((OcVideoError)999), OC_ERR_INTERNAL);
}


Test(video_error, all_codes_have_nonnull_message_and_name)
{
    for (int i = 0; i < (int)OC_VIDEO_ERR__COUNT; ++i) {
        OcVideoError e = (OcVideoError)i;
        cr_assert(oc_video_error_message(e) != NULL);
        cr_assert(oc_video_error_name(e) != NULL);
        cr_assert(strlen(oc_video_error_message(e)) > 0u);
        cr_assert(strlen(oc_video_error_name(e)) > 0u);
    }
}
