#define _POSIX_C_SOURCE 200809L

#include "oxidize/video_error.h"

#include <stddef.h>


static const char *const k_messages[] = {
    "no error",
    "frame/video decoding failed",
    "decoder produced no frames",
    "invalid video configuration",
    "unsupported video format",
    "frame extraction timed out",
    "out of memory",
};

static const char *const k_names[] = {
    "OC_VIDEO_ERR_NONE",
    "OC_VIDEO_ERR_DECODE",
    "OC_VIDEO_ERR_NO_FRAMES",
    "OC_VIDEO_ERR_INVALID_CONFIG",
    "OC_VIDEO_ERR_UNSUPPORTED_FORMAT",
    "OC_VIDEO_ERR_TIMEOUT",
    "OC_VIDEO_ERR_OOM",
};

static const bool k_recoverable[] = {
    false, /* NONE                 */
    true,  /* DECODE (transient)   */
    false, /* NO_FRAMES            */
    false, /* INVALID_CONFIG       */
    false, /* UNSUPPORTED_FORMAT   */
    true,  /* TIMEOUT              */
    false, /* OOM                  */
};


const char *oc_video_error_message(OcVideoError err)
{
    if ((size_t)err >= (size_t)OC_VIDEO_ERR__COUNT) {
        return "unknown";
    }
    return k_messages[(size_t)err];
}

const char *oc_video_error_name(OcVideoError err)
{
    if ((size_t)err >= (size_t)OC_VIDEO_ERR__COUNT) {
        return "unknown";
    }
    return k_names[(size_t)err];
}

bool oc_video_error_is_recoverable(OcVideoError err)
{
    if ((size_t)err >= (size_t)OC_VIDEO_ERR__COUNT) {
        return false;
    }
    return k_recoverable[(size_t)err];
}

OcError oc_video_error_to_oc(OcVideoError err)
{
    switch (err) {
    case OC_VIDEO_ERR_NONE:
        return OC_OK;
    case OC_VIDEO_ERR_DECODE:
    case OC_VIDEO_ERR_NO_FRAMES:
        return OC_ERR_FORMAT;
    case OC_VIDEO_ERR_INVALID_CONFIG:
        return OC_ERR_INVALID_ARG;
    case OC_VIDEO_ERR_UNSUPPORTED_FORMAT:
        return OC_ERR_FORMAT;
    case OC_VIDEO_ERR_TIMEOUT:
        return OC_ERR_IO;
    case OC_VIDEO_ERR_OOM:
        return OC_ERR_OOM;
    default:
        return OC_ERR_INTERNAL;
    }
}
