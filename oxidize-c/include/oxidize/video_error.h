/*
 * video_error.h — Error type for the video multimodal pipeline.
 *
 * Port of oxidize-core/src/video/error.rs. Provides a small enum of
 * video-specific failure modes distinct from the generic OcError codes,
 * along with conversion helpers to OcError and recoverability flags.
 *
 * Video errors are produced by the frame sampler / decoder glue and are
 * meant to be surfaced to callers via oc_video_error_to_oc() when a
 * generic OcError return contract is required.
 */
#ifndef OXIDIZE_VIDEO_ERROR_H
#define OXIDIZE_VIDEO_ERROR_H

#include <stdbool.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Video error codes ───────────────────────────────────────────────── */

typedef enum {
    OC_VIDEO_ERR_NONE = 0,
    OC_VIDEO_ERR_DECODE,             /* frame/video decoding failure       */
    OC_VIDEO_ERR_NO_FRAMES,          /* decoder produced no frames         */
    OC_VIDEO_ERR_INVALID_CONFIG,     /* bad sampler / decoder configuration */
    OC_VIDEO_ERR_UNSUPPORTED_FORMAT, /* container/codec not supported      */
    OC_VIDEO_ERR_TIMEOUT,            /* frame extraction timed out         */
    OC_VIDEO_ERR_OOM,                /* allocation failure                 */
    /* sentinel for bounds checking; not a valid error code */
    OC_VIDEO_ERR__COUNT,
} OcVideoError;

/* ─── API ─────────────────────────────────────────────────────────────── */

/* Human-readable, NUL-terminated error message. Returns "unknown" for codes
 * outside the enum range. Never returns NULL. */
const char *oc_video_error_message(OcVideoError err);

/* Short, NUL-terminated enum-style name (e.g. "OC_VIDEO_ERR_DECODE").
 * Returns "unknown" for codes outside the enum range. Never returns NULL. */
const char *oc_video_error_name(OcVideoError err);

/* True if the error is recoverable (retry may succeed). Returns false for
 * OC_VIDEO_ERR_NONE. Returns false for codes outside the enum range. */
bool oc_video_error_is_recoverable(OcVideoError err);

/* Map a video error to the closest generic OcError code. Returns OC_OK for
 * OC_VIDEO_ERR_NONE and OC_ERR_INTERNAL for unknown codes. */
OcError oc_video_error_to_oc(OcVideoError err);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_ERROR_H */
