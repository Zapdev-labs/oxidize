/* video_decoder.h — Video frame decoding primitives (no FFmpeg). */
#ifndef OXIDIZE_VIDEO_DECODER_H
#define OXIDIZE_VIDEO_DECODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcVideoFrameFloat {
    uint32_t width;
    uint32_t height;
    uint32_t channels;   /* always 3 (RGB) */
    float   *data;        /* owned; [height * width * channels] */
} OcVideoFrame;

/* Owning, growable list of frames. */
typedef struct OcVideoFrameList {
    OcVideoFrame *frames;
    size_t        count;
    size_t        capacity;
} OcVideoFrameList;

/* Initialize a frame list with the given initial capacity. Capacity 0
 * is allowed (the list grows on first add). Returns OC_ERR_INVALID_ARG
 * if `list` is NULL, OC_ERR_OOM on allocation failure. */
OcError oc_video_frame_list_init(OcVideoFrameList *list, size_t capacity);

/* Free a frame list and all owned frame data. Safe on NULL list and
 * on already-freed lists (idempotent). */
void oc_video_frame_list_free(OcVideoFrameList *list);

/* Add a frame by copying `data` ([h * w * 3] floats). The list grows
 * automatically. Returns OC_ERR_INVALID_ARG on NULL/zero-dim args or
 * OC_ERR_OOM on allocation failure. */
OcError oc_video_frame_list_add(OcVideoFrameList *list,
                                 uint32_t w, uint32_t h,
                                 const float *data);

/* Add a pre-constructed frame by taking ownership of `frame->data`.
 * After this call, `frame` is zeroed and should not be reused by the
 * caller. Returns OC_ERR_INVALID_ARG on NULL/zero-dim args or OOM. */
OcError oc_video_frame_list_add_raw(OcVideoFrameList *list, OcVideoFrame *frame);

/* RepetitiveFrameDecoder: synthesize `n_copies` copies of a single image. Each copy is a freshly-allocated duplicate of `single_frame_data` ([h * w * 3] floats). The output list is initialized by this call; call oc_video_frame_list_free() after. Returns OC_ERR_INVALID_ARG if any pointer/dim is bad, n_copies==0, or OC_ERR_OOM. */
OcError oc_video_decoder_repetitive(OcVideoFrameList *out,
                                     uint32_t w, uint32_t h,
                                     const float *single_frame_data,
                                     size_t n_copies);

/* Get a borrowed pointer to frame `idx` in `list`. Returns OC_OK and
 * writes *out, or OC_ERR_INVALID_ARG on bad args / out-of-range idx. */
OcError oc_video_frame_get(const OcVideoFrameList *list, size_t idx,
                            const OcVideoFrame **out);

/* Size in bytes of a single frame's float data (w * h * 3 * sizeof(float)). */
size_t oc_video_frame_size_bytes(uint32_t w, uint32_t h);

/* Total size in bytes of all frame data in the list. */
size_t oc_video_frame_list_size_bytes(const OcVideoFrameList *list);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_VIDEO_DECODER_H */
