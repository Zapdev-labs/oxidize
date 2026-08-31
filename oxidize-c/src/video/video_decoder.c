/* video_decoder.c — Video frame decoding primitives implementation. */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/video_decoder.h"

#include <stdlib.h>
#include <string.h>


static size_t frame_float_count(uint32_t w, uint32_t h)
{
    return (size_t)w * (size_t)h * 3u;
}

static OcError frame_alloc_data(uint32_t w, uint32_t h, float **out)
{
    if (w == 0 || h == 0) {
        return OC_ERR_INVALID_ARG;
    }
    size_t n = frame_float_count(w, h);
    float *p = (float *)malloc(n * sizeof(float));
    if (p == NULL) {
        return OC_ERR_OOM;
    }
    *out = p;
    return OC_OK;
}

static OcError list_grow(OcVideoFrameList *list)
{
    size_t new_cap = list->capacity == 0 ? 4 : list->capacity * 2;
    OcVideoFrame *p = (OcVideoFrame *)realloc(list->frames,
                                              new_cap * sizeof(OcVideoFrame));
    if (p == NULL) {
        return OC_ERR_OOM;
    }
    /* Zero the new slots so free() is safe on early failures. */
    memset(p + list->count, 0,
           (new_cap - list->count) * sizeof(OcVideoFrame));
    list->frames   = p;
    list->capacity = new_cap;
    return OC_OK;
}


OcError oc_video_frame_list_init(OcVideoFrameList *list, size_t capacity)
{
    if (list == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    list->frames   = NULL;
    list->count    = 0;
    list->capacity = 0;
    if (capacity > 0) {
        list->frames = (OcVideoFrame *)calloc(capacity, sizeof(OcVideoFrame));
        if (list->frames == NULL) {
            return OC_ERR_OOM;
        }
        list->capacity = capacity;
    }
    return OC_OK;
}

void oc_video_frame_list_free(OcVideoFrameList *list)
{
    if (list == NULL) {
        return;
    }
    if (list->frames != NULL) {
        for (size_t i = 0; i < list->count; ++i) {
            free(list->frames[i].data);
            list->frames[i].data = NULL;
        }
        free(list->frames);
        list->frames = NULL;
    }
    list->count    = 0;
    list->capacity = 0;
}


OcError oc_video_frame_list_add(OcVideoFrameList *list,
                                 uint32_t w, uint32_t h,
                                 const float *data)
{
    if (list == NULL || data == NULL || w == 0 || h == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (list->count == list->capacity) {
        OcError e = list_grow(list);
        if (e != OC_OK) {
            return e;
        }
    }
    float *copy = NULL;
    OcError e = frame_alloc_data(w, h, &copy);
    if (e != OC_OK) {
        return e;
    }
    memcpy(copy, data, frame_float_count(w, h) * sizeof(float));

    OcVideoFrame *f = &list->frames[list->count];
    f->width    = w;
    f->height   = h;
    f->channels = 3;
    f->data     = copy;
    list->count += 1;
    return OC_OK;
}

OcError oc_video_frame_list_add_raw(OcVideoFrameList *list, OcVideoFrame *frame)
{
    if (list == NULL || frame == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (frame->data == NULL || frame->width == 0 || frame->height == 0) {
        return OC_ERR_INVALID_ARG;
    }
    /* Channels must be 3 for this decoder. */
    if (frame->channels != 3) {
        return OC_ERR_INVALID_ARG;
    }
    if (list->count == list->capacity) {
        OcError e = list_grow(list);
        if (e != OC_OK) {
            return e;
        }
    }
    OcVideoFrame *dst = &list->frames[list->count];
    dst->width    = frame->width;
    dst->height   = frame->height;
    dst->channels = frame->channels;
    dst->data     = frame->data;   /* take ownership */
    /* Zero caller's struct so they cannot double-free. */
    frame->width = frame->height = frame->channels = 0;
    frame->data  = NULL;
    list->count += 1;
    return OC_OK;
}


OcError oc_video_decoder_repetitive(OcVideoFrameList *out,
                                     uint32_t w, uint32_t h,
                                     const float *single_frame_data,
                                     size_t n_copies)
{
    if (out == NULL || single_frame_data == NULL ||
        w == 0 || h == 0 || n_copies == 0) {
        return OC_ERR_INVALID_ARG;
    }
    OcError e = oc_video_frame_list_init(out, n_copies);
    if (e != OC_OK) {
        return e;
    }
    for (size_t i = 0; i < n_copies; ++i) {
        e = oc_video_frame_list_add(out, w, h, single_frame_data);
        if (e != OC_OK) {
            oc_video_frame_list_free(out);
            return e;
        }
    }
    return OC_OK;
}


OcError oc_video_frame_get(const OcVideoFrameList *list, size_t idx,
                            const OcVideoFrame **out)
{
    if (list == NULL || out == NULL) {
        return OC_ERR_INVALID_ARG;
    }
    if (idx >= list->count) {
        return OC_ERR_INVALID_ARG;
    }
    *out = &list->frames[idx];
    return OC_OK;
}

size_t oc_video_frame_size_bytes(uint32_t w, uint32_t h)
{
    return frame_float_count(w, h) * sizeof(float);
}

size_t oc_video_frame_list_size_bytes(const OcVideoFrameList *list)
{
    if (list == NULL) {
        return 0;
    }
    size_t total = 0;
    for (size_t i = 0; i < list->count; ++i) {
        total += oc_video_frame_size_bytes(list->frames[i].width,
                                           list->frames[i].height);
    }
    return total;
}
