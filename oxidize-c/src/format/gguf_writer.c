/*
 * gguf_writer.c — GGUF v3 file writer implementation.
 *
 * See gguf_writer.h for the public API and on-disk layout.
 *
 * All multi-byte integers are written in little-endian byte order, matching
 * the GGUF spec (and the host is assumed LE, consistent with the parser).
 *
 * Tensor data layout: all tensor info entries are written first (streamed
 * during add_tensor), then at finalize() the data section is aligned to a
 * 32-byte boundary and all buffered tensor data is written. Each tensor
 * info's offset field is patched at finalize time once the data section
 * start is known.
 */

#define _POSIX_C_SOURCE 200809L

#include "oxidize/gguf_writer.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Constants ─────────────────────────────────────────────────────────── */

#define GW_ALIGNMENT 32u

/* ─── Low-level LE write helpers ────────────────────────────────────────── */

static void gw_u32(FILE *fp, uint32_t v)
{
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    fwrite(b, 1, 4, fp);
}

static void gw_u64(FILE *fp, uint64_t v)
{
    uint8_t b[8];
    for (int i = 0; i < 8; i++) {
        b[i] = (uint8_t)((v >> (i * 8)) & 0xFF);
    }
    fwrite(b, 1, 8, fp);
}

static void gw_f32(FILE *fp, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    gw_u32(fp, bits);
}

static void gw_string(FILE *fp, const char *s)
{
    uint64_t len = s ? strlen(s) : 0;
    gw_u64(fp, len);
    if (len > 0) {
        fwrite(s, 1, (size_t)len, fp);
    }
}

static void gw_pad(FILE *fp, uint32_t alignment)
{
    long pos = ftell(fp);
    if (pos < 0) return;
    uint64_t rem = (uint64_t)pos % alignment;
    if (rem > 0) {
        uint64_t pad = alignment - rem;
        static const uint8_t zeros[32] = {0};
        while (pad > 0) {
            uint64_t chunk = pad < sizeof(zeros) ? pad : sizeof(zeros);
            fwrite(zeros, 1, (size_t)chunk, fp);
            pad -= chunk;
        }
    }
}

/* ─── Buffered tensor data ──────────────────────────────────────────────── */

/* Each pending tensor's data is buffered in memory; at finalize() we write
 * the aligned data section and patch each tensor info's offset field. */
typedef struct GwPendingTensor {
    long  offset_field_pos;  /* file position of the u64 offset field to patch */
    void *data;              /* heap-owned tensor data */
    uint64_t data_size;      /* byte length of data */
} GwPendingTensor;

typedef struct GwPendingList {
    GwPendingTensor *items;
    size_t count;
    size_t cap;
} GwPendingList;

static OcError gw_pending_push(GwPendingList *pl, long offset_pos,
                               void *data, uint64_t data_size)
{
    if (pl->count >= pl->cap) {
        size_t ncap = pl->cap > 0 ? pl->cap * 2 : 8;
        GwPendingTensor *ni = realloc(pl->items, ncap * sizeof(GwPendingTensor));
        if (!ni) return OC_ERR_OOM;
        pl->items = ni;
        pl->cap = ncap;
    }
    pl->items[pl->count].offset_field_pos = offset_pos;
    pl->items[pl->count].data = data;
    pl->items[pl->count].data_size = data_size;
    pl->count++;
    return OC_OK;
}

static void gw_pending_free(GwPendingList *pl)
{
    for (size_t i = 0; i < pl->count; i++) {
        free(pl->items[i].data);
    }
    free(pl->items);
    pl->items = NULL;
    pl->count = 0;
    pl->cap = 0;
}

/* ─── Init / free ──────────────────────────────────────────────────────── */

OcError oc_gguf_writer_init(const char *path, const char *arch_name,
                            OcGgufWriter *w)
{
    if (!w) return OC_ERR_INVALID_ARG;
    memset(w, 0, sizeof(*w));

    if (!path) return OC_ERR_INVALID_ARG;

    w->fp = fopen(path, "wb");
    if (!w->fp) return OC_ERR_IO;
    w->owns_fp = true;

    /* Write header: magic, version, tensor_count=0, metadata_count=0. */
    gw_u32(w->fp, OC_GGUF_MAGIC);
    gw_u32(w->fp, OC_GGUF_VERSION_3);
    w->tensor_count_off = ftell(w->fp);
    gw_u64(w->fp, 0);
    w->metadata_count_off = ftell(w->fp);
    gw_u64(w->fp, 0);

    w->pending = NULL;  /* lazy-allocated on first add_tensor */
    w->finalized = false;

    /* Auto-add general.architecture metadata if arch_name is provided. */
    if (arch_name) {
        OcError e = oc_gguf_writer_add_string(w, "general.architecture", arch_name);
        if (e != OC_OK) {
            oc_gguf_writer_free(w);
            return e;
        }
    }

    return OC_OK;
}

/* ─── Metadata writers ─────────────────────────────────────────────────── */

OcError oc_gguf_writer_add_string(OcGgufWriter *w, const char *key,
                                  const char *value)
{
    if (!w || !w->fp || w->finalized) return OC_ERR_INVALID_ARG;
    if (!key || !value) return OC_ERR_INVALID_ARG;
    if (w->tensor_count > 0) return OC_ERR_INVALID_ARG; /* metadata before tensors */

    gw_string(w->fp, key);
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_STRING);
    gw_string(w->fp, value);
    w->metadata_count++;
    return OC_OK;
}

OcError oc_gguf_writer_add_uint32(OcGgufWriter *w, const char *key,
                                  uint32_t value)
{
    if (!w || !w->fp || w->finalized || !key) return OC_ERR_INVALID_ARG;
    if (w->tensor_count > 0) return OC_ERR_INVALID_ARG;

    gw_string(w->fp, key);
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_UINT32);
    gw_u32(w->fp, value);
    w->metadata_count++;
    return OC_OK;
}

OcError oc_gguf_writer_add_uint64(OcGgufWriter *w, const char *key,
                                  uint64_t value)
{
    if (!w || !w->fp || w->finalized || !key) return OC_ERR_INVALID_ARG;
    if (w->tensor_count > 0) return OC_ERR_INVALID_ARG;

    gw_string(w->fp, key);
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_UINT64);
    gw_u64(w->fp, value);
    w->metadata_count++;
    return OC_OK;
}

OcError oc_gguf_writer_add_float32(OcGgufWriter *w, const char *key,
                                   float value)
{
    if (!w || !w->fp || w->finalized || !key) return OC_ERR_INVALID_ARG;
    if (w->tensor_count > 0) return OC_ERR_INVALID_ARG;

    gw_string(w->fp, key);
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_FLOAT32);
    gw_f32(w->fp, value);
    w->metadata_count++;
    return OC_OK;
}

OcError oc_gguf_writer_add_array_string(OcGgufWriter *w, const char *key,
                                        const char *const *values, size_t count)
{
    if (!w || !w->fp || w->finalized || !key) return OC_ERR_INVALID_ARG;
    if (w->tensor_count > 0) return OC_ERR_INVALID_ARG;
    if (count > 0 && !values) return OC_ERR_INVALID_ARG;

    gw_string(w->fp, key);
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_ARRAY);   /* ARRAY type */
    gw_u32(w->fp, (uint32_t)OC_GGUF_MT_STRING);   /* element type: STRING */
    gw_u64(w->fp, (uint64_t)count);
    for (size_t i = 0; i < count; i++) {
        const char *s = values[i] ? values[i] : "";
        gw_string(w->fp, s);
    }
    w->metadata_count++;
    return OC_OK;
}

/* ─── Tensor writer ────────────────────────────────────────────────────── */

OcError oc_gguf_writer_add_tensor(OcGgufWriter *w, const char *name,
                                  uint32_t n_dims, const uint64_t *dims,
                                  uint32_t type, const void *data,
                                  uint64_t data_size)
{
    if (!w || !w->fp || w->finalized || !name || !dims)
        return OC_ERR_INVALID_ARG;
    if (n_dims == 0 || n_dims > OC_GGUF_WRITER_MAX_DIMS)
        return OC_ERR_INVALID_ARG;
    if (data_size > 0 && !data)
        return OC_ERR_INVALID_ARG;

    /* Lazy-allocate the pending list. */
    if (!w->pending) {
        w->pending = calloc(1, sizeof(GwPendingList));
        if (!w->pending) return OC_ERR_OOM;
    }

    /* Write the tensor info entry. The offset field is a placeholder —
     * we'll patch it at finalize() once the data section start is known. */
    gw_string(w->fp, name);
    gw_u32(w->fp, n_dims);
    for (uint32_t i = 0; i < n_dims; i++) {
        gw_u64(w->fp, dims[i]);
    }
    gw_u32(w->fp, type);

    /* Record the file position of the offset field for later patching. */
    long offset_pos = ftell(w->fp);
    gw_u64(w->fp, 0);  /* placeholder offset */

    /* Buffer the tensor data (copy it so the caller can free/reuse). */
    void *data_copy = NULL;
    if (data_size > 0) {
        data_copy = malloc((size_t)data_size);
        if (!data_copy) return OC_ERR_OOM;
        memcpy(data_copy, data, (size_t)data_size);
    }

    OcError e = gw_pending_push((GwPendingList *)w->pending, offset_pos,
                                data_copy, data_size);
    if (e != OC_OK) {
        free(data_copy);
        return e;
    }

    w->tensor_count++;
    return OC_OK;
}

/* ─── Finalize ──────────────────────────────────────────────────────────── */

OcError oc_gguf_writer_finalize(OcGgufWriter *w)
{
    if (!w || !w->fp) return OC_ERR_INVALID_ARG;
    if (w->finalized) return OC_OK;

    GwPendingList *pl = (GwPendingList *)w->pending;

    /* The data section starts at the current file position (after all tensor
     * infos). Align to 32-byte boundary. */
    gw_pad(w->fp, GW_ALIGNMENT);
    uint64_t data_section_start = (uint64_t)ftell(w->fp);

    /* Patch each tensor info's offset field and write the tensor data. */
    uint64_t running_offset = 0;
    if (pl) {
        for (size_t i = 0; i < pl->count; i++) {
            /* Patch the offset (relative to data_section_start). */
            long save = ftell(w->fp);
            fseek(w->fp, pl->items[i].offset_field_pos, SEEK_SET);
            gw_u64(w->fp, running_offset);
            fseek(w->fp, save, SEEK_SET);

            /* Write the tensor data at the current position. */
            if (pl->items[i].data_size > 0) {
                fwrite(pl->items[i].data, 1, (size_t)pl->items[i].data_size, w->fp);
            }
            running_offset += pl->items[i].data_size;
        }
    }

    /* Patch tensor_count and metadata_count in the header. */
    long end = ftell(w->fp);
    fseek(w->fp, w->tensor_count_off, SEEK_SET);
    gw_u64(w->fp, w->tensor_count);
    fseek(w->fp, w->metadata_count_off, SEEK_SET);
    gw_u64(w->fp, w->metadata_count);
    fseek(w->fp, end, SEEK_SET);
    fflush(w->fp);

    w->data_section_start = data_section_start;
    w->finalized = true;
    return OC_OK;
}

void oc_gguf_writer_free(OcGgufWriter *w)
{
    if (!w) return;
    if (w->fp) {
        if (!w->finalized) {
            oc_gguf_writer_finalize(w);
        }
        if (w->owns_fp) {
            fclose(w->fp);
        }
    }
    if (w->pending) {
        gw_pending_free((GwPendingList *)w->pending);
        free(w->pending);
    }
    memset(w, 0, sizeof(*w));
}
