#define _POSIX_C_SOURCE 200809L
#include "oxidize/seq_kv.h"

#include <stdlib.h>
#include <string.h>

OcError oc_seq_kv_init(OcSeqKv *kv, size_t kv_layer_count,
                       size_t capacity_tokens, size_t kv_len)
{
    if (!kv || kv_layer_count == 0 || capacity_tokens == 0 || kv_len == 0)
        return OC_ERR_INVALID_ARG;

    size_t elems = kv_layer_count * capacity_tokens * kv_len;
    kv->key = calloc(elems, sizeof(float));
    kv->value = calloc(elems, sizeof(float));
    if (!kv->key || !kv->value) {
        free(kv->key);
        free(kv->value);
        memset(kv, 0, sizeof(*kv));
        return OC_ERR_OOM;
    }
    kv->len = 0;
    kv->capacity_tokens = capacity_tokens;
    kv->kv_layer_count = kv_layer_count;
    kv->kv_len = kv_len;
    return OC_OK;
}

void oc_seq_kv_free(OcSeqKv *kv)
{
    if (!kv) return;
    free(kv->key);
    free(kv->value);
    memset(kv, 0, sizeof(*kv));
}

OcError oc_seq_kv_get(const OcSeqKv *kv, size_t layer, size_t pos,
                       const float **out_k, const float **out_v)
{
    if (!kv || !out_k || !out_v) return OC_ERR_INVALID_ARG;
    if (layer >= kv->kv_layer_count) return OC_ERR_INVALID_ARG;
    if (pos >= kv->capacity_tokens) return OC_ERR_INVALID_ARG;

    size_t offset = (layer * kv->capacity_tokens + pos) * kv->kv_len;
    *out_k = &kv->key[offset];
    *out_v = &kv->value[offset];
    return OC_OK;
}

OcError oc_seq_kv_put(OcSeqKv *kv, size_t layer, size_t pos,
                      const float *k, const float *v)
{
    if (!kv || !k || !v) return OC_ERR_INVALID_ARG;
    if (layer >= kv->kv_layer_count) return OC_ERR_INVALID_ARG;
    if (pos >= kv->capacity_tokens) return OC_ERR_INVALID_ARG;

    size_t offset = (layer * kv->capacity_tokens + pos) * kv->kv_len;
    memcpy(&kv->key[offset], k, kv->kv_len * sizeof(float));
    memcpy(&kv->value[offset], v, kv->kv_len * sizeof(float));
    return OC_OK;
}

OcError oc_seq_kv_advance(OcSeqKv *kv, size_t n)
{
    if (!kv) return OC_ERR_INVALID_ARG;
    kv->len += n;
    if (kv->len > kv->capacity_tokens)
        kv->len = kv->capacity_tokens;
    return OC_OK;
}

OcError oc_seq_kv_truncate(OcSeqKv *kv, size_t n)
{
    if (!kv) return OC_ERR_INVALID_ARG;
    if (n > kv->capacity_tokens) return OC_ERR_INVALID_ARG;
    if (n < kv->len) kv->len = n;
    return OC_OK;
}

void oc_seq_kv_clear(OcSeqKv *kv)
{
    if (!kv) return;
    size_t elems = kv->kv_layer_count * kv->capacity_tokens * kv->kv_len;
    if (kv->key) memset(kv->key, 0, elems * sizeof(float));
    if (kv->value) memset(kv->value, 0, elems * sizeof(float));
    kv->len = 0;
}

size_t oc_seq_kv_size_bytes(const OcSeqKv *kv)
{
    if (!kv) return 0;
    return kv->kv_layer_count * kv->capacity_tokens * kv->kv_len * sizeof(float) * 2;
}

bool oc_seq_kv_is_empty(const OcSeqKv *kv)
{
    return kv ? (kv->len == 0) : true;
}

size_t oc_seq_kv_capacity(const OcSeqKv *kv)
{
    return kv ? kv->capacity_tokens : 0;
}
