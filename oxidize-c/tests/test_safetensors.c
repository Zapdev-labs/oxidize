#define _POSIX_C_SOURCE 200809L

#include <criterion/criterion.h>

#include "oxidize/safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


typedef struct {
    const char *name;
    const char *dtype;
    uint64_t    shape[OC_SAFETENSORS_MAX_DIMS];
    uint32_t    n_dims;
    const void *data;
    uint64_t    data_len;
} StTensorSpec;

/* Build the JSON header + concatenate tensor data, writing the complete
 * .safetensors file to `path`. Returns OC_OK on success. */
static OcError write_safetensors(const char *path,
                                 const StTensorSpec *specs, size_t n_specs)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) return OC_ERR_IO;

    /* Compute data_offsets as we append tensors sequentially. */
    uint64_t *offs = calloc(n_specs + 1, sizeof(uint64_t));
    if (offs == NULL) { fclose(fp); return OC_ERR_OOM; }
    for (size_t i = 0; i < n_specs; i++) {
        offs[i + 1] = offs[i] + specs[i].data_len;
    }

    /* Build the JSON header. */
    char *json = NULL;
    size_t json_cap = 0;
    size_t json_len = 0;
    FILE *mem = open_memstream(&json, &json_cap);
    if (mem == NULL) { free(offs); fclose(fp); return OC_ERR_OOM; }
    fputc('{', mem);
    for (size_t i = 0; i < n_specs; i++) {
        if (i > 0) fputc(',', mem);
        fprintf(mem, "\"%s\":{\"dtype\":\"%s\",\"shape\":[", specs[i].name,
                specs[i].dtype);
        for (uint32_t d = 0; d < specs[i].n_dims; d++) {
            if (d > 0) fputc(',', mem);
            fprintf(mem, "%llu", (unsigned long long)specs[i].shape[d]);
        }
        fprintf(mem, "],\"data_offsets\":[%llu,%llu]}",
                 (unsigned long long)offs[i],
                 (unsigned long long)offs[i + 1]);
    }
    fputc('}', mem);
    fclose(mem);
    json_len = strlen(json);

    /* Write 8-byte header length. */
    uint64_t hdr_len = (uint64_t)json_len;
    fwrite(&hdr_len, 1, 8, fp);
    /* Write JSON header. */
    fwrite(json, 1, json_len, fp);
    /* Write raw tensor data. */
    for (size_t i = 0; i < n_specs; i++) {
        if (specs[i].data_len > 0 && specs[i].data != NULL) {
            fwrite(specs[i].data, 1, specs[i].data_len, fp);
        }
    }
    free(json);
    free(offs);
    fclose(fp);
    return OC_OK;
}

static const char *tmp_path(const char *suffix)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oc_st_test_%d_%s", (int)getpid(),
             suffix);
    return path;
}


Test(safetensors, open_single_tensor)
{
    const char *path = tmp_path("single.safetensors");
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    StTensorSpec specs[] = {
        {"weight.0", "F32", {4}, 1, data, sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(st.n_tensors, 1);
    cr_assert_str_eq(st.tensors[0].name, "weight.0");
    cr_assert_str_eq(st.tensors[0].dtype, "F32");
    cr_assert_eq(st.tensors[0].n_dims, 1);
    cr_assert_eq(st.tensors[0].shape[0], 4);
    cr_assert_eq(st.tensors[0].data_length, sizeof(data));
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, open_multiple_tensors)
{
    const char *path = tmp_path("multi.safetensors");
    float a[2] = {1.0f, 2.0f};
    int32_t b[3] = {10, 20, 30};
    StTensorSpec specs[] = {
        {"tensor.a", "F32", {2}, 1, a, sizeof(a)},
        {"tensor.b", "I32", {3}, 1, b, sizeof(b)},
    };
    cr_assert_eq(write_safetensors(path, specs, 2), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(st.n_tensors, 2);
    cr_assert_str_eq(st.tensors[0].name, "tensor.a");
    cr_assert_str_eq(st.tensors[1].name, "tensor.b");
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, open_2d_shape)
{
    const char *path = tmp_path("2d.safetensors");
    float data[6] = {0}; /* 2x3 */
    StTensorSpec specs[] = {
        {"mat", "F32", {2, 3}, 2, data, sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(st.tensors[0].n_dims, 2);
    cr_assert_eq(st.tensors[0].shape[0], 2);
    cr_assert_eq(st.tensors[0].shape[1], 3);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, get_tensor_found)
{
    const char *path = tmp_path("getfound.safetensors");
    float data[4] = {0};
    StTensorSpec specs[] = {
        {"alpha", "F32", {4}, 1, data, sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    const OcSafetensorsTensor *t = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(&st, "alpha", &t), OC_OK);
    cr_assert_not_null(t);
    cr_assert_str_eq(t->name, "alpha");
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, get_tensor_not_found)
{
    const char *path = tmp_path("getnotfound.safetensors");
    float data[4] = {0};
    StTensorSpec specs[] = {
        {"alpha", "F32", {4}, 1, data, sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    const OcSafetensorsTensor *t = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(&st, "missing", &t),
                 OC_ERR_TENSOR);
    cr_assert_null(t);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, get_tensor_data_content)
{
    const char *path = tmp_path("datacontent.safetensors");
    float data[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    StTensorSpec specs[] = {
        {"w", "F32", {4}, 1, data, sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    const OcSafetensorsTensor *t = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(&st, "w", &t), OC_OK);
    const void *raw = NULL;
    cr_assert_eq(oc_safetensors_get_tensor_data(&st, t, &raw), OC_OK);
    cr_assert_not_null(raw);
    float fraw[4];
    memcpy(fraw, raw, sizeof(fraw));
    cr_assert_float_eq(fraw[0], 1.5f, 1e-6);
    cr_assert_float_eq(fraw[1], 2.5f, 1e-6);
    cr_assert_float_eq(fraw[2], 3.5f, 1e-6);
    cr_assert_float_eq(fraw[3], 4.5f, 1e-6);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, get_tensor_data_offsets)
{
    const char *path = tmp_path("offsets.safetensors");
    float a[2] = {1.0f, 2.0f};
    float b[3] = {3.0f, 4.0f, 5.0f};
    StTensorSpec specs[] = {
        {"a", "F32", {2}, 1, a, sizeof(a)},
        {"b", "F32", {3}, 1, b, sizeof(b)},
    };
    cr_assert_eq(write_safetensors(path, specs, 2), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    /* Tensor b should start at offset sizeof(a) = 8 bytes. */
    const OcSafetensorsTensor *tb = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(&st, "b", &tb), OC_OK);
    cr_assert_eq(tb->data_offset, sizeof(a));
    cr_assert_eq(tb->data_length, sizeof(b));
    const void *raw = NULL;
    cr_assert_eq(oc_safetensors_get_tensor_data(&st, tb, &raw), OC_OK);
    float fraw[1];
    memcpy(fraw, raw, sizeof(fraw));
    cr_assert_float_eq(fraw[0], 3.0f, 1e-6);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, n_tensors)
{
    const char *path = tmp_path("ntensors.safetensors");
    float a[1] = {0};
    StTensorSpec specs[] = {
        {"t0", "F32", {1}, 1, a, sizeof(a)},
        {"t1", "F32", {1}, 1, a, sizeof(a)},
        {"t2", "F32", {1}, 1, a, sizeof(a)},
    };
    cr_assert_eq(write_safetensors(path, specs, 3), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(oc_safetensors_n_tensors(&st), 3);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, n_tensors_null)
{
    cr_assert_eq(oc_safetensors_n_tensors(NULL), 0);
}

Test(safetensors, open_null_args)
{
    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(NULL, &st), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_safetensors_open("foo", NULL), OC_ERR_INVALID_ARG);
}

Test(safetensors, open_missing_file)
{
    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open("/tmp/oc_st_does_not_exist_xyz.safetensors",
                                     &st),
                 OC_ERR_IO);
}

Test(safetensors, open_truncated_file)
{
    const char *path = tmp_path("trunc.safetensors");
    FILE *fp = fopen(path, "wb");
    cr_assert_not_null(fp);
    /* Only 4 bytes — less than the 8-byte header length. */
    uint32_t bogus = 0xDEADBEEFu;
    fwrite(&bogus, 1, 4, fp);
    fclose(fp);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_ERR_FORMAT);
    unlink(path);
}

Test(safetensors, open_bad_header_length)
{
    const char *path = tmp_path("badhdr.safetensors");
    FILE *fp = fopen(path, "wb");
    cr_assert_not_null(fp);
    /* Header length larger than the file. */
    uint64_t hdr_len = 999999ull;
    fwrite(&hdr_len, 1, 8, fp);
    fclose(fp);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_ERR_FORMAT);
    unlink(path);
}

Test(safetensors, get_tensor_null_args)
{
    const OcSafetensorsTensor *t = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(NULL, "x", &t), OC_ERR_INVALID_ARG);
    OcSafetensorsFile st = {0};
    cr_assert_eq(oc_safetensors_get_tensor(&st, "x", &t), OC_ERR_TENSOR);
    cr_assert_eq(oc_safetensors_get_tensor(&st, NULL, &t), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_safetensors_get_tensor(&st, "x", NULL), OC_ERR_INVALID_ARG);
}

Test(safetensors, get_tensor_data_null_args)
{
    const void *raw = NULL;
    cr_assert_eq(oc_safetensors_get_tensor_data(NULL, NULL, &raw),
                 OC_ERR_INVALID_ARG);
    OcSafetensorsFile st = {0};
    OcSafetensorsTensor t = {0};
    cr_assert_eq(oc_safetensors_get_tensor_data(&st, &t, &raw),
                 OC_ERR_FORMAT);
}

Test(safetensors, close_idempotent)
{
    OcSafetensorsFile st = {0};
    oc_safetensors_close(&st);
    oc_safetensors_close(NULL);
    /* Should not crash; st zeroed. */
    cr_assert_eq(st.n_tensors, 0);
}

Test(safetensors, metadata_key_skipped)
{
    const char *path = tmp_path("meta.safetensors");
    float data[2] = {1.0f, 2.0f};
    /* Manually craft a file with a __metadata__ key. */
    FILE *fp = fopen(path, "wb");
    cr_assert_not_null(fp);
    const char *json =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    uint64_t hdr_len = (uint64_t)strlen(json);
    fwrite(&hdr_len, 1, 8, fp);
    fwrite(json, 1, hdr_len, fp);
    fwrite(data, 1, sizeof(data), fp);
    fclose(fp);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    /* __metadata__ must not be counted as a tensor. */
    cr_assert_eq(st.n_tensors, 1);
    cr_assert_str_eq(st.tensors[0].name, "w");
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, empty_data_tensor)
{
    const char *path = tmp_path("empty.safetensors");
    StTensorSpec specs[] = {
        {"empty", "F32", {0}, 1, NULL, 0},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_eq(st.n_tensors, 1);
    cr_assert_eq(st.tensors[0].data_length, 0);
    const void *raw = NULL;
    const OcSafetensorsTensor *t = NULL;
    cr_assert_eq(oc_safetensors_get_tensor(&st, "empty", &t), OC_OK);
    cr_assert_eq(oc_safetensors_get_tensor_data(&st, t, &raw), OC_OK);
    oc_safetensors_close(&st);
    unlink(path);
}

Test(safetensors, special_chars_in_name)
{
    const char *path = tmp_path("special.safetensors");
    float data[1] = {0};
    StTensorSpec specs[] = {
        {"model.layers.0.self_attn.q_proj.weight", "F32", {1}, 1, data,
         sizeof(data)},
    };
    cr_assert_eq(write_safetensors(path, specs, 1), OC_OK);

    OcSafetensorsFile st;
    cr_assert_eq(oc_safetensors_open(path, &st), OC_OK);
    cr_assert_str_eq(st.tensors[0].name,
                     "model.layers.0.self_attn.q_proj.weight");
    oc_safetensors_close(&st);
    unlink(path);
}
