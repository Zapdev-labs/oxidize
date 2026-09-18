#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>

#include "oxidize/error.h"
#include "oxidize/prerouter.h"
#include "oxidize/safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *name;
    const char *dtype;
    uint64_t    shape[2];
    uint32_t    n_dims;
    const void *data;
    uint64_t    data_len;
} StTensorSpec;

static OcError write_safetensors(const char *path,
                                 const StTensorSpec *specs, size_t n_specs)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) return OC_ERR_IO;
    uint64_t *offs = calloc(n_specs + 1, sizeof(uint64_t));
    if (offs == NULL) { fclose(fp); return OC_ERR_OOM; }
    for (size_t i = 0; i < n_specs; i++)
        offs[i + 1] = offs[i] + specs[i].data_len;

    char *json = NULL;
    size_t json_cap = 0;
    FILE *mem = open_memstream(&json, &json_cap);
    if (mem == NULL) { free(offs); fclose(fp); return OC_ERR_OOM; }
    fputc('{', mem);
    for (size_t i = 0; i < n_specs; i++) {
        if (i > 0) fputc(',', mem);
        fprintf(mem, "\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],"
                     "\"data_offsets\":[%llu,%llu]}",
                specs[i].name, specs[i].dtype,
                (unsigned long long)specs[i].shape[0],
                (unsigned long long)specs[i].shape[1],
                (unsigned long long)offs[i],
                (unsigned long long)offs[i + 1]);
    }
    fputc('}', mem);
    fclose(mem);

    uint64_t hdr_len = (uint64_t)strlen(json);
    fwrite(&hdr_len, 1, 8, fp);
    fwrite(json, 1, (size_t)hdr_len, fp);
    for (size_t i = 0; i < n_specs; i++) {
        if (specs[i].data && specs[i].data_len)
            fwrite(specs[i].data, 1, (size_t)specs[i].data_len, fp);
    }
    free(json);
    free(offs);
    fclose(fp);
    return OC_OK;
}

static const char *tmp_path(void)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oc_prerouter_%d.safetensors",
             (int)getpid());
    return path;
}

Test(prerouter, commit_then_consume)
{
    const uint32_t hidden = 4;
    const uint32_t n_exp = 4;
    const uint32_t k = 2;
    const uint32_t width = 8;
    const uint32_t input = hidden + 2u * n_exp;
    float *fc1 = calloc((size_t)width * input, sizeof(float));
    float *fc2 = calloc((size_t)n_exp * width, sizeof(float));
    float *lin = calloc((size_t)n_exp * input, sizeof(float));
    cr_assert(fc1 && fc2 && lin);
    /* Bias the first two experts via linear_init on the hidden channels. */
    for (uint32_t e = 0; e < 2; e++)
        lin[e * input + 0] = 8.0f;

    const char *path = tmp_path();
    StTensorSpec specs[] = {
        {"layers.0.fc1.weight", "F32", {width, input}, 2, fc1,
         (uint64_t)width * input * sizeof(float)},
        {"layers.0.fc2.weight", "F32", {n_exp, width}, 2, fc2,
         (uint64_t)n_exp * width * sizeof(float)},
        {"layers.0.linear_init.weight", "F32", {n_exp, input}, 2, lin,
         (uint64_t)n_exp * input * sizeof(float)},
    };
    cr_assert_eq(write_safetensors(path, specs, 3), OC_OK);

    OcPrerouter *p = NULL;
    cr_assert_eq(oc_prerouter_new(2, n_exp, hidden, k, &p), OC_OK);
    cr_assert_eq(oc_prerouter_load_safetensors(p, path), OC_OK);
    cr_assert_eq(oc_prerouter_n_heads(p), 1);
    cr_assert(oc_prerouter_replace_routing(p));

    float hidden_v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    uint32_t sel[2] = {1, 3};
    cr_assert_eq(oc_prerouter_commit(p, 0, hidden_v, sel, k), OC_OK);
    cr_assert(oc_prerouter_has_prediction(p, 1));
    cr_assert_not(oc_prerouter_has_prediction(p, 0));

    uint32_t got[2] = {0, 0};
    float w[4];
    cr_assert_eq(oc_prerouter_consume(p, 1, got, k, w), OC_OK);
    cr_assert_not(oc_prerouter_has_prediction(p, 1));
    cr_assert(got[0] < n_exp);
    cr_assert(got[1] < n_exp);
    cr_assert_neq(got[0], got[1]);
    float sum = w[got[0]] + w[got[1]];
    cr_assert_float_eq(sum, 1.0f, 1e-5f);

    oc_prerouter_free(p);
    unlink(path);
    free(fc1); free(fc2); free(lin);
}

Test(prerouter, rejects_empty_file)
{
    OcPrerouter *p = NULL;
    cr_assert_eq(oc_prerouter_new(2, 4, 4, 2, &p), OC_OK);
    cr_assert_eq(oc_prerouter_load_safetensors(p, "/tmp/oc_prerouter_missing"),
                 OC_ERR_IO);
    oc_prerouter_free(p);
}

Test(prerouter, reset_clears_prediction)
{
    OcPrerouter *p = NULL;
    cr_assert_eq(oc_prerouter_new(2, 4, 4, 2, &p), OC_OK);
    cr_assert_not(oc_prerouter_has_prediction(p, 1));
    oc_prerouter_reset(p);
    cr_assert_not(oc_prerouter_has_prediction(p, 0));
    oc_prerouter_free(p);
}
