/* test_fingerprint.c — Model fingerprint tests. */
#include "framework.h"
#include "oxidize/fingerprint.h"
#include <string.h>

Test(fp, init)
{
    OcModelFingerprint fp;
    cr_assert_eq(oc_fingerprint_init(&fp), OC_OK);
    cr_assert_str_eq(fp.architecture, "unknown");
    cr_assert_eq(fp.n_layers, 0);
    cr_assert_float_eq(fp.rope_theta, 10000.0f, 0.1f);
}

Test(fp, init_null)
{
    cr_assert_neq(oc_fingerprint_init(NULL), OC_OK);
}

Test(fp, from_file_notfound)
{
    OcModelFingerprint fp;
    cr_assert_neq(oc_fingerprint_from_file("/nonexistent.gguf", &fp), OC_OK);
}

Test(fp, from_file_null)
{
    cr_assert_neq(oc_fingerprint_from_file(NULL, NULL), OC_OK);
}

Test(fp, validate_empty)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    cr_assert_neq(oc_fingerprint_validate(&fp), OC_OK);
}

Test(fp, validate_ok)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    fp.n_layers = 32;
    cr_assert_eq(oc_fingerprint_validate(&fp), OC_OK);
}

Test(fp, validate_null)
{
    cr_assert_neq(oc_fingerprint_validate(NULL), OC_OK);
}

Test(fp, is_quantized)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    cr_assert(!oc_fingerprint_is_quantized(&fp));
    fp.quant_type = 3;
    cr_assert(oc_fingerprint_is_quantized(&fp));
}

Test(fp, is_quantized_null)
{
    cr_assert(!oc_fingerprint_is_quantized(NULL));
}

Test(fp, is_moe)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    cr_assert(!oc_fingerprint_is_moe(&fp));
    fp.n_expert = 8;
    cr_assert(oc_fingerprint_is_moe(&fp));
    fp.n_expert = 1;
    cr_assert(!oc_fingerprint_is_moe(&fp));
}

Test(fp, has_gqa)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    fp.n_heads = 32;
    fp.n_kv_heads = 8;
    cr_assert(oc_fingerprint_has_gqa(&fp));
    fp.n_kv_heads = 32;
    cr_assert(!oc_fingerprint_has_gqa(&fp));
}

Test(fp, has_gqa_null)
{
    cr_assert(!oc_fingerprint_has_gqa(NULL));
}

Test(fp, model_size_gb)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    fp.file_size = (uint64_t)4 * 1024 * 1024 * 1024;
    cr_assert_float_eq(oc_fingerprint_model_size_gb(&fp), 4.0, 0.01);
}

Test(fp, model_size_null)
{
    cr_assert_eq(oc_fingerprint_model_size_gb(NULL), 0.0);
}

Test(fp, bits_per_param)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    fp.file_size = (uint64_t)4 * 1024 * 1024 * 1024;
    fp.estimated_params = 8000000000ULL; /* 8B params */
    /* 4GB * 8 bits / 8B params ≈ 4.3 bits per param */
    cr_assert_float_eq(oc_fingerprint_bits_per_param(&fp), 4.3, 0.2);
}

Test(fp, summary)
{
    OcModelFingerprint fp;
    oc_fingerprint_init(&fp);
    fp.n_layers = 32;
    fp.hidden_dim = 4096;
    fp.n_heads = 32;
    fp.n_kv_heads = 8;
    fp.vocab_size = 32000;
    fp.quant_type = 3;
    fp.file_size = (uint64_t)4 * 1024 * 1024 * 1024;
    fp.estimated_params = 7000000000ULL;
    char out[512];
    oc_fingerprint_summary(&fp, out, sizeof(out));
    cr_assert(strstr(out, "arch=unknown") != NULL);
    cr_assert(strstr(out, "layers=32") != NULL);
}

Test(fp, summary_null)
{
    char out[10];
    cr_assert_str_eq(oc_fingerprint_summary(NULL, out, sizeof(out)), "");
}
