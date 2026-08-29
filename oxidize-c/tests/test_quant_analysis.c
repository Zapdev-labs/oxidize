/* test_quant_analysis.c — quantization analysis tests. */
#include "framework.h"
#include "oxidize/quant_analysis.h"
#include <string.h>

Test(qa, type_name)
{
    cr_assert_str_eq(oc_quant_analysis_type_name(OC_QUANT_F32), "F32");
    cr_assert_str_eq(oc_quant_analysis_type_name(OC_QUANT_Q4_K_M), "Q4_K_M");
    cr_assert_str_eq(oc_quant_analysis_type_name(OC_QUANT_Q8_0), "Q8_0");
}

Test(qa, bits_per_element)
{
    cr_assert_float_eq(oc_quant_bits_per_element(OC_QUANT_F32), 32.0f, 1e-6f);
    cr_assert_float_eq(oc_quant_bits_per_element(OC_QUANT_F16), 16.0f, 1e-6f);
    cr_assert(oc_quant_bits_per_element(OC_QUANT_Q4_0) < 5.0f);
    cr_assert(oc_quant_bits_per_element(OC_QUANT_Q4_0) > 4.0f);
}

Test(qa, ppl_delta)
{
    cr_assert_float_eq(oc_quant_estimated_ppl_delta(OC_QUANT_F16), 0.0, 1e-6);
    cr_assert(oc_quant_estimated_ppl_delta(OC_QUANT_Q4_K_M) > 0.0);
    cr_assert(oc_quant_estimated_ppl_delta(OC_QUANT_Q2_K) > oc_quant_estimated_ppl_delta(OC_QUANT_Q4_K_M));
}

Test(qa, recommend_speed)
{
    OcQuantRecommendation r;
    cr_assert_eq(oc_quant_recommend(7000000000ULL, 16ULL * 1024 * 1024 * 1024,
                                     OC_QUANT_GOAL_SPEED, &r), OC_OK);
    cr_assert(r.recommended != OC_QUANT_F32);
    cr_assert(strlen(r.rationale) > 0);
}

Test(qa, recommend_quality)
{
    OcQuantRecommendation r;
    cr_assert_eq(oc_quant_recommend(7000000000ULL, 32ULL * 1024 * 1024 * 1024,
                                     OC_QUANT_GOAL_QUALITY, &r), OC_OK);
    cr_assert(r.estimated_ppl_delta < 0.1);
}

Test(qa, recommend_memory)
{
    OcQuantRecommendation r;
    cr_assert_eq(oc_quant_recommend(7000000000ULL, 4ULL * 1024 * 1024 * 1024,
                                     OC_QUANT_GOAL_MEMORY, &r), OC_OK);
    cr_assert(r.estimated_size_gb < 5.0);
}

Test(qa, recommend_null)
{
    cr_assert_neq(oc_quant_recommend(0, 0, OC_QUANT_GOAL_SPEED, NULL), OC_OK);
}

Test(qa, metrics_format_json)
{
    OcQuantMetrics m = {0};
    m.mse = 0.001;
    m.cos_sim = 0.999;
    m.n_elements = 1000;
    m.original_bytes = 4000;
    m.quantized_bytes = 600;
    m.compression_ratio = 6.67;
    char buf[512];
    size_t n = oc_quant_metrics_format(&m, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "mse") != NULL);
    cr_assert(strstr(buf, "cos_sim") != NULL);
}

Test(qa, metrics_table)
{
    OcQuantMetrics m = {0};
    m.mse = 0.001;
    m.cos_sim = 0.999;
    m.n_elements = 1000;
    char buf[1024];
    size_t n = oc_quant_metrics_table(&m, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "Quantization") != NULL);
}

Test(qa, recommend_format)
{
    OcQuantRecommendation r = {0};
    r.recommended = OC_QUANT_Q4_K_M;
    r.alternative = OC_QUANT_Q5_K_M;
    r.estimated_ppl_delta = 0.15;
    r.estimated_size_gb = 4.2;
    r.estimated_tok_per_sec = 30.0;
    strcpy(r.rationale, "test rationale");
    char buf[512];
    size_t n = oc_quant_recommend_format(&r, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "Q4_K") != NULL);
    cr_assert(strstr(buf, "Q5_K") != NULL);
}

Test(qa, comparison_table)
{
    char buf[2048];
    size_t n = oc_quant_comparison_table(7000000000ULL, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "Type") != NULL);
    cr_assert(strstr(buf, "F32") != NULL);
    cr_assert(strstr(buf, "Q4_K") != NULL);
}
