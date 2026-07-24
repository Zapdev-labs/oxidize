/* test_conversion.c — Conversion module tests. */
#include <criterion/criterion.h>
#include "oxidize/conversion.h"
#include <string.h>

Test(conv, config_init)
{
    OcConvConfig cfg;
    cr_assert_eq(oc_conv_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.target, OC_CONV_Q_Q4_K_M);
    cr_assert(!cfg.verbose);
    cr_assert(cfg.copy_metadata);
}

Test(conv, config_init_null)
{
    cr_assert_neq(oc_conv_config_init(NULL), OC_OK);
}

Test(conv, run_null)
{
    cr_assert_neq(oc_conv_run(NULL, NULL), OC_OK);
}

Test(conv, run_no_input)
{
    OcConvConfig cfg;
    oc_conv_config_init(&cfg);
    cfg.input_path = NULL;
    cfg.output_path = "out.gguf";
    cr_assert_neq(oc_conv_run(&cfg, NULL), OC_OK);
}

Test(conv, run_nonexistent_file)
{
    OcConvConfig cfg;
    oc_conv_config_init(&cfg);
    cfg.input_path = "/tmp/nonexistent_input.safetensors";
    cfg.output_path = "/tmp/output.gguf";
    OcConvResult result;
    /* Real conversion should fail on non-existent input file. */
    cr_assert_neq(oc_conv_run(&cfg, &result), OC_OK);
}

Test(conv, quant_from_str)
{
    OcConvQuantType q;
    cr_assert_eq(oc_conv_quant_type_from_str("F32", &q), OC_OK);
    cr_assert_eq(q, OC_CONV_Q_F32);
    cr_assert_eq(oc_conv_quant_type_from_str("Q4_K_M", &q), OC_OK);
    cr_assert_eq(q, OC_CONV_Q_Q4_K_M);
    cr_assert_eq(oc_conv_quant_type_from_str("Q8_0", &q), OC_OK);
    cr_assert_eq(q, OC_CONV_Q_Q8_0);
}

Test(conv, quant_from_str_invalid)
{
    OcConvQuantType q;
    cr_assert_neq(oc_conv_quant_type_from_str("INVALID", &q), OC_OK);
}

Test(conv, quant_from_str_null)
{
    cr_assert_neq(oc_conv_quant_type_from_str(NULL, NULL), OC_OK);
}

Test(conv, quant_name)
{
    cr_assert_str_eq(oc_conv_quant_type_name(OC_CONV_Q_F32), "F32");
    cr_assert_str_eq(oc_conv_quant_type_name(OC_CONV_Q_Q4_K_M), "Q4_K_M");
    cr_assert_str_eq(oc_conv_quant_type_name(OC_CONV_Q_Q8_0), "Q8_0");
    cr_assert_str_eq(oc_conv_quant_type_name(OC_CONV_Q_BF16), "BF16");
}

Test(conv, is_valid)
{
    cr_assert(oc_conv_is_valid_quant_type(OC_CONV_Q_F32));
    cr_assert(oc_conv_is_valid_quant_type(OC_CONV_Q_Q4_K_M));
    cr_assert(!oc_conv_is_valid_quant_type((OcConvQuantType)99));
}

Test(conv, bits_per_weight)
{
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_F32), 32);
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_F16), 16);
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_Q8_0), 8);
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_Q4_0), 4);
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_Q5_K), 5);
    cr_assert_eq(oc_conv_bits_per_weight(OC_CONV_Q_Q6_K), 6);
}

Test(conv, gguf_name)
{
    cr_assert_str_eq(oc_conv_quant_type_gguf_name(OC_CONV_Q_F32), "F32");
    cr_assert_str_eq(oc_conv_quant_type_gguf_name(OC_CONV_Q_Q4_K), "Q4_K");
}
