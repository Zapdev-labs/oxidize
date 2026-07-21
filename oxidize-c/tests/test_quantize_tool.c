/* test_quantize_tool.c — offline quantization tool tests. */
#include <criterion/criterion.h>
#include "oxidize/quantize_tool.h"
#include "oxidize/quant.h"

Test(quantize_tool, parse_type_q4_0)
{
    OcGgufQuantizationType t;
    cr_assert_eq(oc_quantize_parse_type("Q4_0", &t), OC_OK);
    cr_assert_eq(t, OC_QUANT_Q4_0);
}

Test(quantize_tool, parse_type_q4_k_m)
{
    OcGgufQuantizationType t;
    cr_assert_eq(oc_quantize_parse_type("Q4_K_M", &t), OC_OK);
    cr_assert_eq(t, OC_QUANT_Q4_K_M);
}

Test(quantize_tool, parse_type_q8_0)
{
    OcGgufQuantizationType t;
    cr_assert_eq(oc_quantize_parse_type("Q8_0", &t), OC_OK);
    cr_assert_eq(t, OC_QUANT_Q8_0);
}

Test(quantize_tool, parse_type_f16)
{
    OcGgufQuantizationType t;
    cr_assert_eq(oc_quantize_parse_type("F16", &t), OC_OK);
    cr_assert_eq(t, OC_QUANT_F16);
}

Test(quantize_tool, parse_type_invalid)
{
    OcGgufQuantizationType t;
    cr_assert_neq(oc_quantize_parse_type("INVALID_TYPE", &t), OC_OK);
}

Test(quantize_tool, parse_type_null)
{
    cr_assert_eq(oc_quantize_parse_type(NULL, NULL), OC_ERR_INVALID_ARG);
}

Test(quantize_tool, block_size)
{
    /* Q4_0: 32 elements per block, 18 bytes per block. */
    size_t sz = oc_quantize_block_size(OC_QUANT_Q4_0, 32);
    cr_assert_eq(sz, 18);
    /* Q8_0: 32 elements, 34 bytes. */
    sz = oc_quantize_block_size(OC_QUANT_Q8_0, 32);
    cr_assert_eq(sz, 34);
}
