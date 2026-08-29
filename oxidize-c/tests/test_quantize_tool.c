#define _POSIX_C_SOURCE 200809L

/* test_quantize_tool.c — offline quantization tool tests. */
#include "oxidize/quant.h"
#include "oxidize/quantize_tool.h"
#include "framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put_u32(FILE *f, uint32_t value) { fwrite(&value, 4, 1, f); }
static void put_u64(FILE *f, uint64_t value) { fwrite(&value, 8, 1, f); }

static void put_string(FILE *f, const char *value)
{
    size_t len = strlen(value);
    put_u64(f, len);
    fwrite(value, 1, len, f);
}

static char *write_input_gguf(void)
{
    char *path = strdup("/var/tmp/oxidize-quantize-input-XXXXXX");
    cr_assert_not_null(path);
    int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    FILE *f = fdopen(fd, "wb");
    cr_assert_not_null(f);

    put_u32(f, OC_GGUF_MAGIC);
    put_u32(f, 3);
    put_u64(f, 1);
    put_u64(f, 4);

    put_string(f, "test.u8");
    put_u32(f, OC_GGUF_MT_UINT8);
    fputc(0x7f, f);

    put_string(f, "test.u16");
    put_u32(f, OC_GGUF_MT_UINT16);
    uint16_t u16 = 0x1234;
    fwrite(&u16, sizeof(u16), 1, f);

    put_string(f, "test.bool");
    put_u32(f, OC_GGUF_MT_BOOL);
    fputc(1, f);

    put_string(f, "test.array");
    put_u32(f, OC_GGUF_MT_ARRAY);
    put_u32(f, OC_GGUF_MT_UINT32);
    put_u64(f, 2);
    put_u32(f, 11);
    put_u32(f, 22);

    put_string(f, "weight");
    put_u32(f, 2);
    put_u64(f, 32);
    put_u64(f, 1);
    put_u32(f, 0);
    put_u64(f, 0);

    long pos = ftell(f);
    while (pos % 32 != 0) {
        fputc(0, f);
        pos++;
    }
    for (size_t i = 0; i < 32; i++) {
        float value = (float)i - 16.0f;
        fwrite(&value, sizeof(value), 1, f);
    }
    cr_assert_eq(fclose(f), 0);
    return path;
}

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

Test(quantize_tool, parse_type_null) { cr_assert_eq(oc_quantize_parse_type(NULL, NULL), OC_ERR_INVALID_ARG); }

Test(quantize_tool, rejects_target_without_encoder)
{
    OcQuantizeConfig cfg = {
        .input_path = "missing.gguf",
        .output_path = "unused.gguf",
        .target_type = "IQ2_XXS",
    };
    cr_assert_eq(oc_quantize_model(&cfg), OC_ERR_QUANT);
}

Test(quantize_tool, rewrites_metadata_and_f32_weights)
{
    char *input = write_input_gguf();
    char output[] = "/var/tmp/oxidize-quantize-output-XXXXXX";
    int fd = mkstemp(output);
    cr_assert_geq(fd, 0);
    close(fd);

    OcQuantizeConfig cfg = {
        .input_path = input,
        .output_path = output,
        .target_type = "Q4_0",
    };
    cr_assert_eq(oc_quantize_model(&cfg), OC_OK);

    OcGgufFile file;
    cr_assert_eq(oc_gguf_open(output, &file), OC_OK);
    cr_assert_eq(file.tensor_count, 1);
    cr_assert_eq(file.tensors[0].ggml_type, oc_quant_type_to_ggml_id(OC_QUANT_Q4_0));

    const OcGgufMetadataValue *value = oc_gguf_metadata_get(&file, "test.u8");
    cr_assert_not_null(value);
    cr_assert_eq(value->v.u8, 0x7f);
    value = oc_gguf_metadata_get(&file, "test.u16");
    cr_assert_not_null(value);
    cr_assert_eq(value->v.u16, 0x1234);
    value = oc_gguf_metadata_get(&file, "test.bool");
    cr_assert_not_null(value);
    cr_assert(value->v.b);
    value = oc_gguf_metadata_get(&file, "test.array");
    cr_assert_not_null(value);
    cr_assert_eq(value->v.arr.len, 2);
    cr_assert_eq(value->v.arr.values[0].v.u32, 11);
    cr_assert_eq(value->v.arr.values[1].v.u32, 22);

    oc_gguf_free(&file);
    unlink(input);
    unlink(output);
    free(input);
}
