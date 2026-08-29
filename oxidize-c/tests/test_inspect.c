/* test_inspect.c — model inspector tests. */
#include "framework.h"
#include "oxidize/inspect.h"
#include <string.h>

/* Build a manually-constructed OcModelInfo for format tests. */
static OcModelInfo make_test_info(void)
{
    OcModelInfo info;
    memset(&info, 0, sizeof(info));

    strncpy(info.arch, "llama", sizeof(info.arch) - 1);
    strncpy(info.name, "test-model", sizeof(info.name) - 1);

    info.n_layer = 32;
    info.n_embd = 4096;
    info.n_head = 32;
    info.n_head_kv = 8;
    info.n_ff = 11008;
    info.vocab_size = 32000;
    info.n_ctx = 4096;

    info.param_count = 7000000000ULL;  /* 7B */
    info.file_size = 4000000000ULL;    /* 4 GB */
    info.size_gb = 4.0;

    strncpy(info.quant_type, "Q4_K_M", sizeof(info.quant_type) - 1);
    strncpy(info.quant_description, "Dominant: Q4_K_M (85.0% of weight bytes)",
            sizeof(info.quant_description) - 1);
    info.n_tensors = 3;

    /* Allocate 3 tensor summaries. */
    info.tensors = (OcTensorSummary *)calloc(3, sizeof(OcTensorSummary));

    strncpy(info.tensors[0].name, "tok_embeddings.weight",
            sizeof(info.tensors[0].name) - 1);
    info.tensors[0].type = 12;  /* Q4_K */
    info.tensors[0].bytes = 500000000;
    info.tensors[0].n_dims = 2;
    info.tensors[0].dims[0] = 4096;
    info.tensors[0].dims[1] = 32000;

    strncpy(info.tensors[1].name, "blk.0.attn_q.weight",
            sizeof(info.tensors[1].name) - 1);
    info.tensors[1].type = 12;
    info.tensors[1].bytes = 100000000;
    info.tensors[1].n_dims = 2;
    info.tensors[1].dims[0] = 4096;
    info.tensors[1].dims[1] = 4096;

    strncpy(info.tensors[2].name, "output.weight",
            sizeof(info.tensors[2].name) - 1);
    info.tensors[2].type = 12;
    info.tensors[2].bytes = 500000000;
    info.tensors[2].n_dims = 2;
    info.tensors[2].dims[0] = 4096;
    info.tensors[2].dims[1] = 32000;

    info.estimated_ram_usage = 5200000000ULL;
    info.suggested_threads = 8;
    info.suggested_numa_interleave = false;

    strncpy(info.tokenizer_type, "SentencePiece", sizeof(info.tokenizer_type) - 1);
    info.bos_id = 1;
    info.eos_id = 2;
    info.pad_id = 0;

    info.uses_rope = true;
    info.uses_rms_norm = true;
    info.uses_swiglu = true;
    info.uses_gqa = true;
    info.uses_mla = false;
    info.rope_freq_base = 10000.0f;
    info.sliding_window = 0;

    return info;
}

Test(inspect, null_args_model)
{
    OcModelInfo info;
    cr_assert_eq(oc_inspect_model(NULL, &info), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_inspect_model("path", NULL), OC_ERR_INVALID_ARG);
}

Test(inspect, null_args_llama)
{
    OcModelInfo info;
    cr_assert_eq(oc_inspect_llama(NULL, &info), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_inspect_llama((const OcLlamaModel *)1, NULL),
                 OC_ERR_INVALID_ARG);
}

Test(inspect, null_args_format)
{
    OcModelInfo info = make_test_info();

    /* NULL info should return 0. */
    cr_assert_eq(oc_inspect_format(NULL, NULL, 0), 0);
    cr_assert_eq(oc_inspect_format_json(NULL, NULL, 0), 0);

    oc_inspect_free(&info);
}

Test(inspect, format_table)
{
    OcModelInfo info = make_test_info();
    char buf[8192];

    size_t n = oc_inspect_format(&info, buf, sizeof(buf));
    cr_assert(n > 0, "format should produce output");
    cr_assert(strstr(buf, "Model Inspector") != NULL, "should have title");
    cr_assert(strstr(buf, "llama") != NULL, "should show arch");
    cr_assert(strstr(buf, "Dimensions") != NULL, "should have dimensions");
    cr_assert(strstr(buf, "4096") != NULL, "should show n_embd");
    cr_assert(strstr(buf, "Q4_K_M") != NULL, "should show quant type");
    cr_assert(strstr(buf, "SentencePiece") != NULL, "should show tokenizer");
    cr_assert(strstr(buf, "tok_embeddings.weight") != NULL,
              "should list tensors");

    oc_inspect_free(&info);
}

Test(inspect, format_table_no_tensors)
{
    OcModelInfo info = make_test_info();
    /* Free the tensors to test the no-tensor path. */
    free(info.tensors);
    info.tensors = NULL;
    info.n_tensors = 0;

    char buf[8192];
    size_t n = oc_inspect_format(&info, buf, sizeof(buf));
    cr_assert(n > 0, "should still produce output without tensors");
    cr_assert(strstr(buf, "Model Inspector") != NULL);

    oc_inspect_free(&info);
}

Test(inspect, format_json_basic)
{
    OcModelInfo info = make_test_info();
    char buf[8192];

    size_t n = oc_inspect_format_json(&info, buf, sizeof(buf));
    cr_assert(n > 0, "JSON should produce output");

    /* Validate JSON structure. */
    cr_assert(buf[0] == '{', "JSON should start with {");
    cr_assert(buf[n - 1] == '}', "JSON should end with }");
    cr_assert(strstr(buf, "\"arch\":\"llama\"") != NULL, "should have arch");
    cr_assert(strstr(buf, "\"n_layer\":32") != NULL, "should have n_layer");
    cr_assert(strstr(buf, "\"n_embd\":4096") != NULL, "should have n_embd");
    cr_assert(strstr(buf, "\"vocab_size\":32000") != NULL,
              "should have vocab_size");
    cr_assert(strstr(buf, "\"quant_type\":\"Q4_K_M\"") != NULL,
              "should have quant_type");
    cr_assert(strstr(buf, "\"tokenizer_type\":\"SentencePiece\"") != NULL,
              "should have tokenizer_type");
    cr_assert(strstr(buf, "\"uses_rope\":true") != NULL,
              "should have uses_rope");
    cr_assert(strstr(buf, "\"uses_mla\":false") != NULL,
              "should have uses_mla");
    cr_assert(strstr(buf, "\"tensors\":[") != NULL, "should have tensors");
    cr_assert(strstr(buf, "\"name\":\"tok_embeddings.weight\"") != NULL,
              "should have tensor name");

    oc_inspect_free(&info);
}

Test(inspect, format_json_escape)
{
    OcModelInfo info;
    memset(&info, 0, sizeof(info));
    /* Put a string with special characters to test JSON escaping. */
    strncpy(info.arch, "test\"arch\\", sizeof(info.arch) - 1);
    strncpy(info.name, "model\nname", sizeof(info.name) - 1);

    char buf[8192];
    size_t n = oc_inspect_format_json(&info, buf, sizeof(buf));
    cr_assert(n > 0, "JSON with escaped chars should produce output");
    cr_assert(strstr(buf, "\\\"") != NULL, "should escape quotes");
    cr_assert(strstr(buf, "\\\\") != NULL, "should escape backslash");
    cr_assert(strstr(buf, "\\n") != NULL, "should escape newline");

    oc_inspect_free(&info);
}

Test(inspect, format_json_truncation)
{
    OcModelInfo info = make_test_info();
    char buf[32];  /* Very small buffer to force truncation. */

    size_t n = oc_inspect_format_json(&info, buf, sizeof(buf));
    cr_assert_eq(n, 0, "should return 0 on truncation");
    /* Buffer should be NUL-terminated. */
    cr_assert_eq(buf[sizeof(buf) - 1], '\0');

    oc_inspect_free(&info);
}

Test(inspect, format_table_truncation)
{
    OcModelInfo info = make_test_info();
    char buf[16];  /* Very small buffer. */

    size_t n = oc_inspect_format(&info, buf, sizeof(buf));
    cr_assert_eq(n, 0, "should return 0 on truncation");
    cr_assert_eq(buf[sizeof(buf) - 1], '\0');

    oc_inspect_free(&info);
}

Test(inspect, free_null_safe)
{
    /* Should not crash on NULL. */
    oc_inspect_free(NULL);

    /* Should not crash on zeroed struct. */
    OcModelInfo info;
    memset(&info, 0, sizeof(info));
    oc_inspect_free(&info);

    /* Double-free should be safe. */
    OcModelInfo info2 = make_test_info();
    oc_inspect_free(&info2);
    oc_inspect_free(&info2);
}

Test(inspect, format_json_no_tensors)
{
    OcModelInfo info = make_test_info();
    free(info.tensors);
    info.tensors = NULL;
    info.n_tensors = 0;

    char buf[8192];
    size_t n = oc_inspect_format_json(&info, buf, sizeof(buf));
    cr_assert(n > 0, "should produce output");
    cr_assert(strstr(buf, "\"tensors\":[]") != NULL,
              "should have empty tensors array");

    oc_inspect_free(&info);
}

Test(inspect, format_estimate_no_buf)
{
    OcModelInfo info = make_test_info();

    /* NULL buffer should return estimated length. */
    size_t est = oc_inspect_format(&info, NULL, 0);
    cr_assert(est > 0, "should return estimated length");

    size_t est_json = oc_inspect_format_json(&info, NULL, 0);
    cr_assert(est_json > 0, "should return estimated JSON length");

    oc_inspect_free(&info);
}
