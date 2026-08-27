/* test_error.c — OcError enum + OcErrorCtx tests. */
#include <criterion/criterion.h>
#include "oxidize/error.h"

Test(error, msg_returns_ok)
{
    cr_assert_str_eq(oc_error_msg(OC_OK), "ok", "expected 'ok'");
    cr_assert(oc_error_is_ok(OC_OK), "OC_OK should be ok");
}

Test(error, msg_all_codes_nonempty)
{
    OcError codes[] = {
        OC_OK, OC_ERR_IO, OC_ERR_FORMAT, OC_ERR_QUANT, OC_ERR_TENSOR,
        OC_ERR_TOKENIZER, OC_ERR_MODEL, OC_ERR_BACKEND, OC_ERR_OOM,
        OC_ERR_INVALID_ARG, OC_ERR_NETWORK, OC_ERR_AUTH, OC_ERR_INTERNAL,
    };
    for (size_t i = 0; i < sizeof(codes)/sizeof(codes[0]); i++) {
        const char *m = oc_error_msg(codes[i]);
        cr_assert_not_null(m, "msg for code %d should not be NULL", codes[i]);
        cr_assert(m[0] != '\0', "msg for code %d should not be empty", codes[i]);
    }
}

Test(error, msg_unknown_code)
{
    const char *m = oc_error_msg((OcError)9999);
    cr_assert_str_eq(m, "unknown error", "expected 'unknown error'");
}

Test(error, ctx_chain_construct_and_free)
{
    OcErrorCtx *leaf = oc_error_ctx_new(OC_ERR_IO, "file not found", NULL);
    cr_assert_not_null(leaf, "leaf should allocate");
    OcErrorCtx *mid  = oc_error_ctx_new(OC_ERR_MODEL, "load failed", leaf);
    cr_assert_not_null(mid, "mid should allocate");
    OcErrorCtx *top  = oc_error_ctx_new(OC_ERR_INTERNAL, "boom", mid);
    cr_assert_not_null(top, "top should allocate");

    /* Walk the chain. */
    cr_assert_eq(top->code, OC_ERR_INTERNAL, "");
    cr_assert_not_null(top->cause, "");
    cr_assert_eq(top->cause->code, OC_ERR_MODEL, "");
    cr_assert_not_null(top->cause->cause, "");
    cr_assert_eq(top->cause->cause->code, OC_ERR_IO, "");
    cr_assert_null(top->cause->cause->cause, "");

    oc_error_ctx_free(top);  /* frees entire chain */
}

Test(error, ctx_format)
{
    OcErrorCtx *leaf = oc_error_ctx_new(OC_ERR_IO, "missing file", NULL);
    OcErrorCtx *top  = oc_error_ctx_new(OC_ERR_MODEL, "load failed", leaf);

    char buf[256];
    size_t n = oc_error_ctx_format(top, buf, sizeof(buf));
    cr_assert(n > 0, "format should produce output");
    cr_assert(strstr(buf, "model error") != NULL, "buf should contain code name");
    cr_assert(strstr(buf, "load failed") != NULL, "buf should contain msg");
    cr_assert(strstr(buf, "caused by") != NULL, "buf should contain chain");
    cr_assert(strstr(buf, "missing file") != NULL, "buf should contain cause msg");

    oc_error_ctx_free(top);
}

