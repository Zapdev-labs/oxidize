/* test_mtp.c — MTP/nextn draft generation tests. */
#include <criterion/criterion.h>
#include "oxidize/mtp.h"
#include <string.h>

static float test_rng(void) { return 0.5f; }

Test(mtp, config_init)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cr_assert_eq(cfg.hidden_size, 4096);
    cr_assert_eq(cfg.vocab_size, 32000);
    cr_assert_eq(cfg.n_layers, 1);
    cr_assert_eq(cfg.max_tokens, 4);
    cr_assert(!cfg.quantspec_draft_kv);
}

Test(mtp, engine_init)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    OcMtpEngine engine;
    cr_assert_eq(oc_mtp_engine_init(&engine, &cfg), OC_OK);
    cr_assert(oc_mtp_has_block(&engine));
    cr_assert_eq(oc_mtp_n_draft(&engine), 0);
    oc_mtp_engine_free(&engine);
}

Test(mtp, engine_init_null)
{
    cr_assert_neq(oc_mtp_engine_init(NULL, NULL), OC_OK);
}

Test(mtp, engine_init_bad_config)
{
    OcMtpConfig cfg = {0};
    OcMtpEngine engine;
    cr_assert_neq(oc_mtp_engine_init(&engine, &cfg), OC_OK);
}

Test(mtp, draft_basic)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 128;
    cfg.vocab_size = 100;
    cfg.max_tokens = 3;
    OcMtpEngine engine;
    cr_assert_eq(oc_mtp_engine_init(&engine, &cfg), OC_OK);

    float hidden[128] = {0};
    for (int i = 0; i < 128; i++) hidden[i] = (float)i * 0.01f;

    cr_assert_eq(oc_mtp_draft(&engine, 5, hidden, 128, 3, test_rng), OC_OK);
    cr_assert_eq(oc_mtp_n_draft(&engine), 3);
    oc_mtp_engine_free(&engine);
}

Test(mtp, draft_zero_max)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    cfg.max_tokens = 4;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    cr_assert_eq(oc_mtp_draft(&engine, 0, hidden, 64, 0, test_rng), OC_OK);
    cr_assert_eq(oc_mtp_n_draft(&engine), 0);
    oc_mtp_engine_free(&engine);
}

Test(mtp, draft_hidden_mismatch)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 128;
    cfg.vocab_size = 100;
    cfg.max_tokens = 2;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    cr_assert_neq(oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng), OC_OK);
    oc_mtp_engine_free(&engine);
}

Test(mtp, get_draft_token)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    cfg.max_tokens = 2;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng);

    uint32_t tok0, tok1;
    cr_assert_eq(oc_mtp_get_draft_token(&engine, 0, &tok0), OC_OK);
    cr_assert_eq(oc_mtp_get_draft_token(&engine, 1, &tok1), OC_OK);
    cr_assert(tok0 < 50);
    oc_mtp_engine_free(&engine);
}

Test(mtp, get_draft_token_out_of_range)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    cfg.max_tokens = 2;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng);

    uint32_t tok;
    cr_assert_neq(oc_mtp_get_draft_token(&engine, 99, &tok), OC_OK);
    oc_mtp_engine_free(&engine);
}

Test(mtp, get_draft_logits)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    cfg.max_tokens = 2;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng);

    float *logits;
    cr_assert_eq(oc_mtp_get_draft_logits(&engine, 0, &logits), OC_OK);
    cr_assert_not_null(logits);
    oc_mtp_engine_free(&engine);
}

Test(mtp, reset)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    cfg.max_tokens = 2;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);

    float hidden[64] = {0};
    oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng);
    cr_assert_eq(oc_mtp_n_draft(&engine), 2);
    oc_mtp_reset(&engine);
    cr_assert_eq(oc_mtp_n_draft(&engine), 0);
    oc_mtp_engine_free(&engine);
}

Test(mtp, config_name)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cr_assert_str_eq(oc_mtp_config_name(&cfg), "mtp/nextn");
}

Test(mtp, draft_no_mtp)
{
    OcMtpConfig cfg;
    oc_mtp_config_init(&cfg);
    cfg.hidden_size = 64;
    cfg.vocab_size = 50;
    OcMtpEngine engine;
    oc_mtp_engine_init(&engine, &cfg);
    engine.has_mtp = false;

    float hidden[64] = {0};
    cr_assert_eq(oc_mtp_draft(&engine, 0, hidden, 64, 2, test_rng), OC_ERR_MODEL);
    oc_mtp_engine_free(&engine);
}

Test(mtp, draft_null_args)
{
    cr_assert_neq(oc_mtp_draft(NULL, 0, NULL, 0, 0, NULL), OC_OK);
}

Test(mtp, free_null)
{
    oc_mtp_engine_free(NULL);
}
