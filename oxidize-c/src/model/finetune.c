/*
 * finetune.c — LoRA/SFT finetuning implementation (stubs + infrastructure).
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/finetune.h"

#include "oxidize/chat.h"
#include "oxidize/sampling.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *oc_ft_strategy_name(OcFtStrategy s)
{
    switch (s) {
    case OC_FT_SFT:        return "sft";
    case OC_FT_SELF_TRAIN: return "self-train";
    case OC_FT_DPO:        return "dpo";
    case OC_FT_PPO:        return "ppo";
    default: return "unknown";
    }
}

OcError oc_finetune_run(const OcFtConfig *cfg)
{
    if (!cfg || !cfg->model_path || !cfg->dataset_path) return OC_ERR_INVALID_ARG;

    if (cfg->verbose) {
        fprintf(stderr, "finetune: strategy=%s model=%s dataset=%s\n",
                oc_ft_strategy_name(cfg->strategy),
                cfg->model_path, cfg->dataset_path);
        fprintf(stderr, "  lora_rank=%u lora_alpha=%u epochs=%u batch_size=%u\n",
                cfg->lora_rank, cfg->lora_alpha, cfg->epochs, cfg->batch_size);
        fprintf(stderr, "  lr=%.6f max_seq=%u\n",
                cfg->learning_rate, cfg->max_seq_length);
    }

    /* The full training loop requires:
     * 1. Load base model
     * 2. Initialize LoRA adapters (A=random, B=zero)
     * 3. For each epoch:
     *    a. Load batch from dataset (JSONL)
     *    b. Tokenize prompt + response
     *    c. Forward pass with LoRA applied
     *    d. Compute loss (cross-entropy on response tokens)
     *    e. Backward pass (compute gradients for A, B)
     *    f. Update A, B with optimizer (Adam)
     * 4. Save LoRA adapters
     *
     * Steps d-g require autograd which is not implemented in the C port.
     * For now, we provide the infrastructure and return OC_ERR_UNSUPPORTED.
     */

    fprintf(stderr, "finetune: training loop not yet implemented (requires autograd)\n");
    fprintf(stderr, "  Use the Rust oxidize-finetuning crate for actual training.\n");
    fprintf(stderr, "  The C port supports LoRA adapter inference (see lora.h).\n");

    /* Initialize and save a zero LoRA adapter as a placeholder. */
    OcLlamaModel model;
    OcError e = oc_llama_load(cfg->model_path, &model);
    if (e != OC_OK) return e;

    OcLoraModel lm;
    e = oc_lora_model_init(&lm, model.cfg.n_layer);
    if (e != OC_OK) { oc_llama_free(&model); return e; }

    /* Save empty adapter. */
    if (cfg->output_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/adapter.bin", cfg->output_dir);
        e = oc_lora_save(&lm, path);
        if (cfg->verbose) {
            fprintf(stderr, "  saved placeholder adapter to %s\n", path);
        }
    }

    oc_lora_model_free(&lm);
    oc_llama_free(&model);
    return e;
}

OcError oc_lora_save(const OcLoraModel *lm, const char *path)
{
    if (!lm || !path) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;

    /* Write header: magic + n_layers + active flag. */
    uint32_t magic = 0x4C4F5241; /* "LORA" */
    fwrite(&magic, 4, 1, f);
    fwrite(&lm->n_layers, sizeof(size_t), 1, f);
    fwrite(&lm->active, sizeof(bool), 1, f);

    /* Write each adapter array. */
    const OcLoraAdapter *arrays[] = {
        lm->q_adapters, lm->k_adapters, lm->v_adapters, lm->o_adapters,
        lm->gate_adapters, lm->up_adapters, lm->down_adapters
    };
    for (size_t a = 0; a < 7; a++) {
        for (size_t l = 0; l < lm->n_layers; l++) {
            const OcLoraAdapter *ad = &arrays[a][l];
            fwrite(&ad->rank, sizeof(uint32_t), 1, f);
            fwrite(&ad->rows, sizeof(uint32_t), 1, f);
            fwrite(&ad->cols, sizeof(uint32_t), 1, f);
            fwrite(&ad->alpha, sizeof(float), 1, f);
            if (ad->a && ad->rank > 0) {
                size_t n = (size_t)ad->rank * ad->cols;
                fwrite(ad->a, sizeof(float), n, f);
                size_t m = (size_t)ad->rows * ad->rank;
                fwrite(ad->b, sizeof(float), m, f);
            }
        }
    }

    fclose(f);
    return OC_OK;
}

OcError oc_lora_load(const char *path, OcLoraModel *lm)
{
    if (!path || !lm) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;

    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4C4F5241) {
        fclose(f);
        return OC_ERR_FORMAT;
    }
    size_t n_layers = 0;
    if (fread(&n_layers, sizeof(size_t), 1, f) != 1) { fclose(f); return OC_ERR_FORMAT; }
    bool active = false;
    if (fread(&active, sizeof(bool), 1, f) != 1) { fclose(f); return OC_ERR_FORMAT; }

    OcError e = oc_lora_model_init(lm, n_layers);
    if (e != OC_OK) { fclose(f); return e; }
    lm->active = active;

    OcLoraAdapter *arrays[] = {
        lm->q_adapters, lm->k_adapters, lm->v_adapters, lm->o_adapters,
        lm->gate_adapters, lm->up_adapters, lm->down_adapters
    };
    for (size_t a = 0; a < 7; a++) {
        for (size_t l = 0; l < n_layers; l++) {
            OcLoraAdapter *ad = &arrays[a][l];
            if (fread(&ad->rank, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->rows, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->cols, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->alpha, sizeof(float), 1, f) != 1) goto fail;
            if (ad->rank > 0 && ad->rows > 0 && ad->cols > 0) {
                size_t n = (size_t)ad->rank * ad->cols;
                ad->a = malloc(n * sizeof(float));
                size_t m = (size_t)ad->rows * ad->rank;
                ad->b = malloc(m * sizeof(float));
                if (!ad->a || !ad->b) goto fail;
                if (fread(ad->a, sizeof(float), n, f) != n) goto fail;
                if (fread(ad->b, sizeof(float), m, f) != m) goto fail;
            }
        }
    }

    fclose(f);
    return OC_OK;
fail:
    fclose(f);
    oc_lora_model_free(lm);
    return OC_ERR_FORMAT;
}

OcError oc_finetune_generate_synthetic(OcLlamaModel *model, OcTokenizer *tok,
                                        const char *prompt, uint32_t n_samples,
                                        const char *output_path)
{
    if (!model || !tok || !prompt || !output_path) return OC_ERR_INVALID_ARG;

    FILE *f = fopen(output_path, "w");
    if (!f) return OC_ERR_IO;

    OcLlamaSession sess;
    OcError e = oc_llama_session_init(model, &sess);
    if (e != OC_OK) { fclose(f); return e; }

    OcSamplerConfig scfg = {
        .type = OC_SAMPLER_TEMPERATURE,
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.9f,
        .repeat_penalty = 1.1f,
        .seed = 42,
    };

    for (uint32_t i = 0; i < n_samples; i++) {
        /* Encode the prompt. */
        uint32_t *ids = NULL;
        size_t n_ids = 0;
        OcSpecialTokenPolicy pol = tok->has_add_bos_token && tok->add_bos_token
            ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
        e = oc_tokenizer_encode(tok, prompt, pol, &ids, &n_ids);
        if (e != OC_OK) continue;

        /* Reset session for each sample. */
        oc_llama_session_reset(&sess);

        /* Forward the prompt. */
        for (size_t j = 0; j < n_ids; j++) {
            oc_llama_forward(&sess, ids[j], NULL);
        }
        free(ids);

        /* Generate response. */
        char *response = NULL;
        size_t resp_len = 0;
        float *logits = sess.logits;
        for (size_t t = 0; t < 256; t++) {
            e = oc_llama_forward(&sess,
                t == 0 ? oc_argmax(logits, model->cfg.vocab_size) : 0,
                logits);
            if (e != OC_OK) break;
            uint32_t token = oc_sample(logits, model->cfg.vocab_size, &scfg, NULL, 0);
            if (tok->has_eos && token == tok->eos_id) break;

            char *piece = NULL;
            if (oc_tokenizer_decode(tok, &token, 1, &piece) == OC_OK && piece) {
                size_t pl = strlen(piece);
                char *tmp = realloc(response, resp_len + pl + 1);
                if (tmp) {
                    response = tmp;
                    memcpy(response + resp_len, piece, pl);
                    resp_len += pl;
                    response[resp_len] = '\0';
                }
                free(piece);
            }
        }

        /* Write as JSONL. */
        if (response) {
            fprintf(f, "{\"prompt\":\"%s\",\"response\":\"%s\"}\n", prompt, response);
            free(response);
        }

        scfg.seed++; /* Different seed for each sample. */
    }

    oc_llama_session_free(&sess);
    fclose(f);
    return OC_OK;
}

OcError oc_finetune_format_sft(const char *system, const char *user,
                               const char *assistant, char *out, size_t out_cap)
{
    if (!out || out_cap == 0) return OC_ERR_INVALID_ARG;
    size_t n = 0;
    if (system) {
        int w = snprintf(out + n, out_cap - n, "<|im_start|>system\n%s<|im_end|>\n", system);
        if (w < 0 || (size_t)w >= out_cap - n) return OC_ERR_OOM;
        n += (size_t)w;
    }
    if (user) {
        int w = snprintf(out + n, out_cap - n, "<|im_start|>user\n%s<|im_end|>\n", user);
        if (w < 0 || (size_t)w >= out_cap - n) return OC_ERR_OOM;
        n += (size_t)w;
    }
    if (assistant) {
        int w = snprintf(out + n, out_cap - n, "<|im_start|>assistant\n%s<|im_end|>\n", assistant);
        if (w < 0 || (size_t)w >= out_cap - n) return OC_ERR_OOM;
        n += (size_t)w;
    }
    return OC_OK;
    return OC_OK;
}
