/*
 * diffusion_gemma.c — Gemma diffusion model implementation.
 */
#include "oxidize/diffusion_gemma.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

OcError oc_diff_gemma_config_init(OcDiffGemmaConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_layers = 18;
    cfg->hidden_dim = 2048;
    cfg->vocab_size = 256000;
    cfg->n_diffusion_steps = 50;
    cfg->sigma_min = 0.002f;
    cfg->sigma_max = 80.0f;
    cfg->rho = 7.0f;
    cfg->sampler = OC_DIFF_GEMMA_FLOW_MATCH;
    cfg->use_classifier_free_guidance = false;
    cfg->guidance_scale = 3.5f;
    return OC_OK;
}

OcError oc_diff_gemma_model_init(OcDiffGemmaModel *model, const OcDiffGemmaConfig *cfg)
{
    if (!model) return OC_ERR_INVALID_ARG;
    OcDiffGemmaConfig defaults;
    if (!cfg) {
        oc_diff_gemma_config_init(&defaults);
        cfg = &defaults;
    }
    if (cfg->n_layers == 0 || cfg->hidden_dim == 0 || cfg->vocab_size == 0)
        return OC_ERR_INVALID_ARG;

    memset(model, 0, sizeof(*model));
    model->config = *cfg;

    size_t h = cfg->hidden_dim;
    size_t nl = cfg->n_layers;
    size_t hh = h * h;

    model->embedding = calloc((size_t)cfg->vocab_size * h, sizeof(float));
    if (!model->embedding) return OC_ERR_OOM;

    model->output = calloc((size_t)cfg->vocab_size * h, sizeof(float));
    if (!model->output) goto fail;

    model->attn_norm = calloc(nl * h, sizeof(float));
    model->wq = calloc(nl * hh, sizeof(float));
    model->wk = calloc(nl * hh, sizeof(float));
    model->wv = calloc(nl * hh, sizeof(float));
    model->wo = calloc(nl * hh, sizeof(float));
    model->ffn_norm = calloc(nl * h, sizeof(float));
    model->w_gate = calloc(nl * hh, sizeof(float));
    model->w_up = calloc(nl * hh, sizeof(float));
    model->w_down = calloc(nl * hh, sizeof(float));
    model->final_norm = calloc(h, sizeof(float));
    if (!model->attn_norm || !model->wq || !model->wk || !model->wv ||
        !model->wo || !model->ffn_norm || !model->w_gate ||
        !model->w_up || !model->w_down || !model->final_norm)
        goto fail;

    model->initialized = true;
    return OC_OK;

fail:
    oc_diff_gemma_free(model);
    return OC_ERR_OOM;
}

OcError oc_diff_gemma_forward(OcDiffGemmaModel *model, uint32_t token, float sigma, float *logits)
{
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    OcDiffGemmaConfig *cfg = &model->config;
    size_t h = cfg->hidden_dim;
    size_t vocab = cfg->vocab_size;
    size_t tok_idx = (size_t)token;
    if (tok_idx >= vocab) tok_idx = vocab - 1;
    float eps = 1e-6f;

    /* Embed token, scale by sigma (noise level conditioning). */
    float *hidden = malloc(h * sizeof(float));
    if (!hidden) return OC_ERR_OOM;

    if (model->embedding)
        memcpy(hidden, model->embedding + tok_idx * h, h * sizeof(float));
    else
        memset(hidden, 0, h * sizeof(float));

    /* Scale hidden by sigma (noise-level conditioning). */
    float sigma_val = (sigma > 0.0f) ? sigma : 1.0f;
    for (size_t i = 0; i < h; i++)
        hidden[i] *= sigma_val;

    /* Transformer layers: attention + FFN with residual connections. */
    for (uint32_t li = 0; li < cfg->n_layers; li++) {
        /* Attention RMSNorm. */
        float *normed = malloc(h * sizeof(float));
        if (!normed) { free(hidden); return OC_ERR_OOM; }
        float *a_norm = model->attn_norm + li * h;
        float ss = 0.0f;
        for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
        float rms = 1.0f / sqrtf(ss / h + eps);
        for (size_t i = 0; i < h; i++)
            normed[i] = hidden[i] * rms * a_norm[i];

        /* QKV projections (single-head for simplicity). */
        float *q = calloc(h, sizeof(float));
        float *k = calloc(h, sizeof(float));
        float *v = calloc(h, sizeof(float));
        float *wq = model->wq + li * h * h;
        float *wk = model->wk + li * h * h;
        float *wv = model->wv + li * h * h;
        if (!q || !k || !v) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }
        for (size_t r = 0; r < h; r++) {
            float dq = 0.0f, dk = 0.0f, dv = 0.0f;
            for (size_t c = 0; c < h; c++) {
                float n = normed[c];
                dq += wq[r * h + c] * n;
                dk += wk[r * h + c] * n;
                dv += wv[r * h + c] * n;
            }
            q[r] = dq; k[r] = dk; v[r] = dv;
        }

        /* Self-attention: single-token, so just use V directly (no KV cache). */
        float *attn_out = calloc(h, sizeof(float));
        if (!attn_out) { free(hidden); free(normed); free(q); free(k); free(v); return OC_ERR_OOM; }
        /* Scale Q by 1/sqrt(h) and compute attention over single position. */
        float scale = 1.0f / sqrtf((float)h);
        (void)scale; /* single-token attention: score * v = v */
        /* Softmax over single position = 1.0, so attn_out = v. */
        memcpy(attn_out, v, h * sizeof(float));

        /* Output projection. */
        float *attn_resid = calloc(h, sizeof(float));
        if (!attn_resid) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); return OC_ERR_OOM; }
        float *wo = model->wo + li * h * h;
        for (size_t r = 0; r < h; r++) {
            float dot = 0.0f;
            for (size_t c = 0; c < h; c++) dot += wo[r * h + c] * attn_out[c];
            attn_resid[r] = dot;
        }
        /* Residual. */
        for (size_t i = 0; i < h; i++) hidden[i] += attn_resid[i];

        /* FFN RMSNorm. */
        float *f_norm = model->ffn_norm + li * h;
        ss = 0.0f;
        for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
        rms = 1.0f / sqrtf(ss / h + eps);
        for (size_t i = 0; i < h; i++)
            normed[i] = hidden[i] * rms * f_norm[i];

        /* SwiGLU FFN: gate * up -> down. */
        float *gate_out = calloc(h, sizeof(float));
        float *up_out = calloc(h, sizeof(float));
        float *w_gate = model->w_gate + li * h * h;
        float *w_up = model->w_up + li * h * h;
        if (!gate_out || !up_out) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid); free(gate_out); free(up_out); return OC_ERR_OOM; }
        for (size_t r = 0; r < h; r++) {
            float dg = 0.0f, du = 0.0f;
            for (size_t c = 0; c < h; c++) {
                float n = normed[c];
                dg += w_gate[r * h + c] * n;
                du += w_up[r * h + c] * n;
            }
            gate_out[r] = dg; up_out[r] = du;
        }
        /* SiLU(gate) * up. */
        float *act = calloc(h, sizeof(float));
        if (!act) { /* cleanup */ free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid); free(gate_out); free(up_out); return OC_ERR_OOM; }
        for (size_t i = 0; i < h; i++) {
            float silu = gate_out[i] / (1.0f + expf(-gate_out[i]));
            act[i] = silu * up_out[i];
        }
        /* Down projection. */
        float *w_down = model->w_down + li * h * h;
        float *mlp_out = calloc(h, sizeof(float));
        if (!mlp_out) { free(hidden); free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid); free(gate_out); free(up_out); free(act); return OC_ERR_OOM; }
        for (size_t r = 0; r < h; r++) {
            float dot = 0.0f;
            for (size_t c = 0; c < h; c++) dot += w_down[r * h + c] * act[c];
            mlp_out[r] = dot;
        }
        /* Residual. */
        for (size_t i = 0; i < h; i++) hidden[i] += mlp_out[i];

        free(normed); free(q); free(k); free(v); free(attn_out); free(attn_resid);
        free(gate_out); free(up_out); free(act); free(mlp_out);
    }

    /* Final RMSNorm. */
    float ss = 0.0f;
    for (size_t i = 0; i < h; i++) ss += hidden[i] * hidden[i];
    float rms = 1.0f / sqrtf(ss / h + eps);
    for (size_t i = 0; i < h; i++)
        hidden[i] *= rms * model->final_norm[i];

    /* LM head. */
    if (model->output) {
        for (size_t r = 0; r < vocab; r++) {
            float dot = 0.0f;
            const float *orow = model->output + r * h;
            for (size_t c = 0; c < h; c++)
                dot += orow[c] * hidden[c];
            logits[r] = dot;
        }
    } else {
        memset(logits, 0, vocab * sizeof(float));
    }

    free(hidden);
    return OC_OK;
}

OcError oc_diff_gemma_sample(OcDiffGemmaModel *model, float *logits, size_t n, uint32_t step)
{
    if (!model || !logits) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    OcDiffGemmaConfig *cfg = &model->config;
    if (n == 0) return OC_OK;

    /* Temperature based on diffusion step: colder at later steps. */
    float progress = (cfg->n_diffusion_steps > 0)
        ? (float)step / (float)cfg->n_diffusion_steps : 0.5f;
    float temperature = 1.0f - progress * 0.9f; /* 1.0 → 0.1 */
    if (temperature < 0.01f) temperature = 0.01f;

    /* Softmax with temperature, then argmax. */
    float max_logit = -INFINITY;
    for (size_t i = 0; i < n; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += logits[i];
    }

    if (sum_exp > 0.0f) {
        float inv = 1.0f / sum_exp;
        /* Argmax (greedy sampling). */
        size_t best = 0;
        float best_prob = 0.0f;
        for (size_t i = 0; i < n; i++) {
            logits[i] *= inv;
            if (logits[i] > best_prob) {
                best_prob = logits[i];
                best = i;
            }
        }
        /* Set logits to one-hot for the argmax. */
        memset(logits, 0, n * sizeof(float));
        logits[best] = 1.0f;
    }

    return OC_OK;
}

OcError oc_diff_gemma_denoise(OcDiffGemmaModel *model, float *tokens, size_t n,
                             float sigma_from, float sigma_to)
{
    if (!model || !tokens) return OC_ERR_INVALID_ARG;
    if (!model->initialized) return OC_ERR_MODEL;

    /* Euler denoising step: tokens = tokens + (sigma_to - sigma_from) * velocity.
     * In a real diffusion model, velocity comes from the model forward pass.
     * Here we implement the Euler step structure. */
    float delta = sigma_to - sigma_from;

    if (delta == 0.0f || n == 0) return OC_OK;

    /* Compute velocity = forward(tokens, sigma_from) - tokens (simplified). */
    float *logits = malloc(model->config.vocab_size * sizeof(float));
    if (!logits) return OC_ERR_OOM;

    for (size_t i = 0; i < n; i++) {
        uint32_t tok = (uint32_t)tokens[i];
        if (tok >= model->config.vocab_size) tok = 0;

        OcError e = oc_diff_gemma_forward(model, tok, sigma_from, logits);
        if (e != OC_OK) { free(logits); return e; }

        /* Sample to get denoised token. */
        e = oc_diff_gemma_sample(model, logits, model->config.vocab_size, 0);
        if (e != OC_OK) { free(logits); return e; }

        /* Find the argmax (one-hot in logits). */
        for (size_t j = 0; j < model->config.vocab_size; j++) {
            if (logits[j] > 0.5f) {
                /* Euler step: blend old and new token. */
                if (delta < 0.0f) {
                    /* Denoising: move toward new token. */
                    tokens[i] = (float)j;
                }
                break;
            }
        }
    }

    free(logits);
    return OC_OK;
}

const char *oc_diff_gemma_sampler_name(OcDiffGemmaSampler s)
{
    switch (s) {
    case OC_DIFF_GEMMA_DDIM:       return "ddim";
    case OC_DIFF_GEMMA_DPM2M:     return "dpm2m";
    case OC_DIFF_GEMMA_EULER:      return "euler";
    case OC_DIFF_GEMMA_FLOW_MATCH: return "flow_match";
    default: return "unknown";
    }
}

void oc_diff_gemma_free(OcDiffGemmaModel *model)
{
    if (!model) return;
    free(model->embedding);
    free(model->output);
    free(model->attn_norm);
    free(model->wq);
    free(model->wk);
    free(model->wv);
    free(model->wo);
    free(model->ffn_norm);
    free(model->w_gate);
    free(model->w_up);
    free(model->w_down);
    free(model->final_norm);
    memset(model, 0, sizeof(*model));
}
