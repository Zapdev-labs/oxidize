/*
 * finetune.c — LoRA/SFT finetuning implementation (stubs + infrastructure).
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/finetune.h"

#include "oxidize/chat.h"
#include "oxidize/log.h"
#include "oxidize/sampling.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *oc_ft_strategy_name(OcFtStrategy s)
{
    switch (s) {
    case OC_FT_SFT:        return "sft";
    case OC_FT_SELF_TRAIN: return "self-train";
    case OC_FT_DPO:        return "dpo";
    case OC_FT_PPO:        return "ppo";
    case OC_FT_DISTILL:    return "distill";
    default: return "unknown";
    }
}

OcError oc_finetune_run(const OcFtConfig *cfg)
{
    if (!cfg || !cfg->dataset_path) return OC_ERR_INVALID_ARG;

    if (cfg->strategy == OC_FT_DISTILL) {
        const char *dir = cfg->output_dir ? cfg->output_dir : "./adapters";
        /* Owner-only: distill traces hold prompts and responses. */
        if (mkdir(dir, 0700) != 0 && errno != EEXIST) return OC_ERR_IO;
        char out_path[1024];
        int nw = snprintf(out_path, sizeof(out_path), "%s/distill.jsonl", dir);
        if (nw < 0 || (size_t)nw >= sizeof(out_path)) return OC_ERR_INVALID_ARG;
        uint32_t kept = 0, dropped = 0;
        OcError me = oc_finetune_mold_dataset(cfg->dataset_path, out_path, &kept, &dropped);
        if (me != OC_OK) return me;
        fprintf(stderr, "distill: kept=%u dropped=%u output=%s\n", kept, dropped, out_path);
        return OC_OK;
    }

    if (!cfg->model_path) return OC_ERR_INVALID_ARG;

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

    /* Validate that the base model loads, but do not emit a placeholder
     * adapter: reporting success here would let callers deploy an
     * untrained (zero) adapter as a completed fine-tune. */
    OcLlamaModel model;
    OcError e = oc_llama_load(cfg->model_path, &model);
    if (e != OC_OK) return e;
    oc_llama_free(&model);
    return OC_ERR_MODEL; /* training is not implemented in the C port */
}

OcError oc_lora_save(const OcLoraModel *lm, const char *path)
{
    if (!lm || !path) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;

    /* Write header: magic + n_layers + active flag. Every write is
     * checked so a failed/short save is reported, not silently kept. */
    bool ok = true;
    uint32_t magic = 0x4C4F5232; /* "LOR2" — includes shexp/ssm adapters */
    ok = ok && fwrite(&magic, 4, 1, f) == 1;
    ok = ok && fwrite(&lm->n_layers, sizeof(size_t), 1, f) == 1;
    ok = ok && fwrite(&lm->active, sizeof(bool), 1, f) == 1;

    /* Write each adapter array. */
    const OcLoraAdapter *arrays[] = {
        lm->q_adapters, lm->k_adapters, lm->v_adapters, lm->o_adapters,
        lm->gate_adapters, lm->up_adapters, lm->down_adapters,
        lm->shexp_gate_adapters, lm->shexp_up_adapters, lm->shexp_down_adapters,
        lm->ssm_qkv_adapters, lm->ssm_gate_adapters, lm->ssm_out_adapters,
        lm->ssm_alpha_adapters, lm->ssm_beta_adapters
    };
    const size_t n_arrays = sizeof(arrays) / sizeof(arrays[0]);
    for (size_t a = 0; ok && a < n_arrays; a++) {
        for (size_t l = 0; ok && l < lm->n_layers; l++) {
            const OcLoraAdapter *ad = &arrays[a][l];
            ok = ok && fwrite(&ad->rank, sizeof(uint32_t), 1, f) == 1;
            ok = ok && fwrite(&ad->rows, sizeof(uint32_t), 1, f) == 1;
            ok = ok && fwrite(&ad->cols, sizeof(uint32_t), 1, f) == 1;
            ok = ok && fwrite(&ad->alpha, sizeof(float), 1, f) == 1;
            if (ok && ad->a && ad->rank > 0) {
                size_t n = (size_t)ad->rank * ad->cols;
                size_t m = (size_t)ad->rows * ad->rank;
                ok = fwrite(ad->a, sizeof(float), n, f) == n
                  && fwrite(ad->b, sizeof(float), m, f) == m;
            }
        }
    }

    if (fclose(f) != 0) ok = false;
    if (!ok) {
        remove(path); /* do not leave a partial checkpoint behind */
        return OC_ERR_IO;
    }
    return OC_OK;
}

OcError oc_lora_load(const char *path, OcLoraModel *lm)
{
    if (!path || !lm) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;

    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 ||
        (magic != 0x4C4F5241 && magic != 0x4C4F5232)) {
        fclose(f);
        return OC_ERR_FORMAT;
    }
    size_t n_kinds = (magic == 0x4C4F5232) ? 15 : 7;
    size_t n_layers = 0;
    if (fread(&n_layers, sizeof(size_t), 1, f) != 1 ||
        n_layers == 0 || n_layers > 100000) { fclose(f); return OC_ERR_FORMAT; }
    bool active = false;
    if (fread(&active, sizeof(bool), 1, f) != 1) { fclose(f); return OC_ERR_FORMAT; }

    OcError e = oc_lora_model_init(lm, n_layers);
    if (e != OC_OK) { fclose(f); return e; }
    lm->active = active;

    OcLoraAdapter *arrays[] = {
        lm->q_adapters, lm->k_adapters, lm->v_adapters, lm->o_adapters,
        lm->gate_adapters, lm->up_adapters, lm->down_adapters,
        lm->shexp_gate_adapters, lm->shexp_up_adapters, lm->shexp_down_adapters,
        lm->ssm_qkv_adapters, lm->ssm_gate_adapters, lm->ssm_out_adapters,
        lm->ssm_alpha_adapters, lm->ssm_beta_adapters
    };
    for (size_t a = 0; a < n_kinds; a++) {
        for (size_t l = 0; l < n_layers; l++) {
            OcLoraAdapter *ad = &arrays[a][l];
            if (fread(&ad->rank, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->rows, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->cols, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&ad->alpha, sizeof(float), 1, f) != 1) goto fail;
            if (ad->rank > 0 && ad->rows > 0 && ad->cols > 0) {
                /* Overflow-check the shape arithmetic before allocating:
                 * a corrupt adapter must not make fread write past an
                 * undersized buffer. */
                uint64_t n64 = (uint64_t)ad->rank * ad->cols;
                uint64_t m64 = (uint64_t)ad->rows * ad->rank;
                if (n64 > SIZE_MAX / sizeof(float) ||
                    m64 > SIZE_MAX / sizeof(float)) goto fail;
                size_t n = (size_t)n64;
                ad->a = malloc(n * sizeof(float));
                size_t m = (size_t)m64;
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

/* Write `s` to `f` with JSON string escaping. */
static void fput_json_escaped(FILE *f, const char *s)
{
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f);  break;
        case '\r': fputs("\\r", f);  break;
        case '\t': fputs("\\t", f);  break;
        default:
            if (c < 0x20) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
    }
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
        if (n_ids == 0) { free(ids); continue; }

        /* Reset session for each sample. */
        oc_llama_session_reset(&sess);

        /* Prefill all but the final prompt token, then forward the final
         * token requesting logits so generation continues the prompt. */
        float *logits = sess.logits;
        for (size_t j = 0; j + 1 < n_ids; j++) {
            oc_llama_forward(&sess, ids[j], NULL);
        }
        e = oc_llama_forward(&sess, ids[n_ids - 1], logits);
        free(ids);
        if (e != OC_OK) continue;

        /* Generate response: sample from the prompt's next-token
         * distribution, then forward each sampled token. */
        char *response = NULL;
        size_t resp_len = 0;
        for (size_t t = 0; t < 256; t++) {
            uint32_t token = oc_sample(logits, model->cfg.vocab_size, &scfg, NULL, 0);
            if (oc_tokenizer_is_eog(tok, token)) break;

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

            e = oc_llama_forward(&sess, token, logits);
            if (e != OC_OK) break;
        }

        /* Write as JSONL (JSON-escaped so special characters stay valid). */
        if (response) {
            fputs("{\"prompt\":\"", f);
            fput_json_escaped(f, prompt);
            fputs("\",\"response\":\"", f);
            fput_json_escaped(f, response);
            fputs("\"}\n", f);
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
}

/* ─── Minimal JSON reader for the mold step ────────────────────────────── */

/* Longest decoded field kept; longer rows are counted as oversize. */
#define FT_FIELD_MAX ((size_t)1 << 20)
#define FT_JSON_DEPTH 64

typedef struct {
    char  *p;
    size_t n, cap;
    int    oversize;
} FtStr;

static int ft_str_putc(FtStr *b, char c)
{
    if (b->n + 1 >= FT_FIELD_MAX) { b->oversize = 1; return 1; }
    if (b->n + 2 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 64;
        char *np = realloc(b->p, nc);
        if (!np) return 0;
        b->p = np;
        b->cap = nc;
    }
    b->p[b->n++] = c;
    b->p[b->n] = '\0';
    return 1;
}

static int ft_str_put_utf8(FtStr *b, uint32_t cp)
{
    if (cp < 0x80) return ft_str_putc(b, (char)cp);
    if (cp < 0x800)
        return ft_str_putc(b, (char)(0xC0 | (cp >> 6))) &&
               ft_str_putc(b, (char)(0x80 | (cp & 0x3F)));
    if (cp < 0x10000)
        return ft_str_putc(b, (char)(0xE0 | (cp >> 12))) &&
               ft_str_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F))) &&
               ft_str_putc(b, (char)(0x80 | (cp & 0x3F)));
    return ft_str_putc(b, (char)(0xF0 | (cp >> 18))) &&
           ft_str_putc(b, (char)(0x80 | ((cp >> 12) & 0x3F))) &&
           ft_str_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F))) &&
           ft_str_putc(b, (char)(0x80 | (cp & 0x3F)));
}

static void ft_str_free(FtStr *b) { free(b->p); memset(b, 0, sizeof(*b)); }

static void ft_ws(const char **pp)
{
    const char *p = *pp;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    *pp = p;
}

static int ft_hex4(const char *p, uint32_t *out)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

/* Decode a JSON string at *pp (which points at the opening quote) into
 * `out` (may be NULL to skip). Handles every escape, including \uXXXX and
 * surrogate pairs (to UTF-8); lone surrogates become U+FFFD. Returns 1 on
 * success, 0 on malformed input or OOM. */
static int ft_json_string(const char **pp, FtStr *out)
{
    const char *p = *pp;
    if (*p != '"') return 0;
    p++;
    FtStr sink = {0};
    FtStr *b = out ? out : &sink;
    int ok = 1;
    while (ok && *p && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c != '\\') {
            ok = out ? ft_str_putc(b, (char)c) : 1;
            continue;
        }
        char e = *p++;
        uint32_t cp = 0;
        switch (e) {
        case '"':  cp = '"';  break;
        case '\\': cp = '\\'; break;
        case '/':  cp = '/';  break;
        case 'b':  cp = '\b'; break;
        case 'f':  cp = '\f'; break;
        case 'n':  cp = '\n'; break;
        case 'r':  cp = '\r'; break;
        case 't':  cp = '\t'; break;
        case 'u':
            if (!ft_hex4(p, &cp)) { ok = 0; break; }
            p += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t lo = 0;
                if (p[0] == '\\' && p[1] == 'u' && ft_hex4(p + 2, &lo) &&
                    lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                } else {
                    cp = 0xFFFD;
                }
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                cp = 0xFFFD;
            }
            break;
        default:
            ok = 0;
        }
        if (ok && out) ok = ft_str_put_utf8(b, cp);
    }
    if (ok && *p != '"') ok = 0;
    if (ok) {
        p++;
        if (out && !out->p) ok = ft_str_putc(out, '\0') && (out->n = 0, 1);
    }
    *pp = p;
    return ok;
}

/* Skip any JSON value. Returns 1 on success. */
static int ft_json_skip(const char **pp, int depth)
{
    if (depth > FT_JSON_DEPTH) return 0;
    const char *p = *pp;
    ft_ws(&p);
    if (*p == '"') {
        int ok = ft_json_string(&p, NULL);
        *pp = p;
        return ok;
    }
    if (*p == '{' || *p == '[') {
        const char close = *p == '{' ? '}' : ']';
        const int obj = *p == '{';
        p++;
        ft_ws(&p);
        if (*p == close) { *pp = p + 1; return 1; }
        for (;;) {
            if (obj) {
                ft_ws(&p);
                if (!ft_json_string(&p, NULL)) return 0;
                ft_ws(&p);
                if (*p++ != ':') return 0;
            }
            if (!ft_json_skip(&p, depth + 1)) return 0;
            ft_ws(&p);
            if (*p == ',') { p++; continue; }
            if (*p == close) { *pp = p + 1; return 1; }
            return 0;
        }
    }
    /* number / true / false / null */
    const char *s0 = p;
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
        p++;
    *pp = p;
    return p > s0;
}

typedef struct {
    FtStr role, content;
} FtMsg;

typedef struct {
    FtStr  system, user, assistant, instruction, input, output, source;
    FtMsg *msgs;
    size_t n_msgs, cap_msgs;
    int    has_messages;
} FtRow;

static void ft_row_free(FtRow *r)
{
    ft_str_free(&r->system); ft_str_free(&r->user);
    ft_str_free(&r->assistant); ft_str_free(&r->instruction);
    ft_str_free(&r->input); ft_str_free(&r->output); ft_str_free(&r->source);
    for (size_t i = 0; i < r->n_msgs; i++) {
        ft_str_free(&r->msgs[i].role);
        ft_str_free(&r->msgs[i].content);
    }
    free(r->msgs);
    memset(r, 0, sizeof(*r));
}

static int ft_row_oversize(const FtRow *r)
{
    int o = r->system.oversize | r->user.oversize | r->assistant.oversize |
            r->instruction.oversize | r->input.oversize | r->output.oversize |
            r->source.oversize;
    for (size_t i = 0; i < r->n_msgs; i++)
        o |= r->msgs[i].role.oversize | r->msgs[i].content.oversize;
    return o;
}

/* Parse one message object {"role":..,"content":..}. */
static int ft_parse_msg(const char **pp, FtRow *r)
{
    const char *p = *pp;
    ft_ws(&p);
    if (*p != '{') return ft_json_skip(pp, 1);
    if (r->n_msgs == r->cap_msgs) {
        size_t nc = r->cap_msgs ? r->cap_msgs * 2 : 8;
        FtMsg *nm = realloc(r->msgs, nc * sizeof(*nm));
        if (!nm) return 0;
        r->msgs = nm;
        r->cap_msgs = nc;
    }
    FtMsg *m = &r->msgs[r->n_msgs++];
    memset(m, 0, sizeof(*m));
    p++;
    ft_ws(&p);
    if (*p == '}') { *pp = p + 1; return 1; }
    for (;;) {
        FtStr key = {0};
        ft_ws(&p);
        if (!ft_json_string(&p, &key)) { ft_str_free(&key); return 0; }
        ft_ws(&p);
        if (*p++ != ':') { ft_str_free(&key); return 0; }
        ft_ws(&p);
        FtStr *dst = NULL;
        if (key.p && strcmp(key.p, "role") == 0) dst = &m->role;
        else if (key.p && strcmp(key.p, "content") == 0) dst = &m->content;
        ft_str_free(&key);
        int ok = (dst && *p == '"') ? ft_json_string(&p, dst)
                                    : ft_json_skip(&p, 2);
        if (!ok) return 0;
        ft_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') { *pp = p + 1; return 1; }
        return 0;
    }
}

/* Parse a top-level row object. Returns 1 on success. */
static int ft_parse_row(const char *line, FtRow *r)
{
    const char *p = line;
    ft_ws(&p);
    if (*p++ != '{') return 0;
    ft_ws(&p);
    if (*p == '}') return 1;
    for (;;) {
        FtStr key = {0};
        ft_ws(&p);
        if (!ft_json_string(&p, &key)) { ft_str_free(&key); return 0; }
        ft_ws(&p);
        if (*p++ != ':') { ft_str_free(&key); return 0; }
        ft_ws(&p);
        const char *k = key.p ? key.p : "";
        FtStr *dst = NULL;
        int ok;
        if (strcmp(k, "messages") == 0 && *p == '[') {
            r->has_messages = 1;
            p++;
            ft_ws(&p);
            ok = 1;
            if (*p == ']') p++;
            else for (;;) {
                if (!ft_parse_msg(&p, r)) { ok = 0; break; }
                ft_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p == ']') { p++; break; }
                ok = 0;
                break;
            }
        } else {
            if (strcmp(k, "system") == 0) dst = &r->system;
            else if (strcmp(k, "user") == 0) dst = &r->user;
            else if (strcmp(k, "assistant") == 0) dst = &r->assistant;
            else if (strcmp(k, "instruction") == 0) dst = &r->instruction;
            else if (strcmp(k, "input") == 0) dst = &r->input;
            else if (strcmp(k, "output") == 0) dst = &r->output;
            else if (strcmp(k, "source") == 0) dst = &r->source;
            ok = (dst && *p == '"') ? ft_json_string(&p, dst)
                                    : ft_json_skip(&p, 1);
        }
        ft_str_free(&key);
        if (!ok) return 0;
        ft_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') return 1;
        return 0;
    }
}

/* ─── Request filter ────────────────────────────────────────────────────
 * Runs on fully decoded text (JSON escapes already resolved), lowercased
 * with every run of non-alphanumeric bytes folded to one space, so
 * escape, case and punctuation/spacing variants hit the same entry.
 * Entries are normalized phrases matched on word boundaries. This table
 * is the list the PR shipped with; widen it from a maintained policy list
 * rather than ad hoc. */
static const char *const k_ft_blocked[] = {
    "reverse shell", "meterpreter", "write an exploit", "exploit poc",
    "ransomware", "webshell", "malware sample",
};

static char *ft_normalize(const char *s)
{
    size_t n = strlen(s);
    char *o = malloc(n + 3);
    if (!o) return NULL;
    size_t k = 0;
    o[k++] = ' ';
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        const int alnum = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                          c >= 0x80;
        if (alnum) o[k++] = (char)c;
        else if (o[k - 1] != ' ') o[k++] = ' ';
    }
    if (o[k - 1] != ' ') o[k++] = ' ';
    o[k] = '\0';
    return o;
}

int oc_finetune_request_blocked(const char *text)
{
    if (!text) return 0;
    char *norm = ft_normalize(text);
    if (!norm) return 1; /* fail closed */
    int hit = 0;
    char pat[96];
    for (size_t i = 0; !hit && i < sizeof(k_ft_blocked) / sizeof(k_ft_blocked[0]); i++) {
        int w = snprintf(pat, sizeof(pat), " %s ", k_ft_blocked[i]);
        if (w > 0 && (size_t)w < sizeof(pat) && strstr(norm, pat)) hit = 1;
    }
    free(norm);
    return hit;
}

/* ─── Mold ──────────────────────────────────────────────────────────── */

static void ft_put_msg(FILE *out, int *first, const char *role, const char *content)
{
    if (!*first) fputc(',', out);
    *first = 0;
    fputs("{\"role\":\"", out);
    fput_json_escaped(out, role);
    fputs("\",\"content\":\"", out);
    fput_json_escaped(out, content);
    fputs("\"}", out);
}

static const char *ft_s(const FtStr *b) { return b->p ? b->p : ""; }

/* Write one row. Returns 1 kept, 0 dropped (missing fields / filtered). */
static int ft_emit_row(FILE *out, FtRow *r)
{
    if (r->has_messages && r->n_msgs > 0) {
        int has_user = 0, has_asst = 0;
        for (size_t i = 0; i < r->n_msgs; i++) {
            const char *role = ft_s(&r->msgs[i].role);
            const char *content = ft_s(&r->msgs[i].content);
            if (strcmp(role, "user") == 0 && content[0]) {
                has_user = 1;
                if (oc_finetune_request_blocked(content)) return 0;
            } else if (strcmp(role, "assistant") == 0 && content[0]) {
                has_asst = 1;
            }
        }
        if (!has_user || !has_asst) return 0;
        fputs("{\"messages\":[", out);
        int first = 1;
        for (size_t i = 0; i < r->n_msgs; i++) {
            const char *role = ft_s(&r->msgs[i].role);
            if (strcmp(role, "system") != 0 && strcmp(role, "user") != 0 &&
                strcmp(role, "assistant") != 0)
                continue;
            ft_put_msg(out, &first, role, ft_s(&r->msgs[i].content));
        }
        fputs("]", out);
    } else {
        /* Flat chat keys first, then Alpaca instruction/input/output. */
        FtStr user = {0};
        const char *u = ft_s(&r->user);
        if (!u[0] && r->instruction.p && r->instruction.p[0]) {
            for (const char *c = r->instruction.p; *c; c++)
                if (!ft_str_putc(&user, *c)) { ft_str_free(&user); return -1; }
            if (r->input.p && r->input.p[0]) {
                if (!ft_str_putc(&user, '\n')) { ft_str_free(&user); return -1; }
                for (const char *c = r->input.p; *c; c++)
                    if (!ft_str_putc(&user, *c)) { ft_str_free(&user); return -1; }
            }
            if (user.oversize) { ft_str_free(&user); return -2; }
            u = ft_s(&user);
        }
        const char *a = r->assistant.p && r->assistant.p[0] ? r->assistant.p
                                                            : ft_s(&r->output);
        if (!u[0] || !a[0] || oc_finetune_request_blocked(u)) {
            ft_str_free(&user);
            return 0;
        }
        fputs("{\"messages\":[", out);
        int first = 1;
        if (r->system.p && r->system.p[0])
            ft_put_msg(out, &first, "system", r->system.p);
        ft_put_msg(out, &first, "user", u);
        ft_put_msg(out, &first, "assistant", a);
        fputs("]", out);
        ft_str_free(&user);
    }
    if (r->source.p && r->source.p[0]) {
        fputs(",\"source\":\"", out);
        fput_json_escaped(out, r->source.p);
        fputc('"', out);
    }
    fputs("}\n", out);
    return 1;
}

OcError oc_finetune_mold_dataset(const char *input_path, const char *output_path,
                                 uint32_t *kept, uint32_t *dropped)
{
    if (!input_path || !output_path) return OC_ERR_INVALID_ARG;
    FILE *in = fopen(input_path, "r");
    if (!in) return OC_ERR_IO;

    /* Write to a private temp file beside the output and rename it into
     * place only on success: input == output can no longer truncate the
     * source, and a failed run leaves no partial dataset. */
    size_t plen = strlen(output_path);
    char *tmp = malloc(plen + sizeof(".tmp.XXXXXX"));
    if (!tmp) { fclose(in); return OC_ERR_OOM; }
    memcpy(tmp, output_path, plen);
    memcpy(tmp + plen, ".tmp.XXXXXX", sizeof(".tmp.XXXXXX"));
    int fd = mkstemp(tmp); /* created 0600 */
    if (fd < 0) { free(tmp); fclose(in); return OC_ERR_IO; }
    (void)fchmod(fd, S_IRUSR | S_IWUSR);
    FILE *out = fdopen(fd, "w");
    if (!out) {
        close(fd);
        unlink(tmp);
        free(tmp);
        fclose(in);
        return OC_ERR_IO;
    }

    OcError err = OC_OK;
    uint32_t k = 0, d = 0, big = 0;
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, in) != -1) {
        FtRow row;
        memset(&row, 0, sizeof(row));
        const int parsed = ft_parse_row(line, &row);
        int res;
        if (ft_row_oversize(&row)) res = -2;
        else if (!parsed) res = 0;
        else res = ft_emit_row(out, &row);
        ft_row_free(&row);
        if (res == 1) k++;
        else if (res == -2) { big++; d++; }
        else if (res == -1) { err = OC_ERR_OOM; break; }
        else d++;
    }
    free(line);
    if (err == OC_OK && ferror(in)) err = OC_ERR_IO;
    fclose(in);
    if (fflush(out) != 0 || ferror(out)) {
        if (err == OC_OK) err = OC_ERR_IO;
    }
    if (fclose(out) != 0 && err == OC_OK) err = OC_ERR_IO;
    if (err == OC_OK && rename(tmp, output_path) != 0) err = OC_ERR_IO;
    if (err != OC_OK) unlink(tmp);
    free(tmp);
    if (err != OC_OK) return err;
    if (big > 0)
        oc_log(OC_LOG_WARN, "distill: dropped %u row(s) with a field over %zu bytes",
               big, FT_FIELD_MAX);
    if (kept) *kept = k;
    if (dropped) *dropped = d;
    return OC_OK;
}
