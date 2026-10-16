/* dump_hf_truncate.c — build a small text-only HF reference checkpoint
 * from the native EXL3 CYBER-FROST pack: dequantize trellis tensors to
 * F32 (raw HF semantics: no V-perm, no +1 folds, no -exp), truncate to
 * the first N layers / M experts, and write plain SafeTensors so
 * llama.cpp's converter can produce an independent reference GGUF.
 *
 * Not installed; build ad hoc:
 *   cc -O2 -Iinclude -o /tmp/dump_hf scripts/dump_hf_truncate.c \
 *       src/format/safetensors.o src/util/mmap.o src/core/star.o -lm -lpthread
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/safetensors.h"
#include "oxidize/exl3.h"
#include "oxidize/oxk.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#define NLAYERS 48
#define NEXP 8
#define D 2560u
#define VOCAB 248320u
#define HC 4u
#define HCD (HC * D)
#define HCLR 320u
#define QROWS (24u * 256u)
#define KVROWS (2u * 256u)
#define KEYDIM (16u * 128u)
#define SINNER (48u * 128u)
#define CONVDIM (2u * KEYDIM + SINNER)
#define ISIZE 640u
#define IDXQ (4u * 128u)
#define IDXDIM 128u

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "dump_hf: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

/* ── safetensors writer (single file, all payloads in memory) ─────────── */
typedef struct {
    char *hdr;
    size_t hn, hcap;
    unsigned char *pay;
    size_t pn, pcap;
    size_t ntensors;
} Stw;

static void stw_init(Stw *w)
{
    memset(w, 0, sizeof(*w));
    w->hcap = 1u << 20;
    w->pcap = 1u << 28;
    w->hdr = malloc(w->hcap);
    w->pay = malloc(w->pcap);
    if (!w->hdr || !w->pay) die("oom");
    w->hdr[w->hn++] = '{';
}

static void stw_add(Stw *w, const char *name, const char *dtype,
                    const size_t *dims, size_t nd, const void *data,
                    size_t bytes)
{
    /* Grow the header buffer for the header ENTRY (name + dims + offsets,
     * a few hundred bytes), not the payload `bytes` — multi-GB tensors
     * would otherwise force a ~4 GiB JSON header allocation. */
    const size_t entry = strlen(name) + nd * 24 + 160;
    while (w->hn + entry + 512 > w->hcap) {
        w->hcap *= 2;
        w->hdr = realloc(w->hdr, w->hcap);
        if (!w->hdr) die("oom");
    }
    while (w->pn + bytes > w->pcap) {
        w->pcap *= 2;
        w->pay = realloc(w->pay, w->pcap);
        if (!w->pay) die("oom");
    }
    char dims_json[160];
    dims_json[0] = 0;
    for (size_t i = 0; i < nd; i++)
        snprintf(dims_json + strlen(dims_json),
                 sizeof(dims_json) - strlen(dims_json), "%s%zu",
                 i ? "," : "", dims[i]);
    w->hn += (size_t)snprintf(w->hdr + w->hn, w->hcap - w->hn,
        "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[%zu,%zu]}",
        w->ntensors ? "," : "", name, dtype, dims_json, w->pn, w->pn + bytes);
    memcpy(w->pay + w->pn, data, bytes);
    w->pn += bytes;
    w->ntensors++;
}

static void stw_write(const Stw *w, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write %s", path);
    char *hdr = malloc(w->hn + 2);
    if (!hdr) die("oom");
    memcpy(hdr, w->hdr, w->hn);
    hdr[w->hn] = '}';
    size_t n = w->hn + 1;
    uint8_t le[8];
    for (size_t i = 0; i < 8; i++) le[i] = (uint8_t)((uint64_t)n >> (i * 8));
    if (fwrite(le, 1, 8, f) != 8 || fwrite(hdr, 1, n, f) != n ||
        fwrite(w->pay, 1, w->pn, f) != w->pn)
        die("short write %s", path);
    free(hdr);
    fclose(f);
}

/* ── checkpoint index across all shards ───────────────────────────────── */
typedef struct {
    OcSafetensorsFile **files;
    size_t n_files;
} Src;

static void src_open(Src *s, const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) die("cannot open %s", dir);
    struct dirent *ent;
    s->files = NULL;
    s->n_files = 0;
    while ((ent = readdir(d))) {
        size_t n = strlen(ent->d_name);
        if (n < 12 || strcmp(ent->d_name + n - 12, ".safetensors")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        if (strstr(ent->d_name, "ngram")) continue; /* PLE: skip */
        OcSafetensorsFile *f = malloc(sizeof(*f));
        if (oc_safetensors_open(path, f) != OC_OK) die("cannot read %s", path);
        s->files = realloc(s->files, (s->n_files + 1) * sizeof(*s->files));
        if (!s->files) die("oom");
        s->files[s->n_files++] = f;    }
    closedir(d);
    if (!s->n_files) die("no shards in %s", dir);
    fprintf(stderr, "opened %zu shards\n", s->n_files);
}

static const OcSafetensorsTensor *src_find(Src *s, const char *name,
                                           const void **data)
{
    for (size_t i = 0; i < s->n_files; i++) {
        OcSafetensorsFile *f = s->files[i];
        for (size_t t = 0; t < f->n_tensors; t++) {
            const OcSafetensorsTensor *ten = &f->tensors[t];
            if (!strcmp(ten->name, name)) {
                /* raw_data points at the start of the data section */
                *(const uint8_t **)data =
                    (const uint8_t *)f->raw_data + ten->data_offset;
                return ten;
            }
        }
    }
    return NULL;
}

/* dequantize key.trellis (+suh/svh) to a malloc'd F32 [out][in] by pushing
 * an identity batch through the tested matmul kernel — exact by
 * construction (suh/svh are Hadamard transforms, not elementwise scales). */
static float *src_dequant(Src *s, const char *key, size_t in, size_t out)
{
    char name[512];
    const void *data = NULL;
    snprintf(name, sizeof(name), "%s.trellis", key);
    const OcSafetensorsTensor *t = src_find(s, name, &data);
    if (!t) die("missing %s", name);
    if (strcmp(t->dtype, "I16") || t->n_dims != 3 ||
        t->shape[0] != in / 16 || t->shape[1] != out / 16 ||
        t->shape[2] < 8 || t->shape[2] > 128 || t->shape[2] % 8)
        die("bad trellis %s", name);
    OcExl3Matrix m;
    memset(&m, 0, sizeof(m));
    m.input_dim = in;
    m.output_dim = out;
    m.packed = data;
    m.packed_size = t->data_length;
    m.bits_twice = (unsigned)(t->shape[2] / 8);
    /* codebook variant: mul1 -> 2, mcg -> 1, else the base codebook */
    const void *cdummy = NULL;
    snprintf(name, sizeof(name), "%s.mul1", key);
    if (src_find(s, name, &cdummy)) m.codebook = 2u;
    else {
        snprintf(name, sizeof(name), "%s.mcg", key);
        if (src_find(s, name, &cdummy)) m.codebook = 1u;
    }
    float *suh = malloc(in * sizeof(float));
    float *svh = malloc(out * sizeof(float));
    if (!suh || !svh) die("oom");
    snprintf(name, sizeof(name), "%s.suh", key);
    const void *sd = NULL;
    t = src_find(s, name, &sd);
    if (t && !strcmp(t->dtype, "F16") && t->shape[0] == in) {
        for (size_t i = 0; i < in; i++)
            suh[i] = oc_oxk_f16_le_to_f32((const uint8_t *)sd + i * 2);
        m.suh = suh;
    }
    snprintf(name, sizeof(name), "%s.svh", key);
    sd = NULL;
    t = src_find(s, name, &sd);
    if (t && !strcmp(t->dtype, "F16") && t->shape[0] == out) {
        for (size_t i = 0; i < out; i++)
            svh[i] = oc_oxk_f16_le_to_f32((const uint8_t *)sd + i * 2);
        m.svh = svh;
    }
    /* identity batch: batch row c is e_c, so the kernel writes column c of
     * W into cols[c*out .. c*out+out) — transpose to row-major after. */
    float *x = calloc(in * in, sizeof(float));
    float *cols = malloc(in * out * sizeof(float));
    if (!x || !cols) die("oom");
    for (size_t c = 0; c < in; c++) x[c * in + c] = 1.0f;
    OcError mm = oc_exl3_matmul(&m, x, in, cols, out, in);
    if (mm != OC_OK) die("matmul %s rc=%d", key, (int)mm);
    fprintf(stderr, "  dequant %s matmul ok\n", key);
    free(x);
    fprintf(stderr, "  free x ok\n");
    float *w = malloc(out * in * sizeof(float));
    if (!w) die("oom");
    for (size_t r = 0; r < out; r++)
        for (size_t c = 0; c < in; c++)
            w[r * in + c] = cols[c * out + r];
    free(cols);
    free(suh);
    free(svh);
    return w;
}

/* raw numeric vector copy: any width → F32, no semantic folds. */
static float *src_vector(Src *s, const char *name, size_t n)
{
    const void *data = NULL;
    const OcSafetensorsTensor *t = src_find(s, name, &data);
    if (!t) die("missing %s", name);
    float *v = malloc(n * sizeof(float));
    if (!v) die("oom");
    const uint8_t *p = data;
    for (size_t i = 0; i < n; i++) {
        if (!strcmp(t->dtype, "F32"))
            memcpy(&v[i], p + i * 4, 4);
        else if (!strcmp(t->dtype, "F16"))
            v[i] = oc_oxk_f16_le_to_f32(p + i * 2);
        else if (!strcmp(t->dtype, "BF16")) {
            uint32_t bits = ((uint32_t)p[i * 2] |
                             (uint32_t)p[i * 2 + 1] << 8) << 16;
            memcpy(&v[i], &bits, 4);
        } else
            die("dtype %s for %s", t->dtype, name);
    }
    return v;
}

static void put_f32(Stw *w, const char *name, const size_t *dims, size_t nd,
                    const float *v, size_t n)
{
    stw_add(w, name, "F32", dims, nd, v, n * 4);
}

static void emit_config_and_tokenizer(const char *src_dir,
                                      const char *out_dir);

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <src_dir> <out_dir>\n", argv[0]);
        return 1;
    }
    Src s;
    src_open(&s, argv[1]);
    static Stw w0;
    stw_init(&w0);
    Stw *w = &w0;
    char name[640], key[512];
    size_t d2[2], d3[3];

    /* embeddings + head */
    d2[0] = VOCAB;
    d2[1] = D;
    {
        float *v = src_vector(&s, "model.language_model.embed_tokens.weight",
                              (size_t)VOCAB * D);
        put_f32(w, "model.language_model.embed_tokens.weight", d2, 2, v,
                (size_t)VOCAB * D);
        free(v);
    }
    {
        float *v = src_dequant(&s, "lm_head", D, VOCAB);
        put_f32(w, "lm_head.weight", d2, 2, v, (size_t)VOCAB * D);
        free(v);
    }
    /* final hyper-connection head */
    static const struct {
        const char *leaf;
        size_t r, c;
    } mixer[] = {
        {"hyper_connection_mixer.hc_norm.weight", HCD, 1},
        {"hyper_connection_mixer.input_mix_weight_down.weight", HCLR, HCD},
        {"hyper_connection_mixer.input_mix_weight_up.weight", HCD, HCLR},
    };
    for (size_t i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "model.language_model.%s", mixer[i].leaf);
        size_t n = mixer[i].r * mixer[i].c;
        float *v = src_vector(&s, name, n);
        if (mixer[i].c == 1) {
            size_t d1[1] = {n};
            put_f32(w, name, d1, 1, v, n);
        } else {
            d2[0] = mixer[i].r;
            d2[1] = mixer[i].c;
            put_f32(w, name, d2, 2, v, n);
        }
        free(v);
    }

    for (size_t l = 0; l < NLAYERS; l++) {
        /* hyper-connection blocks */
        static const char *const sides[2] = {"attn_hyper_connection",
                                             "mlp_hyper_connection"};
        for (size_t side = 0; side < 2; side++) {
            static const struct {
                const char *leaf;
                size_t r, c;
            } hcpart[] = {
                {"hc_norm.weight", HCD, 1},
                {"input_mix_weight_down.weight", HCLR, HCD},
                {"input_mix_weight_up.weight", HCD, HCLR},
                {"block_inject_weight.weight", HC, HCD},
            };
            for (size_t i = 0; i < 4; i++) {
                snprintf(name, sizeof(name),
                         "model.language_model.layers.%zu.%s.%s", l,
                         sides[side], hcpart[i].leaf);
                size_t n = hcpart[i].r * hcpart[i].c;
                float *v = src_vector(&s, name, n);
                if (hcpart[i].c == 1) {
                    size_t d1[1] = {n};
                    put_f32(w, name, d1, 1, v, n);
                } else {
                    d2[0] = hcpart[i].r;
                    d2[1] = hcpart[i].c;
                    put_f32(w, name, d2, 2, v, n);
                }
                free(v);
            }
        }

        /* MoE: router (truncated), experts, shared expert */
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%zu.mlp.gate.weight", l);
        {
            const void *data = NULL;
            const OcSafetensorsTensor *t = src_find(&s, name, &data);
            if (!t || strcmp(t->dtype, "F16") || t->n_dims != 2 ||
                t->shape[0] != 512u || t->shape[1] != D)
                die("bad router %s", name);
            float *v = malloc((size_t)NEXP * D * 4);
            if (!v) die("oom");
            for (size_t r = 0; r < NEXP; r++)
                for (size_t c2 = 0; c2 < D; c2++)
                    v[r * D + c2] =
                        oc_oxk_f16_le_to_f32((const uint8_t *)data +
                                             (r * D + c2) * 2);
            d2[0] = NEXP;
            d2[1] = D;
            put_f32(w, name, d2, 2, v, (size_t)NEXP * D);
            free(v);
        }
        for (size_t ex = 0; ex < NEXP; ex++) {
            static const char *const proj[3] = {"gate_proj", "up_proj",
                                                "down_proj"};
            for (size_t p = 0; p < 3; p++) {
                size_t in = p == 2 ? ISIZE : D;
                size_t out = p == 2 ? D : ISIZE;
                snprintf(key, sizeof(key),
                         "model.language_model.layers.%zu.mlp.experts.%zu.%s",
                         l, ex, proj[p]);
                float *v = src_dequant(&s, key, in, out);
                snprintf(name, sizeof(name), "%s.weight", key);
                d2[0] = out;
                d2[1] = in;
                put_f32(w, name, d2, 2, v, out * in);
                free(v);
            }
        }
        snprintf(key, sizeof(key),
                 "model.language_model.layers.%zu.mlp.shared_expert", l);
        static const char *const proj[3] = {"gate_proj", "up_proj",
                                            "down_proj"};
        for (size_t p = 0; p < 3; p++) {
            size_t in = p == 2 ? ISIZE : D;
            size_t out = p == 2 ? D : ISIZE;
            snprintf(name, sizeof(name), "%s.%s", key, proj[p]);
            float *v = src_dequant(&s, name, in, out);
            size_t nm = strlen(name);
            snprintf(name + nm, sizeof(name) - nm, ".weight");
            d2[0] = out;
            d2[1] = in;
            put_f32(w, name, d2, 2, v, out * in);
            free(v);
        }
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%zu.mlp.shared_expert_gate.weight",
                 l);
        {
            float *v = src_vector(&s, name, D);
            d2[0] = 1;
            d2[1] = D;
            put_f32(w, name, d2, 2, v, D);
            free(v);
        }

        if ((l + 1) % 4 != 0) {
            /* gated delta-net (full attention every 4th layer) */
            snprintf(key, sizeof(key),
                     "model.language_model.layers.%zu.linear_attn", l);
            static const struct {
                const char *proj;
                size_t in, out;
            } mats[] = {
                {"in_proj_qkv", D, CONVDIM},
                {"in_proj_z", D, SINNER},
                {"out_proj", SINNER, D},
            };
            for (size_t i = 0; i < 3; i++) {
                snprintf(name, sizeof(name), "%s.%s", key, mats[i].proj);
                float *v = src_dequant(&s, name, mats[i].in, mats[i].out);
                size_t nm = strlen(name);
                snprintf(name + nm, sizeof(name) - nm, ".weight");
                d2[0] = mats[i].out;
                d2[1] = mats[i].in;
                put_f32(w, name, d2, 2, v, mats[i].out * mats[i].in);
                free(v);
            }
            static const struct {
                const char *leaf;
                size_t n;
            } vecs[] = {
                {"A_log", 48}, {"dt_bias", 48},
                {"in_proj_a.weight", SINNER * 0 + 48u * D},
                {"in_proj_b.weight", 48u * D},
                {"norm.weight", 128},
            };
            for (size_t i = 0; i < 5; i++) {
                snprintf(name, sizeof(name), "%s.%s", key, vecs[i].leaf);
                float *v = src_vector(&s, name, vecs[i].n);
                if (i == 2 || i == 3) {
                    d2[0] = 48;
                    d2[1] = D;
                    put_f32(w, name, d2, 2, v, vecs[i].n);
                } else {
                    size_t d1[1] = {vecs[i].n};
                    put_f32(w, name, d1, 1, v, vecs[i].n);
                }
                free(v);
            }
            /* conv1d [conv_dim, 1, kernel] */
            snprintf(name, sizeof(name), "%s.conv1d.weight", key);
            {
                const void *data = NULL;
                const OcSafetensorsTensor *t =
                    src_find(&s, name, &data);
                if (!t || strcmp(t->dtype, "BF16") || t->n_dims != 3 ||
                    t->shape[0] != CONVDIM || t->shape[1] != 1 ||
                    t->shape[2] != 4)
                    die("bad conv %s", name);
                size_t n = CONVDIM * 4;
                float *v = malloc(n * 4);
                if (!v) die("oom");
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits = ((uint32_t)((const uint16_t *)data)[i])
                                    << 16;
                    memcpy(&v[i], &bits, 4);
                }
                d3[0] = CONVDIM;
                d3[1] = 1;
                d3[2] = 4;
                put_f32(w, name, d3, 3, v, n);
                free(v);
            }
        } else {
            /* full attention */
            snprintf(key, sizeof(key),
                     "model.language_model.layers.%zu.self_attn", l);
            static const struct {
                const char *proj;
                size_t in, out;
            } fmats[] = {
                {"q_proj", D, 2 * QROWS},
                {"k_proj", D, KVROWS},
                {"v_proj", D, KVROWS},
                {"o_proj", QROWS, D},
            };
            for (size_t i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "%s.%s", key, fmats[i].proj);
                float *v = src_dequant(&s, name, fmats[i].in, fmats[i].out);
                size_t nm = strlen(name);
                snprintf(name + nm, sizeof(name) - nm, ".weight");
                d2[0] = fmats[i].out;
                d2[1] = fmats[i].in;
                put_f32(w, name, d2, 2, v, fmats[i].out * fmats[i].in);
                free(v);
            }
            static const char *const qk[2] = {"q_norm.weight",
                                              "k_norm.weight"};
            for (size_t i = 0; i < 2; i++) {
                snprintf(name, sizeof(name), "%s.%s", key, qk[i]);
                float *v = src_vector(&s, name, 256);
                size_t d1[1] = {256};
                put_f32(w, name, d1, 1, v, 256);
                free(v);
            }

            /* indexer: skipped entirely — the dump config zeroes the
             * compress ratio so both engines run dense attention */
            (void)0;
        }
    }

    char out[1024];
    snprintf(out, sizeof(out), "%s/model.safetensors", argv[2]);
    stw_write(w, out);
    fprintf(stderr, "wrote %s (%zu tensors, %zu bytes payload)\n", out,
            w->ntensors, w->pn);
    emit_config_and_tokenizer(argv[1], argv[2]);
    return 0;
}

/* ── adjusted config.json + tokenizer emission ───────────────────────────
 * The weights above are truncated (NLAYERS/NEXP) and the indexer tensors
 * are dropped, so the source config.json is wrong for the dump: emit one
 * whose expert/layer counts match and whose indexer compression is zeroed
 * (dense attention), then copy the tokenizer files so the output dir is a
 * self-contained HF checkpoint the converter can consume directly. */

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); die("oom"); }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;
    if (len) *len = (size_t)n;
    return buf;
}

static void write_file(const char *path, const char *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write %s", path);
    if (len && fwrite(buf, 1, len, f) != len) die("short write %s", path);
    fclose(f);
}

/* Replace the integer value of `"field": N` in place. */
static void rewrite_int_field(char *cfg, const char *field, unsigned value)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", field);
    char *p = strstr(cfg, pat);
    if (!p) return;                 /* field absent in source config */
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    char num[32];
    snprintf(num, sizeof(num), "%u", value);
    char *e = p;
    while (*e >= '0' && *e <= '9') e++;
    if (e == p) return;             /* not an integer field: leave it */
    const size_t tail = strlen(e);
    memmove(p + strlen(num), e, tail + 1);
    memcpy(p, num, strlen(num));
}

/* Truncate a JSON string array ("field": ["a","b",...]) to `keep`
 * entries in place; no-op if it has `keep` or fewer entries. */
static void truncate_str_array(char *cfg, const char *field, unsigned keep)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", field);
    char *p = strstr(cfg, pat);
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    char *q = p + 1;
    unsigned seen = 0;
    while (*q && *q != ']') {
        if (*q == '"') {
            if (seen == keep) {     /* drop this entry and everything after */
                /* walk back over the preceding comma + whitespace */
                char *cut = q;
                while (cut > p + 1 && (cut[-1] == ' ' || cut[-1] == ',')) cut--;
                char *close = strchr(q, ']');
                if (!close) return;
                memmove(cut, close, strlen(close) + 1);
                return;
            }
            seen++;
            q = strchr(q + 1, '"');
            if (!q) return;
        }
        q++;
    }
}

static void copy_file(const char *src_dir, const char *out_dir,
                      const char *name)
{
    char sp[1024], op[1024];
    snprintf(sp, sizeof(sp), "%s/%s", src_dir, name);
    size_t len = 0;
    char *buf = read_file(sp, &len);
    if (!buf) return;               /* optional file absent in source */
    snprintf(op, sizeof(op), "%s/%s", out_dir, name);
    write_file(op, buf, len);
    fprintf(stderr, "copied %s\n", op);
    free(buf);
}

static void emit_config_and_tokenizer(const char *src_dir,
                                      const char *out_dir)
{
    char cfg_path[1024], out_path[1024];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", src_dir);
    size_t len = 0;
    char *cfg = read_file(cfg_path, &len);
    if (!cfg) die("cannot read %s", cfg_path);

    rewrite_int_field(cfg, "num_experts", NEXP);
    rewrite_int_field(cfg, "num_local_experts", NEXP);
    rewrite_int_field(cfg, "n_routed_experts", NEXP);
    rewrite_int_field(cfg, "moe_num_experts", NEXP);
    rewrite_int_field(cfg, "num_hidden_layers", NLAYERS);
    /* The dump carries no indexer tensors: zero the indexer compression
     * fields so consumers run exact dense attention. */
    static const char *const idx_fields[] = {
        "indexer_top_k", "indexer_topk", "index_top_k", "index_topk",
        "indexer_n_heads", "indexer_head_dim", "qsa_indexer_top_k",
    };
    for (size_t i = 0; i < sizeof(idx_fields) / sizeof(idx_fields[0]); i++)
        rewrite_int_field(cfg, idx_fields[i], 0);
    truncate_str_array(cfg, "layer_types", NLAYERS);
    truncate_str_array(cfg, "layers_block_type", NLAYERS);

    snprintf(out_path, sizeof(out_path), "%s/config.json", out_dir);
    write_file(out_path, cfg, strlen(cfg));
    fprintf(stderr, "wrote %s (layers=%u experts=%u, indexer zeroed)\n",
            out_path, NLAYERS, NEXP);
    free(cfg);

    static const char *const tok_files[] = {
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "generation_config.json",
    };
    for (size_t i = 0; i < sizeof(tok_files) / sizeof(tok_files[0]); i++)
        copy_file(src_dir, out_dir, tok_files[i]);
}
