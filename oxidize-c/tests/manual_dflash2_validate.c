/*
 * dflash2_validate.c — compare oxidize-c DFlash2 against the PyTorch
 * reference outputs (/tmp/dflash2_val_*.npy produced by dflash2_ref.py
 * running z-lab/dflash on the real checkpoint).
 *
 * Build: cc -Iinclude -std=c11 -O2 -o /tmp/dflash2_validate \
 *          tests/manual_dflash2_validate.c src/model/dflash2.o \
 *          src/compute/parallel.o src/core/error.o src/core/log.o \
 *          src/format/safetensors.o src/core/bytes.o src/util/mmap.o \
 *          src/util/file.o src/core/dtype.o -lm -lpthread
 *
 * (Not wired into `make test`: one-off cross-validation tool.)
 */
#include "oxidize/dflash2.h"
#include "oxidize/parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal .npy (v1/v2) reader for float32 / int64 arrays. */
static void *npy_load(const char *path, char *dtype, size_t *n_elems,
                      int *fortran_order)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hlen;
    if (ver[0] == 1) {
        uint8_t hb[2];
        if (fread(hb, 1, 2, f) != 2) { fclose(f); return NULL; }
        hlen = (uint16_t)(hb[0] | (hb[1] << 8));
    } else {
        uint8_t hb[4];
        if (fread(hb, 1, 4, f) != 4) { fclose(f); return NULL; }
        hlen = (uint16_t)(hb[0] | (hb[1] << 8) | (hb[2] << 16) | (hb[3] << 24));
    }
    char *hdr = malloc((size_t)hlen + 1);
    if (!hdr) { fclose(f); return NULL; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';

    char *dd = strstr(hdr, "'descr'");
    char *fo = strstr(hdr, "'fortran_order'");
    char *sh = strstr(hdr, "'shape'");
    if (!dd || !fo || !sh) { free(hdr); fclose(f); return NULL; }
    *fortran_order = strstr(fo, "True") != NULL;

    const char *val = strchr(dd + 7, '\'');
    if (!val) { free(hdr); fclose(f); return NULL; }
    val++;
    const char *vend = strchr(val, '\'');
    size_t dlen = (size_t)(vend - val);
    if (dlen > 7) dlen = 7;
    memcpy(dtype, val, dlen);
    dtype[dlen] = '\0';

    size_t shape[8], nd = 0;
    char *p = strchr(sh, '(');
    if (!p) { free(hdr); fclose(f); return NULL; }
    p++;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        if (nd >= 8) break;
        shape[nd++] = (size_t)strtol(p, &p, 10);
    }
    size_t total = 1;
    for (size_t i = 0; i < nd; i++) total *= shape[i];
    *n_elems = total;

    size_t elem_sz = strstr(dtype, "f4") ? 4
                        : (strstr(dtype, "i8") ? 8 : 0);
    if (elem_sz == 0) { free(hdr); fclose(f); return NULL; }

    void *data = malloc(total * elem_sz);
    if (!data) { free(hdr); fclose(f); return NULL; }
    if (fread(data, elem_sz, total, f) != total) {
        free(data); free(hdr); fclose(f); return NULL;
    }
    free(hdr);
    fclose(f);
    return data;
}

int main(int argc, char **argv)
{
    const char *ckpt_dir =
        "/home/dih/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a";
    const char *val_dir = argc > 1 ? argv[1] : "/tmp";
    char path[512], dtype[8];

    oc_parallel_set_threads(1); /* determinism */

    OcDFlash2Model m;
    snprintf(path, sizeof(path), "%s/model.safetensors", ckpt_dir);
    OcError e = oc_dflash2_model_load(&m, path, NULL);
    if (e != OC_OK) {
        fprintf(stderr, "load failed: %s\n", oc_error_msg(e));
        return 1;
    }
    printf("loaded draft (5 layers, hidden %zu, vocab %zu)\n",
           m.cfg.hidden_size, m.cfg.vocab_size);

    /* Load reference artifacts. */
    snprintf(path, sizeof(path), "%s/dflash2_val_noise.npy", val_dir);
    int fo;
    size_t n_noise;
    float *noise = npy_load(path, dtype, &n_noise, &fo);    snprintf(path, sizeof(path), "%s/dflash2_val_ctx.npy", val_dir);
    size_t n_ctx;
    float *ctx = npy_load(path, dtype, &n_ctx, &fo);
    snprintf(path, sizeof(path), "%s/dflash2_val_hidden.npy", val_dir);
    size_t n_hidden;
    float *gold_hidden = npy_load(path, dtype, &n_hidden, &fo);
    snprintf(path, sizeof(path), "%s/dflash2_val_lm.npy", val_dir);
    size_t n_lm;
    float *lm = npy_load(path, dtype, &n_lm, &fo);
    snprintf(path, sizeof(path), "%s/dflash2_val_path.npy", val_dir);
    size_t n_path;
    int64_t *gold_path = npy_load(path, dtype, &n_path, &fo);
    snprintf(path, sizeof(path), "%s/dflash2_val_cand.npy", val_dir);
    size_t n_cand;
    int64_t *gold_cand = npy_load(path, dtype, &n_cand, &fo);
    if (!noise || !ctx || !gold_hidden || !lm || !gold_path || !gold_cand) {
        fprintf(stderr, "npy load failed\n");
        return 1;
    }

    const size_t H = m.cfg.hidden_size;
    const size_t block = m.cfg.block_size;
    const size_t top_k = m.cfg.selector_top_k;
    const size_t n_draft = block - 1;
    if (n_noise != block * H || n_ctx != 5 * H || n_hidden != block * H ||
        n_lm != m.cfg.vocab_size * H || n_path != n_draft) {
        fprintf(stderr, "shape mismatch: noise=%zu ctx=%zu hidden=%zu lm=%zu path=%zu\n",
                n_noise, n_ctx, n_hidden, n_lm, n_path);
        return 1;
    }

    /* Context: 1 fused row (n_ctx raw = 20480 = 5*4096). */
    e = oc_dflash2_set_context(&m, ctx, 1);
    if (e != OC_OK) { fprintf(stderr, "set_context failed\n"); return 1; }

    /* Forward debug: next_noise_pos starts at 1 (ctx occupies pos 0). */
    m.next_noise_pos = 1;
    float *out_hidden = malloc(block * H * sizeof(float));
    e = oc_dflash2_forward_debug(&m, noise, block, out_hidden);
    if (e != OC_OK) { fprintf(stderr, "forward failed: %s\n", oc_error_msg(e)); return 1; }

    /* Compare hidden: report max abs diff + relative. */
    double max_abs = 0.0, sum_abs = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < block * H; i++) {
        double d = fabs((double)out_hidden[i] - (double)gold_hidden[i]);
        sum_abs += d;
        if (d > max_abs) { max_abs = d; worst_i = i; }
    }
    double gold_scale = 0.0;
    for (size_t i = 0; i < block * H; i++)
        gold_scale += fabs((double)gold_hidden[i]);
    printf("hidden: max_abs=%.5e mean_abs=%.5e rel=%.5e (worst row %zu dim %zu)\n",
           max_abs, sum_abs / (block * H), max_abs / (gold_scale / (block * H)),
           worst_i / H, worst_i % H);

    int hidden_ok = max_abs < 2e-2; /* BF16 reference tolerance */

    /* Stage-wise: compare against per-layer captures when present.
     * layer{N}.attn = attention output AFTER o_proj (pre-conv-finish),
     * layer{N} = full layer output. */
    for (size_t li = 0; li < 5; li++) {
        char cap[64];
        snprintf(cap, sizeof(cap), "%s/dflash2_val_cap_layer%zu.npy", val_dir, li);
        size_t n_cap;
        float *gold_l = npy_load(cap, dtype, &n_cap, &fo);
        if (!gold_l) { printf("layer%zu: no capture\n", li); break; }
        double m = 0.0;
        double s = 0.0;
        (void)m;
        for (size_t i = 0; i < n_cap && i < block * H; i++) {
            double d = fabs((double)out_hidden[i] - (double)gold_l[i]);
            /* wrong-stage comparison; skipped numerically, just shape info */
            s += d;
        }
        (void)s;
        printf("layer%zu capture: %zu elems (visual check only)\n", li, n_cap);
        free(gold_l);
    }

    /* Selector debug on the reference's draft hidden (rows 1..7 of gold). */
    OcDFlash2Weight lmw = { lm, NULL, m.cfg.vocab_size, H, 0, NULL, NULL };
    uint32_t anchor = 7;
    uint32_t *out_tok = malloc(n_draft * sizeof(uint32_t));
    uint32_t *out_cand = malloc(n_draft * top_k * sizeof(uint32_t));
    /* Use the C forward's hidden (what production would consume). */
    e = oc_dflash2_selector_debug(&m, out_hidden + H, n_draft, &anchor, 1,
                                  &lmw, out_tok, out_cand);
    if (e != OC_OK) { fprintf(stderr, "selector failed\n"); return 1; }

    printf("path gold: ");
    for (size_t i = 0; i < n_draft; i++) printf("%lld ", (long long)gold_path[i]);
    printf("\npath ours: ");
    for (size_t i = 0; i < n_draft; i++) printf("%u ", out_tok[i]);
    printf("\n");
    int path_ok = 1;
    for (size_t i = 0; i < n_draft; i++)
        if ((int64_t)out_tok[i] != gold_path[i]) { path_ok = 0; break; }

    /* Candidate sets: same members regardless of order. */
    int cand_ok = 1;
    for (size_t p = 0; p < n_draft && cand_ok; p++) {
        for (size_t k = 0; k < top_k; k++) {
            int found = 0;
            for (size_t j = 0; j < top_k; j++) {
                if ((uint32_t)gold_cand[p * top_k + j] == out_cand[p * top_k + k]) {
                    found = 1; break;
                }
            }
            if (!found) { cand_ok = 0; break; }
        }
    }
    printf("candidates match: %s\n", cand_ok ? "yes" : "NO");

    printf("%s: hidden %s, path %s, candidates %s\n",
           (hidden_ok && path_ok && cand_ok) ? "PASS" : "FAIL",
           hidden_ok ? "ok" : "MISMATCH",
           path_ok ? "ok" : "MISMATCH",
           cand_ok ? "ok" : "MISMATCH");
    return (hidden_ok && path_ok && cand_ok) ? 0 : 1;
}
