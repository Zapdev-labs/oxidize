/* Byte-pair tokenizer. Train on a UTF-8 corpus, encode to uint16 tokens.
 *
 *   bpe train <corpus.txt> <vocab.bin> [vocab_size]
 *   bpe encode <corpus.txt> <vocab.bin> <out.bin>
 *
 * vocab.bin: uint32 n, then n records of (uint16 left, uint16 right, uint16 id)
 * for merges id>=256. Bytes 0-255 are identity. 256=BOS 257=EOS.
 * out.bin: uint32 ntok, then ntok uint16 ids.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { BOS = 256, EOS = 257, FIRST_MERGE = 258 };

static void die(const char *m) {
    fprintf(stderr, "%s\n", m);
    exit(1);
}

static uint8_t *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open corpus");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 1) die("empty corpus");
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)sz);
    if (!b) die("oom");
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) die("read");
    fclose(f);
    *n = (size_t)sz;
    return b;
}

typedef struct {
    uint16_t left, right, id;
} Merge;

static void save_vocab(const char *path, Merge *m, int nmerge) {
    FILE *f = fopen(path, "wb");
    if (!f) die("open vocab");
    uint32_t n = (uint32_t)nmerge;
    fwrite(&n, 4, 1, f);
    fwrite(m, sizeof(Merge), (size_t)nmerge, f);
    fclose(f);
}

static Merge *load_vocab(const char *path, int *nmerge) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open vocab");
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) die("vocab header");
    Merge *m = calloc(n ? n : 1, sizeof(Merge));
    if (n && fread(m, sizeof(Merge), n, f) != n) die("vocab body");
    fclose(f);
    *nmerge = (int)n;
    return m;
}

/* Apply merges to a mutable uint16 sequence. Returns new length. */
static int apply_merges(uint16_t *s, int n, const Merge *merges, int nmerge) {
    for (int mi = 0; mi < nmerge; mi++) {
        uint16_t a = merges[mi].left, b = merges[mi].right, id = merges[mi].id;
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && s[i] == a && s[i + 1] == b) {
                s[w++] = id;
                i++;
            } else {
                s[w++] = s[i];
            }
        }
        n = w;
    }
    return n;
}

static int train_cmd(int argc, char **argv) {
    const char *corpus = argv[0];
    const char *out = argv[1];
    int vocab = argc > 2 ? atoi(argv[2]) : 4096;
    if (vocab < FIRST_MERGE + 32) vocab = 512;
    size_t nbytes = 0;
    uint8_t *raw = read_all(corpus, &nbytes);
    /* start as bytes */
    int n = (int)nbytes;
    if (n > 8 * 1024 * 1024) n = 8 * 1024 * 1024; /* cap train slice */
    uint16_t *s = malloc((size_t)n * sizeof(uint16_t));
    if (!s) die("oom seq");
    for (int i = 0; i < n; i++) s[i] = raw[i];
    free(raw);

    int nmerge = vocab - FIRST_MERGE;
    Merge *merges = calloc((size_t)nmerge, sizeof(Merge));
    if (!merges) die("oom merges");

    typedef struct {
        uint16_t a, b;
        uint32_t c;
    } Buck;
    const int nb = 1 << 20;
    Buck *bk = calloc((size_t)nb, sizeof(Buck));
    if (!bk) die("oom buck");

    for (int m = 0; m < nmerge; m++) {
        memset(bk, 0, (size_t)nb * sizeof(Buck));
        uint32_t best_c = 0;
        uint16_t best_a = 0, best_b = 0;
        for (int i = 0; i + 1 < n; i++) {
            uint16_t a = s[i], b = s[i + 1];
            uint32_t h = ((uint32_t)a * 2246822519u) ^ ((uint32_t)b * 3266489917u);
            int j = (int)(h & (uint32_t)(nb - 1));
            for (;;) {
                if (bk[j].c == 0) {
                    bk[j].a = a;
                    bk[j].b = b;
                    bk[j].c = 1;
                    if (1 > best_c) {
                        best_c = 1;
                        best_a = a;
                        best_b = b;
                    }
                    break;
                }
                if (bk[j].a == a && bk[j].b == b) {
                    uint32_t c = ++bk[j].c;
                    if (c > best_c) {
                        best_c = c;
                        best_a = a;
                        best_b = b;
                    }
                    break;
                }
                j = (j + 1) & (nb - 1);
            }
        }
        if (best_c < 2) {
            nmerge = m;
            break;
        }
        uint16_t id = (uint16_t)(FIRST_MERGE + m);
        merges[m] = (Merge){best_a, best_b, id};
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && s[i] == best_a && s[i + 1] == best_b) {
                s[w++] = id;
                i++;
            } else {
                s[w++] = s[i];
            }
        }
        n = w;
        if ((m + 1) % 128 == 0) {
            fprintf(stderr, "merge %d/%d pair=%u+%u count=%u seq=%d\n",
                    m + 1, nmerge, best_a, best_b, best_c, n);
        }
    }
    save_vocab(out, merges, nmerge);
    fprintf(stderr, "wrote %d merges to %s final_seq=%d\n", nmerge, out, n);
    free(bk);
    free(s);
    free(merges);
    return 0;
}

static int encode_cmd(int argc, char **argv) {
    if (argc < 3) die("bpe encode corpus vocab out");
    size_t nbytes = 0;
    uint8_t *raw = read_all(argv[0], &nbytes);
    int nmerge = 0;
    Merge *merges = load_vocab(argv[1], &nmerge);
    /* encode line by line with EOS */
    size_t cap = nbytes + nbytes / 8 + 1024;
    uint16_t *out = malloc(cap * 2);
    if (!out) die("oom out");
    uint32_t ntok = 0;
    size_t i = 0;
    uint16_t line[8192];
    while (i < nbytes) {
        int ln = 0;
        line[ln++] = BOS;
        while (i < nbytes && raw[i] != '\n' && ln < 8000) {
            line[ln++] = raw[i++];
        }
        if (i < nbytes && raw[i] == '\n') i++;
        line[ln++] = EOS;
        ln = apply_merges(line, ln, merges, nmerge);
        if (ntok + (uint32_t)ln > cap) die("out cap");
        memcpy(out + ntok, line, (size_t)ln * 2);
        ntok += (uint32_t)ln;
    }
    FILE *f = fopen(argv[2], "wb");
    if (!f) die("open out");
    fwrite(&ntok, 4, 1, f);
    fwrite(out, 2, ntok, f);
    fclose(f);
    fprintf(stderr, "encoded %u tokens\n", ntok);
    free(raw);
    free(merges);
    free(out);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bpe train|encode ...\n");
        return 1;
    }
    if (!strcmp(argv[1], "train")) return train_cmd(argc - 2, argv + 2);
    if (!strcmp(argv[1], "encode")) return encode_cmd(argc - 2, argv + 2);
    die("unknown cmd");
    return 1;
}
