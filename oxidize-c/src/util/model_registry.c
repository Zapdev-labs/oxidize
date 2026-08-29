#define _POSIX_C_SOURCE 200809L

#include "oxidize/model_registry.h"
#include "oxidize/gguf.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <strings.h>
#include <stdarg.h>


/* Case-insensitive substring search: returns true if `needle` occurs
 * anywhere in `haystack` (case-insensitively). Both must be NUL-terminated. */
static bool ci_substr(const char *haystack, const char *needle)
{
    if (!haystack || !needle) return false;
    if (!*needle) return true;
    for (const char *h = haystack; *h; h++) {
        size_t i = 0;
        while (needle[i] &&
               tolower((unsigned char)h[i]) == tolower((unsigned char)needle[i])) {
            i++;
        }
        if (!needle[i]) return true;
    }
    return false;
}

/* Levenshtein distance between two NUL-terminated strings. Uses a stack
 * scratch buffer (rows of size len_b+1); returns SIZE_MAX on overflow or
 * NULL inputs. Bounded by 256 to keep the scratch buffer small. */
static size_t levenshtein(const char *a, const char *b)
{
    if (!a || !b) return SIZE_MAX;
    size_t la = strlen(a), lb = strlen(b);
    if (la > 256 || lb > 256) return SIZE_MAX;
    if (la == 0) return lb;
    if (lb == 0) return la;

    size_t prev[257], cur[257];
    for (size_t j = 0; j <= lb; j++) prev[j] = j;

    for (size_t i = 1; i <= la; i++) {
        cur[0] = i;
        for (size_t j = 1; j <= lb; j++) {
            size_t cost = (tolower((unsigned char)a[i - 1]) ==
                           tolower((unsigned char)b[j - 1])) ? 0 : 1;
            size_t del = prev[j] + 1;
            size_t ins = cur[j - 1] + 1;
            size_t sub = prev[j - 1] + cost;
            size_t m = del < ins ? del : ins;
            if (sub < m) m = sub;
            cur[j] = m;
        }
        memcpy(prev, cur, (lb + 1) * sizeof(size_t));
    }
    return prev[lb];
}

/* Strip directory and trailing ".gguf" (case-insensitive) from `path` into
 * `out` (cap bytes). Returns true on success. */
static bool basename_no_gguf(const char *path, char *out, size_t cap)
{
    if (!path || !out || cap == 0) return false;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    size_t len = strlen(base);
    const char *gguf_ext = ".gguf";
    size_t ext_len = strlen(gguf_ext);
    if (len >= ext_len) {
        size_t off = len - ext_len;
        bool match = true;
        for (size_t i = 0; i < ext_len; i++) {
            if (tolower((unsigned char)base[off + i]) !=
                tolower((unsigned char)gguf_ext[i])) {
                match = false;
                break;
            }
        }
        if (match) len = off;
    }
    if (len + 1 > cap) len = cap - 1;
    memcpy(out, base, len);
    out[len] = '\0';
    return true;
}

/* Extract the GGUF metadata "quant_type" string. */
/* otherwise fall back to "general.file_type" mapped to a short name, and */
static void extract_quant_type(const OcGgufFile *f, char *out, size_t cap)
{
    if (!out || cap == 0) return;
    out[0] = '\0';

    const char *qs = NULL;
    size_t qlen = 0;
    if (oc_gguf_metadata_get_str(f, "general.quantization_type", &qs, &qlen) && qs) {
        if (qlen >= cap) qlen = cap - 1;
        memcpy(out, qs, qlen);
        out[qlen] = '\0';
        return;
    }

    /* general.file_type is a u32 with values matching ggml_type. Map the
     * common ones to short names; unknown values get "F<ggml_type>". */
    uint32_t ft = 0;
    if (oc_gguf_metadata_get_u32(f, "general.file_type", &ft)) {
        const char *name = "unknown";
        switch (ft) {
            case 0:  name = "F32";       break;
            case 1:  name = "F16";       break;
            case 2:  name = "Q4_0";      break;
            case 3:  name = "Q4_1";      break;
            case 7:  name = "Q8_0";      break;
            case 8:  name = "Q5_0";      break;
            case 9:  name = "Q5_1";      break;
            case 10: name = "Q2_K";      break;
            case 11: name = "Q3_K_S";    break;
            case 12: name = "Q3_K_M";    break;
            case 13: name = "Q3_K_L";    break;
            case 14: name = "Q4_K_S";    break;
            case 15: name = "Q4_K_M";    break;
            case 16: name = "Q5_K_S";    break;
            case 17: name = "Q5_K_M";    break;
            case 18: name = "Q6_K";      break;
            default: snprintf(out, cap, "F%u", ft); return;
        }
        snprintf(out, cap, "%s", name);
        return;
    }
    snprintf(out, cap, "unknown");
}

/* Parse the GGUF header of `path` and populate the entry's arch + counts.
 * GGUF parse failures are NOT fatal: the entry is left with zeros. */
static void parse_gguf_metadata(OcModelEntry *e)
{
    OcGgufFile f;
    if (oc_gguf_open(e->path, &f) != OC_OK) {
        e->arch = OC_ARCH_UNKNOWN;
        return;
    }

    e->arch = oc_gguf_arch_from_file(&f);

    uint32_t u32 = 0;
    if (oc_gguf_metadata_get_u32(&f, "llama.block_count", &u32) ||
        oc_gguf_metadata_get_u32(&f, "general.block_count", &u32)) {
        e->n_layers = u32;
    }
    if (oc_gguf_metadata_get_u32(&f, "llama.embedding_length", &u32) ||
        oc_gguf_metadata_get_u32(&f, "general.embedding_length", &u32)) {
        e->n_embd = u32;
    }

    uint64_t u64 = 0;
    if (oc_gguf_metadata_get_u64(&f, "general.parameter_count", &u64)) {
        e->n_params = u64;
    }

    /* vocab size: prefer tokenizer.ggml.tokens array length if available;
     * otherwise fall back to the common metadata key. */
    const char *toks = NULL;
    size_t toks_len = 0;
    if (oc_gguf_metadata_get_str(&f, "tokenizer.ggml.tokens", &toks, &toks_len)) {
        /* The value is an array of strings; the array length is in the raw metadata. */
        (void)toks; (void)toks_len;
    }
    if (oc_gguf_metadata_get_u32(&f, "tokenizer.ggml.n_tokens", &u32) ||
        oc_gguf_metadata_get_u32(&f, "llama.vocab_size", &u32)) {
        e->vocab_size = u32;
    }

    extract_quant_type(&f, e->quant_type, sizeof(e->quant_type));

    oc_gguf_free(&f);
}

/* Find an entry by exact path match. Returns the index or SIZE_MAX. */
static size_t find_by_path(const OcModelRegistry *reg, const char *path)
{
    if (!reg || !path) return SIZE_MAX;
    for (size_t i = 0; i < reg->count; i++) {
        if (strcmp(reg->entries[i].path, path) == 0) return i;
    }
    return SIZE_MAX;
}

/* Grow the entries array by one slot if capacity allows. Returns true if a
 * new slot is available (either by growing or reusing an existing empty
 * slot). Sets *idx to the slot index. */
static bool reserve_slot(OcModelRegistry *reg, size_t *idx)
{
    if (reg->count >= reg->cap) {
        size_t newcap = reg->cap ? reg->cap * 2 : 8;
        if (newcap > reg->max_entries) newcap = reg->max_entries;
        if (newcap <= reg->cap) return false;  /* at cap */
        OcModelEntry *p = realloc(reg->entries, newcap * sizeof(*p));
        if (!p) return false;
        reg->entries = p;
        reg->cap = newcap;
    }
    *idx = reg->count++;
    return true;
}


OcError oc_model_registry_init(OcModelRegistry *reg, const char *cache_dir,
                               size_t max_entries)
{
    if (!reg) return OC_ERR_INVALID_ARG;
    memset(reg, 0, sizeof(*reg));
    if (max_entries == 0) max_entries = OC_MODEL_REGISTRY_MAX_ENTRIES;
    if (max_entries > OC_MODEL_REGISTRY_MAX_ENTRIES)
        max_entries = OC_MODEL_REGISTRY_MAX_ENTRIES;
    reg->max_entries = max_entries;
    if (cache_dir) {
        snprintf(reg->cache_dir, sizeof(reg->cache_dir), "%s", cache_dir);
    }
    /* Defer allocation of the entries array until the first add/scan so an
     * unused registry costs zero heap. */
    reg->entries = NULL;
    reg->count = 0;
    reg->cap = 0;
    return OC_OK;
}

void oc_model_registry_free(OcModelRegistry *reg)
{
    if (!reg) return;
    free(reg->entries);
    reg->entries = NULL;
    reg->count = 0;
    reg->cap = 0;
}

OcError oc_model_registry_add(OcModelRegistry *reg, const char *path)
{
    if (!reg || !path) return OC_ERR_INVALID_ARG;
    if (strlen(path) >= OC_MODEL_REGISTRY_MAX_PATH) return OC_ERR_INVALID_ARG;

    /* Update in place if the path already exists. */
    size_t existing = find_by_path(reg, path);
    OcModelEntry *e;
    if (existing != SIZE_MAX) {
        e = &reg->entries[existing];
    } else {
        size_t idx;
        if (!reserve_slot(reg, &idx)) return OC_ERR_OOM;
        e = &reg->entries[idx];
        memset(e, 0, sizeof(*e));
        snprintf(e->path, sizeof(e->path), "%s", path);
    }

    /* Populate metadata. */
    struct stat st;
    if (stat(path, &st) == 0) {
        e->size_bytes = (uint64_t)st.st_size;
    } else {
        e->size_bytes = 0;
    }
    e->loaded_at = time(NULL);
    basename_no_gguf(path, e->name, sizeof(e->name));
    parse_gguf_metadata(e);
    return OC_OK;
}

OcError oc_model_registry_scan(OcModelRegistry *reg, const char *dir)
{
    if (!reg || !dir) return OC_ERR_INVALID_ARG;
    DIR *d = opendir(dir);
    if (!d) return OC_ERR_IO;

    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        const char *name = de->d_name;
        size_t nlen = strlen(name);
        /* The extension is ".gguf" (5 chars); skip shorter names. */
        if (nlen < 5) continue;
        if (!(tolower((unsigned char)name[nlen-5]) == '.' &&
              tolower((unsigned char)name[nlen-4]) == 'g' &&
              tolower((unsigned char)name[nlen-3]) == 'g' &&
              tolower((unsigned char)name[nlen-2]) == 'u' &&
              tolower((unsigned char)name[nlen-1]) == 'f')) {
            continue;
        }
        /* Skip "." and ".." (already filtered by length+ext, but be safe). */
        if (name[0] == '.' && (name[1] == '\0' ||
                               (name[1] == '.' && name[2] == '\0'))) {
            continue;
        }

        char full[OC_MODEL_REGISTRY_MAX_PATH];
        int w = snprintf(full, sizeof(full), "%s/%s", dir, name);
        if (w < 0 || (size_t)w >= sizeof(full)) continue;
        if (reg->count >= reg->max_entries) break;
        /* Errors from add() are non-fatal during scan (one bad file should
         * not abort the whole scan). */
        (void)oc_model_registry_add(reg, full);
    }
    closedir(d);
    return OC_OK;
}

OcError oc_model_registry_remove(OcModelRegistry *reg, const char *path)
{
    if (!reg || !path) return OC_ERR_INVALID_ARG;
    size_t idx = find_by_path(reg, path);
    if (idx == SIZE_MAX) return OC_ERR_INVALID_ARG;
    /* Shift down. */
    if (idx + 1 < reg->count) {
        memmove(&reg->entries[idx], &reg->entries[idx + 1],
                (reg->count - idx - 1) * sizeof(*reg->entries));
    }
    reg->count--;
    return OC_OK;
}

const OcModelEntry *oc_model_registry_find(const OcModelRegistry *reg,
                                           const char *query)
{
    if (!reg || !query || !*query) return NULL;
    /* Pass 1: case-insensitive substring match. */
    const OcModelEntry *best = NULL;
    for (size_t i = 0; i < reg->count; i++) {
        if (ci_substr(reg->entries[i].name, query)) {
            return &reg->entries[i];
        }
    }
    /* Pass 2: smallest Levenshtein distance. */
    size_t best_dist = SIZE_MAX;
    size_t threshold = strlen(query) / 2;
    if (threshold == 0) threshold = 1;
    for (size_t i = 0; i < reg->count; i++) {
        size_t d = levenshtein(reg->entries[i].name, query);
        if (d < best_dist && d <= threshold) {
            best_dist = d;
            best = &reg->entries[i];
        }
    }
    return best;
}

/* Comparator payloads for qsort. */
typedef struct {
    OcModelSortKey key;
} SortCtx;

static int cmp_size(const void *a, const void *b)
{
    const OcModelEntry *ea = a, *eb = b;
    if (ea->size_bytes < eb->size_bytes) return -1;
    if (ea->size_bytes > eb->size_bytes) return 1;
    return 0;
}

static int cmp_name(const void *a, const void *b)
{
    const OcModelEntry *ea = a, *eb = b;
    return strcasecmp(ea->name, eb->name);
}

static int cmp_date(const void *a, const void *b)
{
    const OcModelEntry *ea = a, *eb = b;
    /* newest first → descending */
    if (ea->loaded_at < eb->loaded_at) return 1;
    if (ea->loaded_at > eb->loaded_at) return -1;
    return 0;
}

size_t oc_model_registry_list(const OcModelRegistry *reg,
                              OcModelSortKey key,
                              const OcModelEntry **out, size_t cap)
{
    if (!reg) return 0;
    size_t n = reg->count < cap ? reg->count : cap;

    if (out && n > 0) {
        /* Build a sorted index array without mutating the registry. */
        const OcModelEntry **idx = malloc(reg->count * sizeof(*idx));
        if (!idx) return 0;
        for (size_t i = 0; i < reg->count; i++) idx[i] = &reg->entries[i];

        /* Sort a temporary copy of entries via qsort on a scratch array. */
        OcModelEntry *scratch = malloc(reg->count * sizeof(*scratch));
        if (!scratch) { free(idx); return 0; }
        memcpy(scratch, reg->entries, reg->count * sizeof(*scratch));

        switch (key) {
            case OC_MODEL_SORT_BY_SIZE: qsort(scratch, reg->count, sizeof(*scratch), cmp_size); break;
            case OC_MODEL_SORT_BY_NAME: qsort(scratch, reg->count, sizeof(*scratch), cmp_name); break;
            case OC_MODEL_SORT_BY_DATE: qsort(scratch, reg->count, sizeof(*scratch), cmp_date); break;
        }
        /* Map sorted entries back to registry pointers by pointer-value
         * comparison is not stable; instead we re-point by path match. */
        bool *used = calloc(reg->count, sizeof(*used));
        if (!used) { free(idx); free(scratch); return 0; }
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < reg->count; j++) {
                if (!used[j] && strcmp(reg->entries[j].path, scratch[i].path) == 0) {
                    out[i] = &reg->entries[j];
                    used[j] = true;
                    break;
                }
            }
        }
        free(used);
        free(scratch);
        free(idx);
    }
    return n;
}

/* JSON-escape a string into `buf` (cap bytes). Returns bytes written
 * (excluding NUL), or 0 on overflow. Escapes ", \, control chars. */
static size_t json_escape(const char *s, char *buf, size_t cap)
{
    if (!s) s = "";
    size_t off = 0;
    for (const char *p = s; *p; p++) {
        char esc[8];
        size_t elen = 0;
        unsigned char c = (unsigned char)*p;
        if (c == '"') { esc[elen++] = '\\'; esc[elen++] = '"'; }
        else if (c == '\\') { esc[elen++] = '\\'; esc[elen++] = '\\'; }
        else if (c == '\n') { esc[elen++] = '\\'; esc[elen++] = 'n'; }
        else if (c == '\r') { esc[elen++] = '\\'; esc[elen++] = 'r'; }
        else if (c == '\t') { esc[elen++] = '\\'; esc[elen++] = 't'; }
        else if (c < 0x20) { elen = (size_t)snprintf(esc, sizeof(esc), "\\u%04x", c); }
        else { esc[elen++] = (char)c; }
        if (off + elen + 1 > cap) return 0;
        memcpy(buf + off, esc, elen);
        off += elen;
    }
    buf[off] = '\0';
    return off;
}

/* Append a JSON object field via snprintf into buf at offset *off. */
static bool append_field(char *buf, size_t cap, size_t *off,
                         const char *key, const char *fmt, ...)
{
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    int w = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (w < 0 || (size_t)w >= sizeof(tmp)) return false;
    int w2 = snprintf(buf + *off, cap - *off, "\"%s\":%s,", key, tmp);
    if (w2 < 0 || (size_t)w2 >= cap - *off) return false;
    *off += (size_t)w2;
    return true;
}

static bool append_quoted_field(char *buf, size_t cap, size_t *off,
                                const char *key, const char *val)
{
    char esc[OC_MODEL_REGISTRY_MAX_PATH * 2];
    size_t elen = json_escape(val, esc, sizeof(esc));
    if (elen == 0 && val && *val) return false;
    int w = snprintf(buf + *off, cap - *off, "\"%s\":\"%s\",", key, esc);
    if (w < 0 || (size_t)w >= cap - *off) return false;
    *off += (size_t)w;
    return true;
}

size_t oc_model_registry_format(const OcModelRegistry *reg,
                                char *buf, size_t cap)
{
    if (!reg || !buf || cap == 0) return 0;
    int n = snprintf(buf, cap, "[");
    if (n < 0 || (size_t)n >= cap) return 0;
    size_t off = (size_t)n;

    for (size_t i = 0; i < reg->count; i++) {
        const OcModelEntry *e = &reg->entries[i];
        if (off + 1 >= cap) return 0;
        if (i > 0) buf[off++] = ',';

        int w = snprintf(buf + off, cap - off, "{");
        if (w < 0 || (size_t)w >= cap - off) return 0;
        off += (size_t)w;

        if (!append_quoted_field(buf, cap, &off, "path", e->path)) return 0;
        if (!append_quoted_field(buf, cap, &off, "name", e->name)) return 0;
        if (!append_quoted_field(buf, cap, &off, "arch", oc_model_arch_name(e->arch))) return 0;
        if (!append_quoted_field(buf, cap, &off, "quant_type", e->quant_type)) return 0;
        if (!append_field(buf, cap, &off, "size_bytes", "%llu", (unsigned long long)e->size_bytes)) return 0;
        if (!append_field(buf, cap, &off, "n_params", "%llu", (unsigned long long)e->n_params)) return 0;
        if (!append_field(buf, cap, &off, "n_layers", "%u", e->n_layers)) return 0;
        if (!append_field(buf, cap, &off, "n_embd", "%u", e->n_embd)) return 0;
        if (!append_field(buf, cap, &off, "vocab_size", "%u", e->vocab_size)) return 0;
        /* last field: no trailing comma */
        {
            /* strip trailing comma from prior field */
            if (off > 0 && buf[off - 1] == ',') off--;
            int w2 = snprintf(buf + off, cap - off, ",\"loaded_at\":%lld}",
                              (long long)e->loaded_at);
            if (w2 < 0 || (size_t)w2 >= cap - off) return 0;
            off += (size_t)w2;
        }
    }

    int w = snprintf(buf + off, cap - off, "]");
    if (w < 0 || (size_t)w >= cap - off) return 0;
    off += (size_t)w;
    return off;
}

void oc_model_registry_stats(const OcModelRegistry *reg,
                             OcModelRegistryStats *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!reg) return;

    out->total_models = reg->count;
    out->total_size = 0;
    for (size_t i = 0; i < reg->count; i++) {
        const OcModelEntry *e = &reg->entries[i];
        out->total_size += e->size_bytes;
        if (e->arch < OC_ARCH__COUNT) {
            out->by_arch[e->arch]++;
        } else {
            out->by_arch[OC_ARCH_UNKNOWN]++;
        }

        /* Increment the per-quant-type count, inserting if new. */
        bool found = false;
        for (size_t j = 0; j < out->n_quant_types; j++) {
            if (strcmp(out->by_quant[j].type, e->quant_type) == 0) {
                out->by_quant[j].count++;
                found = true;
                break;
            }
        }
        if (!found && out->n_quant_types < 32) {
            snprintf(out->by_quant[out->n_quant_types].type,
                     sizeof(out->by_quant[0].type), "%s", e->quant_type);
            out->by_quant[out->n_quant_types].count = 1;
            out->n_quant_types++;
        }
    }
}
