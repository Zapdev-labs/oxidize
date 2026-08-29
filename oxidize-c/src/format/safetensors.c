#include "oxidize/safetensors.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __unix__
#define OC_HAVE_MMAP 1
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#else
#define OC_HAVE_MMAP 0
#endif


typedef enum {
    OC_STJ_TOP,         /* expecting '{' or ',' at top level */
    OC_STJ_TENSOR_KEY,  /* inside a top-level string key (tensor name) */
    OC_STJ_AFTER_KEY,   /* after the tensor-name string, expecting ':' */
    OC_STJ_DESC_OBJ,   /* inside the descriptor object, expecting '{' or ',' */
    OC_STJ_FIELD_KEY,   /* inside a descriptor string key (dtype, shape, ...) */
    OC_STJ_AFTER_FIELD, /* after a descriptor key string, expecting ':' */
    OC_STJ_VALUE,       /* about to read a value (string, array, etc.) */
} OcStJsonState;

/* Parser context. Holds the in-progress tensor descriptor and the current
 * descriptor field name so value handlers know where to store results. */
typedef struct OcStParser {
    OcSafetensorsTensor cur;        /* in-progress tensor descriptor */
    bool                cur_valid;  /* true once a name has been captured */
    char                field[32];  /* current descriptor field name */
    /* shape / data_offsets accumulation buffers */
    uint64_t shape_vals[OC_SAFETENSORS_MAX_DIMS];
    uint32_t shape_count;
    uint64_t offsets[2];
    uint32_t offset_count;
    /* output collection */
    OcSafetensorsTensor *out_arr;
    size_t               out_cap;
    size_t               out_len;
} OcStParser;

/* Append the in-progress tensor descriptor to the parser's output array.
 * Returns false on OOM. */
static bool parser_emit(OcStParser *p)
{
    if (!p->cur_valid) return true;
    if (p->out_len == p->out_cap) {
        size_t new_cap = p->out_cap == 0 ? 8 : p->out_cap * 2;
        OcSafetensorsTensor *na = realloc(p->out_arr, new_cap * sizeof(*na));
        if (na == NULL) return false;
        p->out_arr = na;
        p->out_cap = new_cap;
    }
    /* Copy shape + offsets into the descriptor. */
    p->cur.n_dims = p->shape_count;
    for (uint32_t i = 0; i < p->shape_count && i < OC_SAFETENSORS_MAX_DIMS; i++) {
        p->cur.shape[i] = p->shape_vals[i];
    }
    if (p->offset_count == 2) {
        p->cur.data_offset = p->offsets[0];
        p->cur.data_length = p->offsets[1] - p->offsets[0];
    }
    p->out_arr[p->out_len++] = p->cur;
    /* Reset in-progress state for the next tensor. */
    memset(&p->cur, 0, sizeof(p->cur));
    p->cur_valid = false;
    p->shape_count = 0;
    p->offset_count = 0;
    p->field[0] = '\0';
    return true;
}

/* Copy a JSON string token (without quotes, with escapes unescaped) into a fixed-size NUL-terminated buffer. */
static int json_decode_string(const char *src, const char **endp,
                              char *dst, size_t dst_cap)
{
    if (*src != '"') { *endp = src; return -1; }
    src++;
    size_t i = 0;
    while (*src && *src != '"') {
        char c = *src++;
        if (c == '\\' && *src) {
            char esc = *src++;
            switch (esc) {
            case 'n':  c = '\n'; break;
            case 't':  c = '\t'; break;
            case 'r':  c = '\r'; break;
            case '"':  c = '"';  break;
            case '\\': c = '\\'; break;
            case '/':  c = '/';  break;
            case 'b':  c = '\b'; break;
            case 'f':  c = '\f'; break;
            case 'u': {
                /* Skip 4 hex digits; emit '?' for non-ASCII (tensor names
                 * are ASCII in practice). */
                for (int h = 0; h < 4 && isxdigit((unsigned char)*src); h++) src++;
                c = '?';
                break;
            }
            default: c = esc; break;
            }
        }
        if (i + 1 < dst_cap) dst[i++] = c;
        else { /* truncation: advance to closing quote */ }
    }
    if (dst != NULL && dst_cap > 0) dst[i] = '\0';
    if (*src == '"') src++;
    *endp = src;
    return (int)i;
}

/* Parse a JSON unsigned integer. `*endp` points past the last digit. */
static uint64_t json_decode_uint(const char *src, const char **endp)
{
    uint64_t v = 0;
    while (isdigit((unsigned char)*src)) {
        v = v * 10 + (uint64_t)(*src - '0');
        src++;
    }
    *endp = src;
    return v;
}

/* Skip a JSON value (string, number, array, object, true/false/null) starting
 * at `src`. Sets `*endp` just past the value. Used to skip unknown fields. */
static void json_skip_value(const char *src, const char **endp)
{
    while (*src && isspace((unsigned char)*src)) src++;
    if (*src == '"') {
        const char *e;
        json_decode_string(src, &e, NULL, 0);
        *endp = e;
        return;
    }
    if (*src == '[' || *src == '{') {
        char open_ch = *src;
        char close_ch = (open_ch == '[') ? ']' : '}';
        int depth = 1;
        src++;
        while (*src && depth > 0) {
            if (*src == '"') {
                const char *e;
                json_decode_string(src, &e, NULL, 0);
                src = e;
                continue;
            }
            if (*src == open_ch) depth++;
            else if (*src == close_ch) depth--;
            src++;
        }
        *endp = src;
        return;
    }
    /* number / true / false / null: skip word chars */
    while (*src && !isspace((unsigned char)*src) && *src != ',' &&
           *src != ']' && *src != '}') {
        src++;
    }
    *endp = src;
}

/* Handle a value for a known descriptor field, given the parser state. */
static void parser_handle_value(OcStParser *p, const char *src, const char **endp)
{
    while (*src && isspace((unsigned char)*src)) src++;
    if (*src == '"') {
        char buf[OC_SAFETENSORS_DTYPE_LEN];
        json_decode_string(src, endp, buf, sizeof(buf));
        if (strcmp(p->field, "dtype") == 0) {
            strncpy(p->cur.dtype, buf, OC_SAFETENSORS_DTYPE_LEN - 1);
            p->cur.dtype[OC_SAFETENSORS_DTYPE_LEN - 1] = '\0';
        }
        /* "endianess" and other string fields are ignored. */
        return;
    }
    if (*src == '[') {
        /* Array value: shape or data_offsets. The skip fallback below
         * handles unknown arrays by re-parsing from the '['. */
        const char *arr_start = src;
        src++;
        if (strcmp(p->field, "shape") == 0) {
            p->shape_count = 0;
            while (*src && *src != ']') {
                while (*src && isspace((unsigned char)*src)) src++;
                if (*src == ']') break;
                const char *e;
                uint64_t v = json_decode_uint(src, &e);
                if (p->shape_count < OC_SAFETENSORS_MAX_DIMS) {
                    p->shape_vals[p->shape_count++] = v;
                }
                src = e;
                while (*src && isspace((unsigned char)*src)) src++;
                if (*src == ',') src++;
            }
            if (*src == ']') src++;
            *endp = src;
            return;
        }
        if (strcmp(p->field, "data_offsets") == 0) {
            p->offset_count = 0;
            while (*src && *src != ']') {
                while (*src && isspace((unsigned char)*src)) src++;
                if (*src == ']') break;
                const char *e;
                uint64_t v = json_decode_uint(src, &e);
                if (p->offset_count < 2) {
                    p->offsets[p->offset_count++] = v;
                }
                src = e;
                while (*src && isspace((unsigned char)*src)) src++;
                if (*src == ',') src++;
            }
            if (*src == ']') src++;
            *endp = src;
            return;
        }
        /* Unknown array: skip from the '['. */
        json_skip_value(arr_start, endp);
        return;
    }
    /* Unknown scalar value: skip. */
    json_skip_value(src, endp);
}

/* Parse the JSON header string. Populates `p->out_arr` / `p->out_len`.
 * Returns OC_OK, OC_ERR_FORMAT (malformed), or OC_ERR_OOM. */
static OcError parse_header(const char *json, size_t json_len,
                            OcStParser *p)
{
    const char *s = json;
    const char *end = json + json_len;
    OcStJsonState st = OC_STJ_TOP;
    /* Track whether the descriptor object is open so top-level commas
     * separate tensors. */
    while (s < end) {
        char c = *s;
        if (isspace((unsigned char)c)) { s++; continue; }
        switch (st) {
        case OC_STJ_TOP:
            if (c == '{') { s++; st = OC_STJ_TENSOR_KEY; }
            else if (c == '}') { s++; /* end */ }
            else if (c == ',') { s++; st = OC_STJ_TENSOR_KEY; }
            else return OC_ERR_FORMAT;
            break;
        case OC_STJ_TENSOR_KEY: {
            if (c == '}') { s++; st = OC_STJ_TOP; break; }
            if (c != '"') return OC_ERR_FORMAT;
            const char *e;
            char namebuf[OC_SAFETENSORS_NAME_LEN];
            int n = json_decode_string(s, &e, namebuf, sizeof(namebuf));
            if (n < 0) return OC_ERR_FORMAT;
            s = e;
            /* Top-level "__metadata__" key is skipped as an unknown value. */
            if (strcmp(namebuf, "__metadata__") == 0) {
                while (*s && isspace((unsigned char)*s)) s++;
                if (*s == ':') s++;
                json_skip_value(s, &e);
                s = e;
                st = OC_STJ_TOP;
                break;
            }
            /* New tensor descriptor. */
            if (!parser_emit(p)) return OC_ERR_OOM;
            memset(&p->cur, 0, sizeof(p->cur));
            size_t nl = strlen(namebuf);
            if (nl >= OC_SAFETENSORS_NAME_LEN) nl = OC_SAFETENSORS_NAME_LEN - 1;
            memcpy(p->cur.name, namebuf, nl);
            p->cur.name[nl] = '\0';
            p->cur_valid = true;
            p->shape_count = 0;
            p->offset_count = 0;
            st = OC_STJ_AFTER_KEY;
            break;
        }
        case OC_STJ_AFTER_KEY:
            if (c == ':') { s++; st = OC_STJ_DESC_OBJ; }
            else return OC_ERR_FORMAT;
            break;
        case OC_STJ_DESC_OBJ:
            if (c == '{') { s++; st = OC_STJ_FIELD_KEY; }
            else if (c == '}') {
                /* End of this descriptor object. */
                s++;
                if (!parser_emit(p)) return OC_ERR_OOM;
                st = OC_STJ_TOP;
            }
            else if (c == ',') { s++; st = OC_STJ_FIELD_KEY; }
            else return OC_ERR_FORMAT;
            break;
        case OC_STJ_FIELD_KEY: {
            if (c == '}') { s++; st = OC_STJ_DESC_OBJ; break; }
            if (c != '"') return OC_ERR_FORMAT;
            const char *e;
            json_decode_string(s, &e, p->field, sizeof(p->field));
            s = e;
            st = OC_STJ_AFTER_FIELD;
            break;
        }
        case OC_STJ_AFTER_FIELD:
            if (c == ':') { s++; st = OC_STJ_VALUE; }
            else return OC_ERR_FORMAT;
            break;
        case OC_STJ_VALUE: {
            const char *e;
            parser_handle_value(p, s, &e);
            s = e;
            /* After a value, expect ',' or '}' inside the descriptor. */
            st = OC_STJ_DESC_OBJ;
            break;
        }
        }
    }
    /* Emit any trailing in-progress tensor (defensive; well-formed JSON
     * emits on the closing '}' of each descriptor). */
    if (!parser_emit(p)) return OC_ERR_OOM;
    return OC_OK;
}


OcError oc_safetensors_open(const char *path, OcSafetensorsFile *out)
{
    if (path == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

#if OC_HAVE_MMAP
    int fd = open(path, O_RDONLY);
    if (fd < 0) return OC_ERR_IO;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return OC_ERR_IO;
    }
    if (st.st_size < 8) {
        close(fd);
        return OC_ERR_FORMAT;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return OC_ERR_IO;
    }
    /* Close fd now; the mapping persists until munmap. */
    close(fd);
    uint8_t *base = (uint8_t *)map;
    uint64_t header_len = 0;
    memcpy(&header_len, base, 8);
    if (header_len == 0 || (uint64_t)st.st_size < 8u + header_len) {
        munmap(map, (size_t)st.st_size);
        return OC_ERR_FORMAT;
    }
    const char *json = (const char *)(base + 8);
    uint64_t data_start = 8u + header_len;
    out->raw_data = base + data_start;
    out->data_start = data_start;
    out->file_size = (uint64_t)st.st_size;
    out->mmapped = true;

    OcStParser p = {0};
    OcError e = parse_header(json, (size_t)header_len, &p);
    if (e != OC_OK) {
        free(p.out_arr);
        munmap(map, (size_t)st.st_size);
        memset(out, 0, sizeof(*out));
        return e;
    }
    out->tensors = p.out_arr;
    out->n_tensors = p.out_len;
    return OC_OK;
#else
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return OC_ERR_IO;
    /* Read header length. */
    uint8_t hdr_len_buf[8];
    if (fread(hdr_len_buf, 1, 8, fp) != 8) { fclose(fp); return OC_ERR_FORMAT; }
    uint64_t header_len = 0;
    memcpy(&header_len, hdr_len_buf, 8);
    if (header_len == 0) { fclose(fp); return OC_ERR_FORMAT; }
    /* Read JSON header. */
    char *json = malloc((size_t)header_len + 1);
    if (json == NULL) { fclose(fp); return OC_ERR_OOM; }
    if (fread(json, 1, (size_t)header_len, fp) != (size_t)header_len) {
        free(json); fclose(fp); return OC_ERR_FORMAT;
    }
    json[header_len] = '\0';
    /* Read the rest into a buffer. */
    uint64_t data_start = 8u + header_len;
    /* Determine file size. */
    if (fseek(fp, 0, SEEK_END) != 0) { free(json); fclose(fp); return OC_ERR_IO; }
    long fsize = ftell(fp);
    if (fsize < 0) { free(json); fclose(fp); return OC_ERR_IO; }
    uint64_t file_size = (uint64_t)fsize;
    uint64_t data_len = file_size > data_start ? file_size - data_start : 0;
    void *raw = NULL;
    if (data_len > 0) {
        raw = malloc((size_t)data_len);
        if (raw == NULL) { free(json); fclose(fp); return OC_ERR_OOM; }
        if (fseek(fp, (long)data_start, SEEK_SET) != 0 ||
            fread(raw, 1, (size_t)data_len, fp) != (size_t)data_len) {
            free(raw); free(json); fclose(fp); return OC_ERR_IO;
        }
    }
    fclose(fp);

    out->raw_data = raw;
    out->data_start = data_start;
    out->file_size = file_size;
    out->mmapped = false;

    OcStParser p = {0};
    OcError e = parse_header(json, (size_t)header_len, &p);
    free(json);
    if (e != OC_OK) {
        free(p.out_arr);
        free(raw);
        memset(out, 0, sizeof(*out));
        return e;
    }
    out->tensors = p.out_arr;
    out->n_tensors = p.out_len;
    return OC_OK;
#endif
}

OcError oc_safetensors_get_tensor(const OcSafetensorsFile *st,
                                  const char *name,
                                  const OcSafetensorsTensor **out)
{
    if (st == NULL || name == NULL || out == NULL) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < st->n_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            *out = &st->tensors[i];
            return OC_OK;
        }
    }
    return OC_ERR_TENSOR;
}

OcError oc_safetensors_get_tensor_data(const OcSafetensorsFile *st,
                                       const OcSafetensorsTensor *tensor,
                                       const void **out_data)
{
    if (st == NULL || tensor == NULL || out_data == NULL) return OC_ERR_INVALID_ARG;
    if (st->raw_data == NULL) return OC_ERR_FORMAT;
    /* Validate the tensor belongs to this file (offset+length within bounds). */
    uint64_t end_off = tensor->data_offset + tensor->data_length;
    if (end_off < tensor->data_offset) return OC_ERR_FORMAT; /* overflow */
    uint64_t avail = st->file_size > st->data_start
                     ? st->file_size - st->data_start : 0;
    if (end_off > avail) return OC_ERR_FORMAT;
    *out_data = (const uint8_t *)st->raw_data + tensor->data_offset;
    return OC_OK;
}

size_t oc_safetensors_n_tensors(const OcSafetensorsFile *st)
{
    return st ? st->n_tensors : 0;
}

void oc_safetensors_close(OcSafetensorsFile *st)
{
    if (st == NULL) return;
    free(st->tensors);
    if (st->mmapped) {
#if OC_HAVE_MMAP
        if (st->raw_data != NULL) {
            /* raw_data points at base + data_start; recover the mmap base. */
            void *base = (uint8_t *)st->raw_data - st->data_start;
            munmap(base, (size_t)st->file_size);
        }
#endif
    } else {
        free(st->raw_data);
    }
    memset(st, 0, sizeof(*st));
}
