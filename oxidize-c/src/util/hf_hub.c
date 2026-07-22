/*
 * hf_hub.c — HuggingFace Hub GGUF model downloader implementation.
 *
 * Uses raw TCP sockets for HTTP/1.1 (no libcurl dependency). All network
 * calls are to plain HTTP — for HTTPS, point api_base at a proxy.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/hf_hub.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void to_upper(char *s)
{
    for (; *s; s++) *s = (char)toupper((unsigned char)*s);
}

static bool ends_with_ci(const char *s, const char *suffix)
{
    size_t ls = strlen(s), lf = strlen(suffix);
    if (lf > ls) return false;
    for (size_t i = 0; i < lf; i++)
        if (tolower((unsigned char)s[ls - lf + i]) != tolower((unsigned char)suffix[i]))
            return false;
    return true;
}

/* ─── Public helpers ────────────────────────────────────────────────────── */

bool oc_hf_is_gguf(const char *filename)
{
    return filename && ends_with_ci(filename, ".gguf");
}

bool oc_hf_parse_quant_type(const char *filename, char *out, size_t cap)
{
    if (!filename || !out || cap == 0) { if (out) out[0] = '\0'; return false; }
    out[0] = '\0';

    /* Find the base filename (strip directory). */
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;

    /* Look for Q\d[_-]\w+ or F16 or F32 pattern. */
    for (const char *p = base; *p; p++) {
        if ((*p == 'Q' || *p == 'q') && isdigit((unsigned char)p[1])) {
            /* Found start of a quant tag. Extract until '-', '.', '_' or end. */
            size_t len = 0;
            const char *q = p;
            while (*q && *q != '-' && *q != '.' && *q != '/' && len < cap - 1) {
                out[len++] = (char)toupper((unsigned char)*q);
                q++;
            }
            if (len > 0) {
                out[len] = '\0';
                return true;
            }
        }
        if ((*p == 'F' || *p == 'f') && (p[1] == '1' || p[1] == '3') &&
            (p[2] == '6' || p[2] == '2')) {
            size_t len = (p[2] == '6') ? 3 : 3;
            if (len < cap) {
                out[0] = 'F'; out[1] = p[1]; out[2] = p[2]; out[3] = '\0';
                return true;
            }
        }
    }
    return false;
}

OcError oc_hf_sanitize_repo_id(const char *repo_id, char *out, size_t cap)
{
    if (!repo_id || !out || cap == 0) return OC_ERR_INVALID_ARG;
    size_t len = strlen(repo_id);
    if (len >= cap) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i <= len; i++) {
        out[i] = (repo_id[i] == '/') ? '_' : repo_id[i];
    }
    return OC_OK;
}

OcError oc_hf_default_cache_dir(char *out, size_t cap)
{
    if (!out || cap == 0) return OC_ERR_INVALID_ARG;
    const char *home = getenv("HOME");
    if (!home || *home == '\0') home = "/tmp";
    int n = snprintf(out, cap, "%s/%s", home, OC_HF_DEFAULT_CACHE_SUFFIX);
    if (n < 0 || (size_t)n >= cap) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

OcError oc_hf_config_init(OcHfConfig *cfg, const char *cache_dir)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    if (cache_dir && *cache_dir) {
        strncpy(cfg->cache_dir, cache_dir, sizeof(cfg->cache_dir) - 1);
    } else {
        OcError e = oc_hf_default_cache_dir(cfg->cache_dir, sizeof(cfg->cache_dir));
        if (e != OC_OK) return e;
    }
    strncpy(cfg->revision, OC_HF_DEFAULT_REVISION, sizeof(cfg->revision) - 1);
    strncpy(cfg->api_base, OC_HF_DEFAULT_API_BASE, sizeof(cfg->api_base) - 1);
    return OC_OK;
}

OcError oc_hf_cache_path(const OcHfConfig *cfg,
                         const char *repo_id, const char *filename,
                         char *out_path, size_t cap)
{
    if (!cfg || !repo_id || !filename || !out_path || cap == 0)
        return OC_ERR_INVALID_ARG;

    char sanitized[OC_HF_MAX_REPO_ID];
    OcError e = oc_hf_sanitize_repo_id(repo_id, sanitized, sizeof(sanitized));
    if (e != OC_OK) return e;

    int n = snprintf(out_path, cap, "%s/%s/%s", cfg->cache_dir, sanitized, filename);
    if (n < 0 || (size_t)n >= cap) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

/* ─── Simple HTTP client ────────────────────────────────────────────────── */

static int http_connect(const char *host, int port)
{
    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0)
        return -1;

    int fd = -1;
    for (struct addrinfo *p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static int http_parse_url(const char *url, char *host, size_t host_cap,
                          char *path, size_t path_cap, int *port)
{
    const char *p = url;
    if (strncmp(p, "http://", 7) == 0) p += 7;
    else if (strncmp(p, "https://", 8) == 0) p += 8;
    *port = (*p == 'h' || url[0] == 'h' && url[5] == 's') ? 443 : 80;
    if (url[4] == 's' || url[5] == 's') *port = 443;
    if (strncmp(url, "http://", 7) == 0) *port = 80;

    /* Extract host. */
    size_t hlen = 0;
    while (*p && *p != '/' && *p != ':' && hlen < host_cap - 1)
        host[hlen++] = *p++;
    host[hlen] = '\0';

    if (*p == ':') {
        p++;
        *port = 0;
        while (isdigit((unsigned char)*p)) {
            *port = *port * 10 + (*p - '0');
            p++;
        }
    }

    if (*p == '\0') {
        strncpy(path, "/", path_cap - 1);
        path[1] = '\0';
    } else {
        strncpy(path, p, path_cap - 1);
        path[path_cap - 1] = '\0';
    }
    return 0;
}

/* ─── Minimal JSON siblings parser ──────────────────────────────────────── */

static const char *find_siblings_array(const char *json, size_t len)
{
    const char *p = json;
    const char *end = json + len;
    while (p < end) {
        if (strncmp(p, "\"siblings\"", 10) == 0) {
            p += 10;
            while (p < end && *p != '[') p++;
            return (p < end) ? p + 1 : NULL;
        }
        p++;
    }
    return NULL;
}

static const char *extract_rfilename(const char *p, const char *end,
                                      char *out, size_t cap)
{
    /* Look for "rfilename":"..." pattern. */
    while (p < end) {
        if (strncmp(p, "\"rfilename\"", 11) == 0) {
            p += 11;
            while (p < end && *p != '"' && *p != ':') p++;
            if (p < end && *p == ':') p++;
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
            if (p < end && *p == '"') {
                p++;
                size_t len = 0;
                while (p < end && *p != '"' && len < cap - 1) {
                    if (*p == '\\' && p + 1 < end) p++; /* skip escape */
                    out[len++] = *p++;
                }
                out[len] = '\0';
                return (p < end) ? p : end;
            }
        }
        p++;
    }
    return NULL;
}

/* ─── Network API ──────────────────────────────────────────────────────── */

OcError oc_hf_list_models(const OcHfConfig *cfg,
                          OcHfModel *out_models, size_t *inout_count)
{
    if (!cfg || !out_models || !inout_count || cfg->repo_id[0] == '\0')
        return OC_ERR_INVALID_ARG;

    /* Build API URL: {api_base}/api/models/{repo_id} */
    char host[256], path[512];
    int port;
    char url[OC_HF_MAX_URL];
    snprintf(url, sizeof(url), "%s/api/models/%s", cfg->api_base, cfg->repo_id);

    if (http_parse_url(url, host, sizeof(host), path, sizeof(path), &port) != 0)
        return OC_ERR_IO;

    int fd = http_connect(host, port);
    if (fd < 0) return OC_ERR_NETWORK;

    /* Send HTTP GET. */
    char request[1024];
    int rn = snprintf(request, sizeof(request),
        "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: oxidize-c\r\n"
        "Accept: application/json\r\nConnection: close\r\n\r\n",
        path, host);
    if (send(fd, request, rn, 0) != rn) { close(fd); return OC_ERR_NETWORK; }

    /* Read response (simple: read all into a buffer). */
    size_t buf_cap = 1 << 20; /* 1 MiB max */
    char *buf = malloc(buf_cap);
    if (!buf) { close(fd); return OC_ERR_OOM; }
    size_t buf_len = 0;
    ssize_t n;
    while (buf_len < buf_cap - 1 &&
           (n = recv(fd, buf + buf_len, buf_cap - buf_len - 1, 0)) > 0)
        buf_len += n;
    buf[buf_len] = '\0';
    close(fd);

    /* Skip HTTP headers (find \r\n\r\n). */
    char *body = strstr(buf, "\r\n\r\n");
    if (body) body += 4;
    else body = buf;

    /* Parse siblings array. */
    const char *arr = find_siblings_array(body, buf_len - (size_t)(body - buf));
    if (!arr) { free(buf); return OC_ERR_FORMAT; }

    size_t count = 0;
    const char *p = arr;
    const char *end = buf + buf_len;
    char filename[OC_HF_MAX_FILENAME];

    while (p < end && count < *inout_count) {
        p = extract_rfilename(p, end, filename, sizeof(filename));
        if (!p) break;

        if (oc_hf_is_gguf(filename)) {
            /* Check quant filter. */
            char qt[OC_HF_MAX_QUANT_TYPE];
            oc_hf_parse_quant_type(filename, qt, sizeof(qt));
            if (cfg->quant_type[0] != '\0') {
                char filter[OC_HF_MAX_QUANT_TYPE];
                strncpy(filter, cfg->quant_type, sizeof(filter) - 1);
                filter[sizeof(filter) - 1] = '\0';
                to_upper(filter);
                if (strcmp(qt, filter) != 0) continue;
            }
            /* Fill model entry. */
            OcHfModel *m = &out_models[count];
            memset(m, 0, sizeof(*m));
            strncpy(m->repo_id, cfg->repo_id, sizeof(m->repo_id) - 1);
            strncpy(m->filename, filename, sizeof(m->filename) - 1);
            strncpy(m->quant_type, qt, sizeof(m->quant_type) - 1);
            snprintf(m->download_url, sizeof(m->download_url),
                     "%s/%s/resolve/%s/%s",
                     cfg->api_base, cfg->repo_id,
                     cfg->revision[0] ? cfg->revision : "main",
                     filename);
            count++;
        }
        p = strstr(p, ",");
        if (!p) break;
        p++;
    }

    free(buf);
    *inout_count = count;
    return OC_OK;
}

OcError oc_hf_resolve(const OcHfConfig *cfg, OcHfModel *out_model)
{
    if (!cfg || !out_model) return OC_ERR_INVALID_ARG;
    if (out_model->filename[0] != '\0') {
        /* Already have filename, just build URL. */
        snprintf(out_model->download_url, sizeof(out_model->download_url),
                 "%s/%s/resolve/%s/%s",
                 cfg->api_base, out_model->repo_id,
                 cfg->revision[0] ? cfg->revision : "main",
                 out_model->filename);
        return OC_OK;
    }
    /* List models and pick the single .gguf. */
    OcHfModel models[OC_HF_MAX_MODELS];
    size_t count = OC_HF_MAX_MODELS;
    OcError e = oc_hf_list_models(cfg, models, &count);
    if (e != OC_OK) return e;
    if (count == 0) return OC_ERR_MODEL;
    if (count > 1) return OC_ERR_MODEL;
    *out_model = models[0];
    return OC_OK;
}

/* ─── Download ────────────────────────────────────────────────────────── */

static pthread_mutex_t g_download_mutex = PTHREAD_MUTEX_INITIALIZER;

OcError oc_hf_download(const OcHfConfig *cfg, const OcHfModel *model,
                       OcHfProgressCb cb, void *user)
{
    if (!cfg || !model) return OC_ERR_INVALID_ARG;
    pthread_mutex_lock(&g_download_mutex);

    /* Compute cache path. */
    char local_path[OC_HF_MAX_CACHE_DIR + OC_HF_MAX_FILENAME];
    OcError e = oc_hf_cache_path(cfg, model->repo_id, model->filename,
                                  local_path, sizeof(local_path));
    if (e != OC_OK) { pthread_mutex_unlock(&g_download_mutex); return e; }

    /* Create directory if needed. */
    char dir[OC_HF_MAX_CACHE_DIR + OC_HF_MAX_REPO_ID];
    snprintf(dir, sizeof(dir), "%s", local_path);
    char *last_slash = strrchr(dir, '/');
    if (last_slash) { *last_slash = '\0'; }
    /* mkdir -p */
    for (char *p = dir + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(dir, 0755);
            *p = '/';
        }
    }
    mkdir(dir, 0755);

    /* Check for existing complete file. */
    struct stat st;
    if (stat(local_path, &st) == 0 && st.st_size > 0) {
        /* Already downloaded. */
        if (cb) {
            OcHfDownloadProgress prog = {
                .downloaded_bytes = st.st_size,
                .total_bytes = st.st_size,
                .speed = 0, .eta = 0
            };
            cb(&prog, user);
        }
        pthread_mutex_unlock(&g_download_mutex);
        return OC_OK;
    }

    /* Check for .part file (resume). */
    char part_path[OC_HF_MAX_CACHE_DIR + OC_HF_MAX_FILENAME + 6];
    snprintf(part_path, sizeof(part_path), "%s.part", local_path);
    uint64_t offset = 0;
    if (stat(part_path, &st) == 0) offset = st.st_size;

    /* Parse URL and connect. */
    char host[256], path[512];
    int port;
    if (http_parse_url(model->download_url, host, sizeof(host),
                       path, sizeof(path), &port) != 0) {
        pthread_mutex_unlock(&g_download_mutex);
        return OC_ERR_IO;
    }

    int fd = http_connect(host, port);
    if (fd < 0) { pthread_mutex_unlock(&g_download_mutex); return OC_ERR_NETWORK; }

    /* Send HTTP GET with Range header for resume. */
    char request[1024];
    int rn;
    if (offset > 0) {
        rn = snprintf(request, sizeof(request),
            "GET %s HTTP/1.1\r\nHost: %s\r\nRange: bytes=%llu-\r\n"
            "User-Agent: oxidize-c\r\nConnection: close\r\n\r\n",
            path, host, (unsigned long long)offset);
    } else {
        rn = snprintf(request, sizeof(request),
            "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: oxidize-c\r\n"
            "Connection: close\r\n\r\n", path, host);
    }
    if (send(fd, request, rn, 0) != rn) {
        close(fd);
        pthread_mutex_unlock(&g_download_mutex);
        return OC_ERR_NETWORK;
    }

    /* Read response headers. */
    char header_buf[OC_HF_HTTP_HEADER_BUF];
    size_t header_len = 0;
    while (header_len < sizeof(header_buf) - 1) {
        ssize_t n = recv(fd, header_buf + header_len,
                         sizeof(header_buf) - header_len - 1, 0);
        if (n <= 0) break;
        header_len += n;
        header_buf[header_len] = '\0';
        if (strstr(header_buf, "\r\n\r\n")) break;
    }

    /* Check status code. */
    int status = 0;
    if (sscanf(header_buf, "HTTP/1.%*d %d", &status) != 1) {
        close(fd);
        pthread_mutex_unlock(&g_download_mutex);
        return OC_ERR_FORMAT;
    }
    if (status != 200 && status != 206) {
        close(fd);
        pthread_mutex_unlock(&g_download_mutex);
        return OC_ERR_IO;
    }

    /* Extract Content-Length. */
    uint64_t total = 0;
    char *cl = strstr(header_buf, "Content-Length:");
    if (cl) sscanf(cl, "Content-Length: %llu", (unsigned long long *)&total);
    if (offset > 0 && total > 0) total += offset;

    /* Find end of headers. */
    char *body_start = strstr(header_buf, "\r\n\r\n");
    size_t header_end = body_start ? (size_t)(body_start + 4 - header_buf) : header_len;

    /* Open .part file for writing (append). */
    FILE *fp = fopen(part_path, offset > 0 ? "ab" : "wb");
    if (!fp) {
        close(fd);
        pthread_mutex_unlock(&g_download_mutex);
        return OC_ERR_IO;
    }

    /* Write any body data we already read. */
    if (header_end < header_len) {
        fwrite(header_buf + header_end, 1, header_len - header_end, fp);
        offset += header_len - header_end;
    }

    /* Download remaining data. */
    char chunk[OC_HF_DOWNLOAD_CHUNK];
    uint64_t downloaded = offset;
    time_t start_time = time(NULL);

    while (1) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        fwrite(chunk, 1, n, fp);
        downloaded += n;

        if (cb && (downloaded % (1 << 20) < (size_t)n)) {
            time_t elapsed = time(NULL) - start_time;
            double speed = elapsed > 0 ? (double)downloaded / elapsed : 0;
            double eta = speed > 0 ? (double)(total - downloaded) / speed : -1;
            OcHfDownloadProgress prog = {
                .downloaded_bytes = downloaded,
                .total_bytes = total,
                .speed = speed,
                .eta = eta
            };
            if (cb(&prog, user) != 0) {
                fclose(fp);
                close(fd);
                pthread_mutex_unlock(&g_download_mutex);
                return OC_ERR_IO;
            }
        }
    }

    fclose(fp);
    close(fd);

    /* Rename .part to final. */
    rename(part_path, local_path);

    pthread_mutex_unlock(&g_download_mutex);
    return OC_OK;
}

/* ─── Cache management ──────────────────────────────────────────────────── */

static void scan_dir_for_gguf(const char *dir, OcHfModel *out, size_t *count, size_t max)
{
    DIR *d = opendir(dir);
    if (!d) return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && *count < max) {
        if (ent->d_name[0] == '.') continue;

        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(full_path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            scan_dir_for_gguf(full_path, out, count, max);
        } else if (oc_hf_is_gguf(ent->d_name)) {
            OcHfModel *m = &out[*count];
            memset(m, 0, sizeof(*m));
            strncpy(m->filename, ent->d_name, sizeof(m->filename) - 1);
            m->size_bytes = st.st_size;
            oc_hf_parse_quant_type(ent->d_name, m->quant_type, sizeof(m->quant_type));
            (*count)++;
        }
    }
    closedir(d);
}

OcError oc_hf_cache_list(const OcHfConfig *cfg,
                         OcHfModel *out_models, size_t *inout_count)
{
    if (!cfg || !out_models || !inout_count) return OC_ERR_INVALID_ARG;
    *inout_count = 0;
    scan_dir_for_gguf(cfg->cache_dir, out_models, inout_count, *inout_count);
    return OC_OK;
}

static void sum_dir_size(const char *dir, uint64_t *total)
{
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            sum_dir_size(full, total);
        } else {
            *total += st.st_size;
        }
    }
    closedir(d);
}

OcError oc_hf_cache_size(const OcHfConfig *cfg, uint64_t *out_bytes)
{
    if (!cfg || !out_bytes) return OC_ERR_INVALID_ARG;
    *out_bytes = 0;
    sum_dir_size(cfg->cache_dir, out_bytes);
    return OC_OK;
}

static void clean_dir_recursive(const char *dir, uint64_t max_age, size_t *removed)
{
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *ent;
    time_t now = time(NULL);
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            clean_dir_recursive(full, max_age, removed);
        } else {
            if (max_age == 0 || (uint64_t)(now - st.st_mtime) > max_age) {
                if (unlink(full) == 0) (*removed)++;
            }
        }
    }
    closedir(d);
}

OcError oc_hf_cache_clean(const OcHfConfig *cfg, uint64_t max_age_seconds,
                          size_t *out_removed)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    size_t removed = 0;
    clean_dir_recursive(cfg->cache_dir, max_age_seconds, &removed);
    if (out_removed) *out_removed = removed;
    return OC_OK;
}
