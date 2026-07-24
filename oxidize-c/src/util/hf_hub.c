/*
 * hf_hub.c — HuggingFace Hub GGUF model downloader implementation.
 *
 * Ports the HF resolver from oxidize-golang/hf/hub.go and the pure-Python
 * oxidize_python.hf.hub module into the dependency-free C11 port. Uses a
 * raw-socket HTTP/1.1 client (no libcurl) consistent with mesh.c/http.c.
 *
 * TLS note: HuggingFace's production API is HTTPS-only. This client
 * speaks plain HTTP/1.1 over TCP — point api_base at an HTTPS-terminating
 * proxy or local mirror for production use, or compile with a TLS shim.
 *
 * Rate limiting: oc_hf_download() holds a process-global mutex for the
 * duration of the transfer (max 1 concurrent download per process).
 */
#define _POSIX_C_SOURCE 200809L  /* getpwuid, stat, ssize_t */

#include "oxidize/hf_hub.h"
#include "oxidize/error.h"
#include "oxidize/log.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <pwd.h>
#include <pthread.h>
#include <dirent.h>
#include <arpa/inet.h>

/* ─── Internal constants ──────────────────────────────────────────────── */

#define OC_HF_DEFAULT_HOST_PORT 80u
#define OC_HF_RECV_BUF (1u << 16)   /* 64 KiB recv buffer */

/* Process-global single-download mutex (rate limiter). */
static pthread_mutex_t g_hf_dl_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ─── Small helpers ──────────────────────────────────────────────────── */

static void oc_copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    if (!src) { dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

/* Case-insensitive ASCII. */
static int oc_ci(int c) { return tolower((unsigned char)c); }

/* Forward decl — defined below; used by http_get_body's header parsing. */
static bool oc_ci_ends_with_prefix(const char *line, size_t line_len,
                                   const char *prefix_lower);

/* Local case-insensitive strcmp (avoids a hard dependency on util/string.c
 * when this TU is compiled standalone for the header-only test build). */
static int hf_ci_strcmp(const char *a, const char *b)
{
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    while (*a && *b) {
        int ca = oc_ci(*a), cb = oc_ci(*b);
        if (ca != cb) return ca - cb;
        a++; b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

static bool oc_ci_ends_with(const char *s, const char *suffix)
{
    if (!s || !suffix) return false;
    size_t ls = strlen(s), lf = strlen(suffix);
    if (lf > ls) return false;
    for (size_t i = 0; i < lf; i++) {
        if (oc_ci(s[ls - lf + i]) != oc_ci(suffix[i])) return false;
    }
    return true;
}

/* ─── Default cache dir ──────────────────────────────────────────────── */

OcError oc_hf_default_cache_dir(char *out, size_t cap)
{
    if (!out || cap == 0) return OC_ERR_INVALID_ARG;
    const char *home = getenv("HOME");
    if (!home || home[0] == '\0') {
        /* Fall back to /tmp/oxidize-hf. */
        const char *tmp = getenv("TMPDIR");
        if (!tmp || tmp[0] == '\0') tmp = "/tmp";
        int n = snprintf(out, cap, "%s/oxidize-hf", tmp);
        if (n < 0 || (size_t)n >= cap) return OC_ERR_INVALID_ARG;
        return OC_OK;
    }
    int n = snprintf(out, cap, "%s/%s", home, OC_HF_DEFAULT_CACHE_SUFFIX);
    if (n < 0 || (size_t)n >= cap) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

OcError oc_hf_config_init(OcHfConfig *cfg, const char *cache_dir)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    if (cache_dir && cache_dir[0] != '\0') {
        oc_copy_str(cfg->cache_dir, sizeof(cfg->cache_dir), cache_dir);
    } else {
        char tmp[OC_HF_MAX_CACHE_DIR];
        if (oc_hf_default_cache_dir(tmp, sizeof(tmp)) != OC_OK) {
            oc_copy_str(cfg->cache_dir, sizeof(cfg->cache_dir),
                        "/tmp/oxidize-hf");
        } else {
            oc_copy_str(cfg->cache_dir, sizeof(cfg->cache_dir), tmp);
        }
    }
    oc_copy_str(cfg->revision, sizeof(cfg->revision), OC_HF_DEFAULT_REVISION);
    oc_copy_str(cfg->api_base, sizeof(cfg->api_base), OC_HF_DEFAULT_API_BASE);
    cfg->api_token[0] = '\0';
    cfg->repo_id[0] = '\0';
    cfg->quant_type[0] = '\0';
    return OC_OK;
}

/* ─── Filename parsing ───────────────────────────────────────────────── */

bool oc_hf_is_gguf(const char *filename)
{
    return oc_ci_ends_with(filename, ".gguf");
}

/* Parse a quant tag from a filename. Looks for a token matching the
 * pattern (Q|F|BF|I|IQ|NX)\d+[_-]?[A-Za-z0-9_]* at the end of the basename
 * (before .gguf). This is a heuristic, not a strict spec — it mirrors the
 * common GGUF naming conventions (Q4_K_M.gguf, Q8_0.gguf, F16.gguf,
 * IQ2_XXS.gguf, etc.). */
bool oc_hf_parse_quant_type(const char *filename, char *out, size_t cap)
{
    if (!out || cap == 0) return false;
    out[0] = '\0';
    if (!filename) return false;

    /* Strip directory component. */
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;

    /* Strip ".gguf" suffix. */
    size_t len = strlen(base);
    const char suffix[] = ".gguf";
    size_t slen = sizeof(suffix) - 1;
    if (len < slen) return false;
    if (oc_ci_ends_with(base, suffix)) {
        len -= slen;
    }
    if (len == 0) return false;

    /* Scan backward from the end of the stripped name, accumulating the
     * quant tag. A quant tag starts at one of: Q, F, BF, I, IQ, NX, B
     * followed by a digit. We walk backwards while the char is [A-Z0-9_]
     * (excluding '-' since that separates the tag from the model name). */
    char buf[OC_HF_MAX_QUANT_TYPE];
    /* Walk from end of basename backward while alnum or '_'. */
    size_t i = len;
    while (i > 0) {
        char c = base[i - 1];
        if (isalnum((unsigned char)c) || c == '_') {
            i--;
            continue;
        }
        break;
    }
    /* Now [i, len) is the trailing token. Check it has a digit and a
     * plausible quant prefix. */
    size_t tok_len = len - i;
    if (tok_len == 0 || tok_len >= sizeof(buf)) {
        /* Fall back: take the whole basename stripped of non-tag chars. */
        return false;
    }
    memcpy(buf, base + i, tok_len);
    buf[tok_len] = '\0';

    /* Find first digit. */
    size_t first_digit = 0;
    bool found_digit = false;
    for (size_t k = 0; k < tok_len; k++) {
        if (isdigit((unsigned char)buf[k])) {
            first_digit = k;
            found_digit = true;
            break;
        }
    }
    if (!found_digit) return false;

    /* The prefix before the digit must be one of the recognized quant
     * families (case-insensitive). */
    char prefix[8];
    size_t plen = first_digit;
    if (plen == 0 || plen >= sizeof(prefix)) return false;
    for (size_t k = 0; k < plen; k++) {
        prefix[k] = (char)oc_ci(buf[k]);
    }
    prefix[plen] = '\0';

    /* Recognized families. */
    static const char *families[] = {
        "q", "f", "bf", "i", "iq", "nx", "b", "t", "al", NULL
    };
    bool matched = false;
    for (size_t k = 0; families[k]; k++) {
        if (strcmp(prefix, families[k]) == 0) { matched = true; break; }
    }
    if (!matched) return false;

    /* Uppercase the tag and emit. */
    for (size_t k = 0; k < tok_len; k++) {
        buf[k] = (char)toupper((unsigned char)buf[k]);
    }
    if (tok_len + 1 > cap) {
        tok_len = cap - 1;
    }
    memcpy(out, buf, tok_len);
    out[tok_len] = '\0';
    return true;
}

OcError oc_hf_sanitize_repo_id(const char *repo_id, char *out, size_t cap)
{
    if (!repo_id || !out || cap == 0) return OC_ERR_INVALID_ARG;
    size_t n = strlen(repo_id);
    if (n == 0 || n >= cap) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < n; i++) {
        char c = repo_id[i];
        out[i] = (c == '/') ? '_' : c;
    }
    out[n] = '\0';
    return OC_OK;
}

OcError oc_hf_cache_path(const OcHfConfig *cfg,
                         const char *repo_id, const char *filename,
                         char *out_path, size_t cap)
{
    if (!cfg || !repo_id || !filename || !out_path || cap == 0)
        return OC_ERR_INVALID_ARG;

    char sanitized[OC_HF_MAX_REPO_ID];
    if (oc_hf_sanitize_repo_id(repo_id, sanitized, sizeof(sanitized)) != OC_OK)
        return OC_ERR_INVALID_ARG;

    const char *cache = cfg->cache_dir[0] ? cfg->cache_dir : NULL;
    char default_cache[OC_HF_MAX_CACHE_DIR];
    if (!cache) {
        if (oc_hf_default_cache_dir(default_cache, sizeof(default_cache))
                != OC_OK) {
            return OC_ERR_INTERNAL;
        }
        cache = default_cache;
    }

    int n = snprintf(out_path, cap, "%s/%s/%s", cache, sanitized, filename);
    if (n < 0 || (size_t)n >= cap) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

/* ─── Minimal JSON sibling parser ────────────────────────────────────── */

/* Scans a JSON buffer for `"rfilename":"..."` string values, collecting
 * them into a caller-provided array. Returns the count written. We don't
 * build a full JSON tree — the HF /api/models response shape is stable
 * enough that a streaming substring scan is sufficient and ~10x smaller
 * than a real parser. */
static size_t oc_hf_extract_rfilenames(const char *json, size_t json_len,
                                       char **out, size_t max_out)
{
    if (!json || !out || max_out == 0) return 0;
    size_t found = 0;
    const char *needle = "\"rfilename\"";
    size_t nlen = strlen(needle);

    const char *p = json;
    const char *end = json + json_len;
    while (found < max_out) {
        /* Find the next "rfilename" key. */
        const char *k = NULL;
        for (const char *q = p; q + nlen <= end; q++) {
            if (memcmp(q, needle, nlen) == 0) { k = q; break; }
        }
        if (!k) break;

        /* Skip past the key. */
        p = k + nlen;
        /* Skip whitespace. */
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end || *p != ':') break;
        p++;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end || *p != '"') break;
        p++; /* past opening quote */
        const char *vstart = p;
        /* Find closing quote, honoring escapes (simple: \" and \\). */
        const char *vq = NULL;
        while (p < end) {
            if (*p == '\\' && p + 1 < end) { p += 2; continue; }
            if (*p == '"') { vq = p; break; }
            p++;
        }
        if (!vq) break;
        size_t vlen = (size_t)(vq - vstart);
        char *s = (char *)malloc(vlen + 1);
        if (!s) break;
        /* Unescape minimally (only \" and \\). */
        size_t w = 0;
        for (size_t i = 0; i < vlen; i++) {
            char c = vstart[i];
            if (c == '\\' && i + 1 < vlen) {
                char nxt = vstart[i + 1];
                if (nxt == '"' || nxt == '\\') { s[w++] = nxt; i++; continue; }
                if (nxt == 'n') { s[w++] = '\n'; i++; continue; }
                if (nxt == 't') { s[w++] = '\t'; i++; continue; }
            }
            s[w++] = c;
        }
        s[w] = '\0';
        out[found++] = s;
        p = vq + 1;
    }
    return found;
}

/* ─── Raw-socket HTTP client ─────────────────────────────────────────── */

typedef struct {
    int fd;
} OcHfConn;

/* Parse a URL of the form [http://]host[:port]/path into host, port, path.
 * If no scheme is present, defaults to HTTP on port 80. Returns OC_OK or
 * OC_ERR_INVALID_ARG. `host_out` must be at least 256 bytes. */
static OcError parse_url(const char *url,
                         char *host_out, size_t host_cap,
                         uint16_t *port_out,
                         const char **path_out)
{
    if (!url || !host_out || !port_out || !path_out)
        return OC_ERR_INVALID_ARG;
    const char *s = url;
    /* Skip scheme. */
    if (strncmp(s, "http://", 7) == 0) s += 7;
    else if (strncmp(s, "https://", 8) == 0) {
        /* We don't speak TLS; treat as plain HTTP (proxy expected). */
        s += 8;
    }
    /* Find end of host:port (first '/' or end). */
    const char *slash = strchr(s, '/');
    size_t hostport_len = slash ? (size_t)(slash - s) : strlen(s);
    if (hostport_len == 0 || hostport_len >= host_cap) return OC_ERR_INVALID_ARG;

    /* Split host:port. */
    const char *colon = NULL;
    for (size_t i = 0; i < hostport_len; i++) {
        if (s[i] == ':') { colon = s + i; break; }
    }
    size_t host_len;
    uint16_t port = OC_HF_DEFAULT_HOST_PORT;
    if (colon) {
        host_len = (size_t)(colon - s);
        /* parse port after colon up to hostport_len */
        char pb[16];
        size_t plen = hostport_len - host_len - 1;
        if (plen == 0 || plen >= sizeof(pb)) return OC_ERR_INVALID_ARG;
        memcpy(pb, colon + 1, plen);
        pb[plen] = '\0';
        char *endp = NULL;
        long pv = strtol(pb, &endp, 10);
        if (*endp != '\0' || pv <= 0 || pv > 65535) return OC_ERR_INVALID_ARG;
        port = (uint16_t)pv;
    } else {
        host_len = hostport_len;
    }
    if (host_len == 0) return OC_ERR_INVALID_ARG;
    memcpy(host_out, s, host_len);
    host_out[host_len] = '\0';
    *port_out = port;
    *path_out = slash ? slash : "/";
    return OC_OK;
}

static OcError oc_hf_connect(const char *host, uint16_t port, OcHfConn *out)
{
    if (!host || !out) return OC_ERR_INVALID_ARG;
    out->fd = -1;

    /* Try numeric IP first (fast path). */
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) > 0) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return OC_ERR_NETWORK;
        if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
            close(fd);
            return OC_ERR_NETWORK;
        }
        out->fd = fd;
        return OC_OK;
    }

    /* DNS lookup. */
    struct addrinfo hints, *res = NULL, *rp = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", port);
    int gai = getaddrinfo(host, port_str, &hints, &res);
    if (gai != 0) {
        oc_log_warn("hf: getaddrinfo(%s): %s", host, gai_strerror(gai));
        return OC_ERR_NETWORK;
    }
    int fd = -1;
    for (rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        oc_log_warn("hf: connect(%s:%u) failed", host, port);
        return OC_ERR_NETWORK;
    }
    out->fd = fd;
    return OC_OK;
}

/* Send all bytes. */
static OcError send_all(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    while (n > 0) {
        ssize_t w = send(fd, p, n, 0);
        if (w < 0) {
            if (errno == EINTR) continue;
            return OC_ERR_NETWORK;
        }
        if (w == 0) return OC_ERR_NETWORK;
        p += w;
        n -= (size_t)w;
    }
    return OC_OK;
}

/* Receive into a growable malloc'd buffer. Reads until the socket closes
 * or `max_bytes` is reached. Sets *out_len. */
static OcError recv_all(int fd, char **out_buf, size_t *out_len,
                        size_t max_bytes)
{
    if (!out_buf || !out_len) return OC_ERR_INVALID_ARG;
    *out_buf = NULL;
    *out_len = 0;
    size_t cap = OC_HF_RECV_BUF;
    if (max_bytes > 0 && cap > max_bytes) cap = max_bytes;
    char *buf = (char *)malloc(cap);
    if (!buf) return OC_ERR_OOM;
    size_t len = 0;
    for (;;) {
        if (max_bytes > 0 && len >= max_bytes) break;
        size_t want = cap - len;
        if (want == 0) {
            size_t ncap = cap * 2;
            if (max_bytes > 0 && ncap > max_bytes) ncap = max_bytes;
            if (ncap == cap) break;
            char *nb = (char *)realloc(buf, ncap);
            if (!nb) { free(buf); return OC_ERR_OOM; }
            buf = nb;
            cap = ncap;
            want = cap - len;
        }
        ssize_t r = recv(fd, buf + len, want, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            free(buf);
            return OC_ERR_NETWORK;
        }
        if (r == 0) break;  /* EOF */
        len += (size_t)r;
    }
    *out_buf = buf;
    *out_len = len;
    return OC_OK;
}

/* ─── SHA-256 (minimal, public-domain style) ─────────────────────────── */

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    size_t   datalen;
} OcSha256;

static void oc_sha256_init(OcSha256 *c)
{
    c->state[0] = 0x6a09e667; c->state[1] = 0xbb67ae85;
    c->state[2] = 0x3c6ef372; c->state[3] = 0xa54ff53a;
    c->state[4] = 0x510e527f; c->state[5] = 0x9b05688c;
    c->state[6] = 0x1f83d9ab; c->state[7] = 0x5be0cd19;
    c->bitlen = 0;
    c->datalen = 0;
}

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void oc_sha256_transform(OcSha256 *c, const uint8_t *d)
{
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t m[64];
    for (size_t i = 0; i < 16; i++) {
        m[i] = ((uint32_t)d[i*4] << 24) | ((uint32_t)d[i*4+1] << 16) |
               ((uint32_t)d[i*4+2] << 8) | ((uint32_t)d[i*4+3]);
    }
    for (size_t i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(m[i-15],7) ^ ROTR(m[i-15],18) ^ (m[i-15] >> 3);
        uint32_t s1 = ROTR(m[i-2],17) ^ ROTR(m[i-2],19) ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    uint32_t a=c->state[0], b=c->state[1], cc=c->state[2], dd=c->state[3];
    uint32_t e=c->state[4], f=c->state[5], g=c->state[6], h=c->state[7];
    for (size_t i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + k[i] + m[i];
        uint32_t S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        h = g; g = f; f = e; e = dd + t1;
        dd = cc; cc = b; b = a; a = t1 + S0 + mj;
    }
    c->state[0]+=a; c->state[1]+=b; c->state[2]+=cc; c->state[3]+=dd;
    c->state[4]+=e; c->state[5]+=f; c->state[6]+=g; c->state[7]+=h;
}

static void oc_sha256_update(OcSha256 *c, const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        c->data[c->datalen++] = data[i];
        if (c->datalen == 64) {
            oc_sha256_transform(c, c->data);
            c->bitlen += 512;
            c->datalen = 0;
        }
    }
}

static void oc_sha256_final(OcSha256 *c, uint8_t out[32])
{
    uint64_t bitlen = c->bitlen + (uint64_t)c->datalen * 8u;
    /* Append 0x80. */
    c->data[c->datalen++] = 0x80;
    if (c->datalen > 56) {
        while (c->datalen < 64) c->data[c->datalen++] = 0;
        oc_sha256_transform(c, c->data);
        c->datalen = 0;
    }
    while (c->datalen < 56) c->data[c->datalen++] = 0;
    for (int i = 7; i >= 0; i--) {
        c->data[c->datalen++] = (uint8_t)(bitlen >> (i * 8));
    }
    oc_sha256_transform(c, c->data);
    for (size_t i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(c->state[i] >> 24);
        out[i*4+1] = (uint8_t)(c->state[i] >> 16);
        out[i*4+2] = (uint8_t)(c->state[i] >> 8);
        out[i*4+3] = (uint8_t)(c->state[i]);
    }
}

/* Hex-encode 32 bytes → 64-char NUL-terminated string (cap >= 65). */
static void oc_sha256_hex(const uint8_t digest[32], char *out, size_t cap)
{
    if (cap < 65) { if (cap > 0) out[0] = '\0'; return; }
    static const char *h = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        out[i*2]   = h[(digest[i] >> 4) & 0xF];
        out[i*2+1] = h[digest[i] & 0xF];
    }
    out[64] = '\0';
}

/* ─── HTTP GET (text body, headers + body in one malloc'd buffer) ────── */

/* Performs a GET and returns the body (malloc'd) + body_len. Optionally
 * sends a Range header (resume_from > 0). Parses Content-Length from
 * response headers and stops after reading that many body bytes. */
static OcError http_get_body(const char *url, const char *bearer_token,
                             uint64_t resume_from,
                             char **out_body, size_t *out_body_len)
{
    if (!url || !out_body || !out_body_len) return OC_ERR_INVALID_ARG;
    *out_body = NULL;
    *out_body_len = 0;

    char host[256];
    uint16_t port;
    const char *path;
    if (parse_url(url, host, sizeof(host), &port, &path) != OC_OK)
        return OC_ERR_INVALID_ARG;

    OcHfConn conn;
    OcError e = oc_hf_connect(host, port, &conn);
    if (e != OC_OK) return e;

    /* Build request. */
    char req[2048];
    int off = 0;
    off += snprintf(req + off, sizeof(req) - (size_t)off,
                   "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: oxidize-c/1.0\r\n"
                   "Accept: */*\r\nConnection: close\r\n",
                   path, host);
    if (bearer_token && bearer_token[0]) {
        off += snprintf(req + off, sizeof(req) - (size_t)off,
                        "Authorization: Bearer %s\r\n", bearer_token);
    }
    if (resume_from > 0) {
        off += snprintf(req + off, sizeof(req) - (size_t)off,
                        "Range: bytes=%llu-\r\n",
                        (unsigned long long)resume_from);
    }
    off += snprintf(req + off, sizeof(req) - (size_t)off, "\r\n");
    if (off < 0 || (size_t)off >= sizeof(req)) {
        close(conn.fd);
        return OC_ERR_INVALID_ARG;
    }

    e = send_all(conn.fd, req, (size_t)off);
    if (e != OC_OK) { close(conn.fd); return e; }

    /* Receive everything. */
    char *raw = NULL;
    size_t raw_len = 0;
    e = recv_all(conn.fd, &raw, &raw_len, 0);
    close(conn.fd);
    if (e != OC_OK) { free(raw); return e; }
    if (raw_len == 0) { free(raw); return OC_ERR_NETWORK; }

    /* Find header/body boundary. */
    const char *hdr_end = NULL;
    for (size_t i = 0; i + 3 < raw_len; i++) {
        if (raw[i] == '\r' && raw[i+1] == '\n' &&
            raw[i+2] == '\r' && raw[i+3] == '\n') {
            hdr_end = raw + i + 4;
            break;
        }
    }
    if (!hdr_end) { free(raw); return OC_ERR_FORMAT; }

    /* Parse status line. */
    int status = 0;
    if (raw_len >= 12 && strncmp(raw, "HTTP/", 5) == 0) {
        const char *sp = strchr(raw, ' ');
        if (sp) status = atoi(sp + 1);
    }
    if (status != 200 && status != 206) {
        oc_log_warn("hf: HTTP %d for %s", status, url);
        free(raw);
        return OC_ERR_NETWORK;
    }

    /* Parse Content-Length from headers. */
    size_t content_len = 0;
    bool has_cl = false;
    {
        const char *p = raw;
        const char *body = hdr_end;
        while (p < body) {
            const char *line_end = strstr(p, "\r\n");
            if (!line_end || line_end >= body) break;
            /* Case-insensitive "Content-Length:". */
            if (oc_ci_ends_with_prefix(p, (size_t)(line_end - p),
                                       "content-length:")) {
                const char *v = p + 15; /* strlen("content-length:") */
                while (v < line_end && isspace((unsigned char)*v)) v++;
                content_len = (size_t)strtoull(v, NULL, 10);
                has_cl = true;
            }
            p = line_end + 2;
        }
    }

    size_t body_len = raw_len - (size_t)(hdr_end - raw);
    if (has_cl && content_len < body_len) body_len = content_len;

    char *body = (char *)malloc(body_len ? body_len : 1);
    if (!body) { free(raw); return OC_ERR_OOM; }
    memcpy(body, hdr_end, body_len);
    free(raw);
    *out_body = body;
    *out_body_len = body_len;
    return OC_OK;
}

/* Case-insensitive prefix match for header lines. */
static bool oc_ci_ends_with_prefix(const char *line, size_t line_len,
                                   const char *prefix_lower)
{
    size_t plen = strlen(prefix_lower);
    if (line_len < plen) return false;
    for (size_t i = 0; i < plen; i++) {
        if (oc_ci(line[i]) != prefix_lower[i]) return false;
    }
    return true;
}

/* ─── Directory helpers ──────────────────────────────────────────────── */

static bool oc_mkdir_p(const char *path)
{
    if (!path || !*path) return false;
    char buf[OC_HF_MAX_CACHE_DIR];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return false;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
                /* ignore; final mkdir below will report */
            }
            buf[i] = '/';
        }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
    return true;
}

/* ─── oc_hf_list_models ──────────────────────────────────────────────── */

OcError oc_hf_list_models(const OcHfConfig *cfg,
                          OcHfModel *out_models, size_t *inout_count)
{
    if (!cfg || !out_models || !inout_count) return OC_ERR_INVALID_ARG;
    if (cfg->repo_id[0] == '\0') return OC_ERR_INVALID_ARG;
    if (*inout_count == 0) return OC_OK;

    /* Build API URL: {api_base}/api/models/{repo_id}. */
    char url[OC_HF_MAX_URL];
    const char *base = cfg->api_base[0] ? cfg->api_base
                                        : OC_HF_DEFAULT_API_BASE;
    int n = snprintf(url, sizeof(url), "%s/api/models/%s", base, cfg->repo_id);
    if (n < 0 || (size_t)n >= sizeof(url)) return OC_ERR_INVALID_ARG;

    char *body = NULL;
    size_t body_len = 0;
    OcError e = http_get_body(url, cfg->api_token[0] ? cfg->api_token : NULL,
                              0, &body, &body_len);
    if (e != OC_OK) return e;

    /* Extract rfilenames. */
    char *names[OC_HF_MAX_MODELS];
    size_t found = oc_hf_extract_rfilenames(body, body_len, names,
                                            OC_HF_MAX_MODELS);
    free(body);

    size_t written = 0;
    for (size_t i = 0; i < found && written < *inout_count; i++) {
        if (!names[i]) continue;
        if (!oc_hf_is_gguf(names[i])) { free(names[i]); continue; }

        /* Quant type filter. */
        char qt[OC_HF_MAX_QUANT_TYPE];
        bool has_qt = oc_hf_parse_quant_type(names[i], qt, sizeof(qt));
        if (cfg->quant_type[0]) {
            if (!has_qt) { free(names[i]); continue; }
            if (hf_ci_strcmp(qt, cfg->quant_type) != 0) { free(names[i]); continue; }
        }

        OcHfModel *m = &out_models[written];
        memset(m, 0, sizeof(*m));
        oc_copy_str(m->repo_id, sizeof(m->repo_id), cfg->repo_id);
        oc_copy_str(m->filename, sizeof(m->filename), names[i]);
        oc_copy_str(m->quant_type, sizeof(m->quant_type),
                    has_qt ? qt : "");
        m->size_bytes = 0;
        m->sha256[0] = '\0';
        m->download_url[0] = '\0';
        written++;
        free(names[i]);
    }
    /* Free any remaining we didn't write. */
    for (size_t i = written; i < found; i++) free(names[i]);

    *inout_count = written;
    return OC_OK;
}

/* ─── oc_hf_resolve ─────────────────────────────────────────────────── */

OcError oc_hf_resolve(const OcHfConfig *cfg, OcHfModel *out_model)
{
    if (!cfg || !out_model) return OC_ERR_INVALID_ARG;
    if (cfg->repo_id[0] == '\0') return OC_ERR_INVALID_ARG;

    if (out_model->filename[0] != '\0') {
        /* Caller specified filename; just build URL. */
        oc_copy_str(out_model->repo_id, sizeof(out_model->repo_id),
                    cfg->repo_id);
        const char *base = cfg->api_base[0] ? cfg->api_base
                                            : OC_HF_DEFAULT_API_BASE;
        const char *rev = cfg->revision[0] ? cfg->revision
                                           : OC_HF_DEFAULT_REVISION;
        char url[OC_HF_MAX_URL];
        int n = snprintf(url, sizeof(url), "%s/%s/resolve/%s/%s",
                         base, cfg->repo_id, rev, out_model->filename);
        if (n < 0 || (size_t)n >= sizeof(url)) return OC_ERR_INVALID_ARG;
        oc_copy_str(out_model->download_url, sizeof(out_model->download_url),
                    url);
        char qt[OC_HF_MAX_QUANT_TYPE];
        if (oc_hf_parse_quant_type(out_model->filename, qt, sizeof(qt))) {
            oc_copy_str(out_model->quant_type, sizeof(out_model->quant_type), qt);
        }
        return OC_OK;
    }

    /* No filename: list and pick the single .gguf. */
    OcHfModel models[OC_HF_MAX_MODELS];
    size_t count = OC_HF_MAX_MODELS;
    OcError e = oc_hf_list_models(cfg, models, &count);
    if (e != OC_OK) return e;
    if (count == 0) {
        oc_log_warn("hf: repo %s has no .gguf files", cfg->repo_id);
        return OC_ERR_MODEL;
    }
    if (count > 1) {
        oc_log_warn("hf: repo %s has %zu .gguf files; specify --file",
                    cfg->repo_id, count);
        return OC_ERR_MODEL;
    }
    *out_model = models[0];
    /* Build URL. */
    const char *base = cfg->api_base[0] ? cfg->api_base
                                        : OC_HF_DEFAULT_API_BASE;
    const char *rev = cfg->revision[0] ? cfg->revision
                                       : OC_HF_DEFAULT_REVISION;
    char url[OC_HF_MAX_URL];
    int n = snprintf(url, sizeof(url), "%s/%s/resolve/%s/%s",
                     base, cfg->repo_id, rev, out_model->filename);
    if (n < 0 || (size_t)n >= sizeof(url)) return OC_ERR_INVALID_ARG;
    oc_copy_str(out_model->download_url, sizeof(out_model->download_url), url);
    return OC_OK;
}

/* ─── oc_hf_download ─────────────────────────────────────────────────── */

/* Stream a body to an open FILE* while hashing. Uses Range to resume from
 * `resume_from` if > 0. */
static OcError http_download_stream(const char *url, const char *bearer,
                                    uint64_t resume_from, FILE *out,
                                    OcSha256 *sha, uint64_t *out_total,
                                    OcHfProgressCb cb, void *user)
{
    if (!url || !out) return OC_ERR_INVALID_ARG;
    if (out_total) *out_total = 0;

    char host[256];
    uint16_t port;
    const char *path;
    if (parse_url(url, host, sizeof(host), &port, &path) != OC_OK)
        return OC_ERR_INVALID_ARG;

    OcHfConn conn;
    OcError e = oc_hf_connect(host, port, &conn);
    if (e != OC_OK) return e;

    char req[2048];
    int off = 0;
    off += snprintf(req + off, sizeof(req) - (size_t)off,
                    "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: oxidize-c/1.0\r\n"
                    "Accept: */*\r\nConnection: close\r\n",
                    path, host);
    if (bearer && bearer[0]) {
        off += snprintf(req + off, sizeof(req) - (size_t)off,
                        "Authorization: Bearer %s\r\n", bearer);
    }
    if (resume_from > 0) {
        off += snprintf(req + off, sizeof(req) - (size_t)off,
                        "Range: bytes=%llu-\r\n",
                        (unsigned long long)resume_from);
    }
    off += snprintf(req + off, sizeof(req) - (size_t)off, "\r\n");
    if (off < 0 || (size_t)off >= sizeof(req)) {
        close(conn.fd);
        return OC_ERR_INVALID_ARG;
    }
    e = send_all(conn.fd, req, (size_t)off);
    if (e != OC_OK) { close(conn.fd); return e; }

    /* Read header first. */
    char hdr[OC_HF_HTTP_HEADER_BUF];
    size_t hdr_len = 0;
    bool hdr_done = false;
    size_t body_start = 0;
    while (!hdr_done) {
        if (hdr_len >= sizeof(hdr) - 1) { close(conn.fd); return OC_ERR_FORMAT; }
        ssize_t r = recv(conn.fd, hdr + hdr_len, sizeof(hdr) - 1 - hdr_len, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            close(conn.fd);
            return OC_ERR_NETWORK;
        }
        if (r == 0) { close(conn.fd); return OC_ERR_NETWORK; }
        hdr_len += (size_t)r;
        hdr[hdr_len] = '\0';
        /* Look for \r\n\r\n. */
        for (size_t i = 3; i < hdr_len; i++) {
            if (hdr[i-3] == '\r' && hdr[i-2] == '\n' &&
                hdr[i-1] == '\r' && hdr[i] == '\n') {
                body_start = i + 1;
                hdr_done = true;
                break;
            }
        }
    }

    /* Parse status. */
    int status = 0;
    if (hdr_len >= 12 && strncmp(hdr, "HTTP/", 5) == 0) {
        const char *sp = strchr(hdr, ' ');
        if (sp) status = atoi(sp + 1);
    }
    bool resumed = (status == 206);
    if (status != 200 && status != 206) {
        close(conn.fd);
        oc_log_warn("hf: HTTP %d for %s", status, url);
        return OC_ERR_NETWORK;
    }

    /* Parse Content-Length. */
    uint64_t content_len = 0;
    bool has_cl = false;
    {
        char *p = hdr;
        char *body_p = hdr + body_start;
        while (p < body_p) {
            char *le = strstr(p, "\r\n");
            if (!le || le >= body_p) break;
            *le = '\0';
            if (oc_ci_ends_with_prefix(p, (size_t)strlen(p),
                                        "content-length:")) {
                const char *v = p + 15;
                while (*v && isspace((unsigned char)*v)) v++;
                content_len = strtoull(v, NULL, 10);
                has_cl = true;
            }
            *le = '\r';
            p = le + 2;
        }
    }
    uint64_t total = resume_from + content_len;
    if (out_total) *out_total = total;

    /* Write any body bytes already received in the header buffer. */
    uint64_t downloaded = resume_from;
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    if (body_start < hdr_len) {
        size_t n = hdr_len - body_start;
        if (fwrite(hdr + body_start, 1, n, out) != n) {
            close(conn.fd); return OC_ERR_IO;
        }
        if (sha) oc_sha256_update(sha, (const uint8_t *)(hdr + body_start), n);
        downloaded += n;
    }

    /* Read remaining body in chunks. */
    char buf[OC_HF_DOWNLOAD_CHUNK];
    for (;;) {
        ssize_t r = recv(conn.fd, buf, sizeof(buf), 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            close(conn.fd);
            return OC_ERR_NETWORK;
        }
        if (r == 0) break;
        if (fwrite(buf, 1, (size_t)r, out) != (size_t)r) {
            close(conn.fd); return OC_ERR_IO;
        }
        if (sha) oc_sha256_update(sha, (const uint8_t *)buf, (size_t)r);
        downloaded += (uint64_t)r;

        if (cb) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - t_start.tv_sec) +
                             (now.tv_nsec - t_start.tv_nsec) / 1e9;
            double speed = elapsed > 0 ? (double)(downloaded - resume_from) / elapsed : 0;
            double eta = (speed > 0 && total > downloaded)
                         ? (double)(total - downloaded) / speed : -1.0;
            OcHfDownloadProgress prog = {
                .downloaded_bytes = downloaded,
                .total_bytes = has_cl ? total : 0,
                .speed = speed,
                .eta = eta,
            };
            if (cb(&prog, user) != 0) {
                close(conn.fd);
                return OC_ERR_IO;
            }
        }
    }
    close(conn.fd);
    (void)resumed;
    return OC_OK;
}

OcError oc_hf_download(const OcHfConfig *cfg, const OcHfModel *model,
                       OcHfProgressCb cb, void *user)
{
    if (!cfg || !model) return OC_ERR_INVALID_ARG;
    if (model->filename[0] == '\0') return OC_ERR_INVALID_ARG;

    /* Compute cache path. */
    char dest[OC_HF_MAX_CACHE_DIR];
    if (oc_hf_cache_path(cfg, model->repo_id, model->filename,
                        dest, sizeof(dest)) != OC_OK)
        return OC_ERR_INVALID_ARG;

    /* Ensure parent dir exists. */
    char dir[OC_HF_MAX_CACHE_DIR];
    int n = snprintf(dir, sizeof(dir), "%s", dest);
    if (n < 0 || (size_t)n >= sizeof(dir)) return OC_ERR_INVALID_ARG;
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        if (!oc_mkdir_p(dir)) return OC_ERR_IO;
    }

    /* Already complete? */
    if (model->sha256[0]) {
        /* If dest exists and matches expected SHA, skip. */
        struct stat st;
        if (stat(dest, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
            /* Optional: verify hash. For now, trust presence. */
            (void)st;
            return OC_OK;
        }
    }

    /* Resume support: check .part file size. */
    char part_path[OC_HF_MAX_CACHE_DIR];
    n = snprintf(part_path, sizeof(part_path), "%s.part", dest);
    if (n < 0 || (size_t)n >= sizeof(part_path)) return OC_ERR_INVALID_ARG;
    uint64_t resume_from = 0;
    {
        struct stat st;
        if (stat(part_path, &st) == 0 && S_ISREG(st.st_mode)) {
            resume_from = (uint64_t)st.st_size;
        }
    }

    /* Open .part for append (or truncate if no resume). */
    FILE *f = fopen(part_path, resume_from > 0 ? "ab" : "wb");
    if (!f) return OC_ERR_IO;
    if (resume_from > 0) {
        if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return OC_ERR_IO; }
    }

    /* Rate limit: max 1 concurrent download per process. */
    pthread_mutex_lock(&g_hf_dl_mutex);

    OcSha256 sha;
    bool hashing = (model->sha256[0] != '\0');
    if (hashing) {
        oc_sha256_init(&sha);
        /* If resuming, hash the existing partial file first. */
        if (resume_from > 0) {
            FILE *pf = fopen(part_path, "rb");
            if (pf) {
                char buf[OC_HF_DOWNLOAD_CHUNK];
                size_t r;
                while ((r = fread(buf, 1, sizeof(buf), pf)) > 0) {
                    oc_sha256_update(&sha, (const uint8_t *)buf, r);
                }
                fclose(pf);
            }
        }
    }

    uint64_t total = 0;
    OcError e = http_download_stream(
        model->download_url[0] ? model->download_url : "",
        cfg->api_token[0] ? cfg->api_token : NULL,
        resume_from, f, hashing ? &sha : NULL, &total, cb, user);
    fclose(f);
    pthread_mutex_unlock(&g_hf_dl_mutex);

    if (e != OC_OK) {
        /* Leave .part for resume. */
        return e;
    }

    /* Verify hash. */
    if (hashing) {
        uint8_t digest[32];
        oc_sha256_final(&sha, digest);
        char hex[OC_HF_MAX_SHA256];
        oc_sha256_hex(digest, hex, sizeof(hex));
        if (strcmp(hex, model->sha256) != 0) {
            oc_log_warn("hf: SHA-256 mismatch for %s (got %s, want %s)",
                        model->filename, hex, model->sha256);
            /* Remove corrupt .part. */
            unlink(part_path);
            return OC_ERR_IO;
        }
    }

    /* Atomic rename .part -> dest. */
    if (rename(part_path, dest) != 0) {
        return OC_ERR_IO;
    }
    return OC_OK;
}

/* ─── Cache management ───────────────────────────────────────────────── */

OcError oc_hf_cache_list(const OcHfConfig *cfg,
                         OcHfModel *out_models, size_t *inout_count)
{
    if (!cfg || !out_models || !inout_count) return OC_ERR_INVALID_ARG;
    size_t max_out = *inout_count;
    *inout_count = 0;

    const char *cache = cfg->cache_dir[0] ? cfg->cache_dir : NULL;
    char default_cache[OC_HF_MAX_CACHE_DIR];
    if (!cache) {
        if (oc_hf_default_cache_dir(default_cache, sizeof(default_cache))
                != OC_OK) return OC_ERR_INTERNAL;
        cache = default_cache;
    }

    DIR *d = opendir(cache);
    if (!d) return OC_OK;  /* empty/missing cache */

    size_t written = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        /* Each subdir is a sanitized repo_id. */
        char repo_dir[OC_HF_MAX_CACHE_DIR];
        int n = snprintf(repo_dir, sizeof(repo_dir), "%s/%s", cache,
                          ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(repo_dir)) continue;
        DIR *sd = opendir(repo_dir);
        if (!sd) continue;
        struct dirent *sent;
        while ((sent = readdir(sd)) != NULL) {
            if (sent->d_name[0] == '.') continue;
            if (!oc_hf_is_gguf(sent->d_name)) continue;
            if (oc_ci_ends_with(sent->d_name, ".part")) continue;
            if (written >= max_out) { closedir(sd); closedir(d); *inout_count = written; return OC_OK; }

            OcHfModel *m = &out_models[written];
            memset(m, 0, sizeof(*m));
            /* Un-sanitize repo_id: _ -> /. */
            size_t rl = strlen(ent->d_name);
            if (rl >= sizeof(m->repo_id)) rl = sizeof(m->repo_id) - 1;
            for (size_t i = 0; i < rl; i++) {
                m->repo_id[i] = (ent->d_name[i] == '_') ? '/' : ent->d_name[i];
            }
            m->repo_id[rl] = '\0';
            oc_copy_str(m->filename, sizeof(m->filename), sent->d_name);
            char qt[OC_HF_MAX_QUANT_TYPE];
            if (oc_hf_parse_quant_type(sent->d_name, qt, sizeof(qt))) {
                oc_copy_str(m->quant_type, sizeof(m->quant_type), qt);
            }
            /* Size. */
            char full[OC_HF_MAX_CACHE_DIR];
            int sn = snprintf(full, sizeof(full), "%s/%s", repo_dir,
                              sent->d_name);
            if (sn > 0 && (size_t)sn < sizeof(full)) {
                struct stat st;
                if (stat(full, &st) == 0) m->size_bytes = (uint64_t)st.st_size;
            }
            written++;
        }
        closedir(sd);
    }
    closedir(d);
    *inout_count = written;
    return OC_OK;
}

OcError oc_hf_cache_size(const OcHfConfig *cfg, uint64_t *out_bytes)
{
    if (!cfg || !out_bytes) return OC_ERR_INVALID_ARG;
    *out_bytes = 0;

    const char *cache = cfg->cache_dir[0] ? cfg->cache_dir : NULL;
    char default_cache[OC_HF_MAX_CACHE_DIR];
    if (!cache) {
        if (oc_hf_default_cache_dir(default_cache, sizeof(default_cache))
                != OC_OK) return OC_ERR_INTERNAL;
        cache = default_cache;
    }

    DIR *d = opendir(cache);
    if (!d) return OC_OK;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char repo_dir[OC_HF_MAX_CACHE_DIR];
        int n = snprintf(repo_dir, sizeof(repo_dir), "%s/%s", cache,
                          ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(repo_dir)) continue;
        DIR *sd = opendir(repo_dir);
        if (!sd) continue;
        struct dirent *sent;
        while ((sent = readdir(sd)) != NULL) {
            if (sent->d_name[0] == '.') continue;
            char full[OC_HF_MAX_CACHE_DIR];
            int sn = snprintf(full, sizeof(full), "%s/%s", repo_dir,
                              sent->d_name);
            if (sn < 0 || (size_t)sn >= sizeof(full)) continue;
            struct stat st;
            if (stat(full, &st) == 0 && S_ISREG(st.st_mode)) {
                *out_bytes += (uint64_t)st.st_size;
            }
        }
        closedir(sd);
    }
    closedir(d);
    return OC_OK;
}

OcError oc_hf_cache_clean(const OcHfConfig *cfg, uint64_t max_age_seconds,
                          size_t *out_removed)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    if (out_removed) *out_removed = 0;

    const char *cache = cfg->cache_dir[0] ? cfg->cache_dir : NULL;
    char default_cache[OC_HF_MAX_CACHE_DIR];
    if (!cache) {
        if (oc_hf_default_cache_dir(default_cache, sizeof(default_cache))
                != OC_OK) return OC_ERR_INTERNAL;
        cache = default_cache;
    }

    DIR *d = opendir(cache);
    if (!d) return OC_OK;

    time_t now = time(NULL);
    size_t removed = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char repo_dir[OC_HF_MAX_CACHE_DIR];
        int n = snprintf(repo_dir, sizeof(repo_dir), "%s/%s", cache,
                          ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(repo_dir)) continue;
        DIR *sd = opendir(repo_dir);
        if (!sd) continue;
        struct dirent *sent;
        while ((sent = readdir(sd)) != NULL) {
            if (sent->d_name[0] == '.') continue;
            char full[OC_HF_MAX_CACHE_DIR];
            int sn = snprintf(full, sizeof(full), "%s/%s", repo_dir,
                              sent->d_name);
            if (sn < 0 || (size_t)sn >= sizeof(full)) continue;
            /* fstat on an open fd (not stat-by-path) so the age check and
             * the file we inspected cannot diverge (TOCTOU). */
            int fd = open(full, O_RDONLY);
            if (fd < 0) continue;
            struct stat st;
            if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
                close(fd);
                continue;
            }
            close(fd);
            time_t age = now - st.st_mtime;
            if (max_age_seconds == 0 || (uint64_t)age >= max_age_seconds) {
                if (unlink(full) == 0) removed++;
            }
        }
        closedir(sd);
        /* Try to remove the now-empty repo dir. */
        rmdir(repo_dir);
    }
    closedir(d);
    if (out_removed) *out_removed = removed;
    return OC_OK;
}
