/* k8s.c — Kubernetes integration. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "oxidize/k8s.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef NI_MAXHOST
#define NI_MAXHOST 1025
#endif
#ifndef NI_MAXSERV
#define NI_MAXSERV 32
#endif

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_k8s_init(OcK8sCluster *cluster, const char *namespace,
                   const char *service_name)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    memset(cluster, 0, sizeof(*cluster));
    if (namespace) copy_str(cluster->namespace, sizeof(cluster->namespace), namespace);
    else strcpy(cluster->namespace, "default");
    if (service_name) copy_str(cluster->service_name, sizeof(cluster->service_name), service_name);
    cluster->available = false;
    return OC_OK;
}

OcError oc_k8s_detect(OcK8sCluster *cluster)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    /* Check for KUBERNETES_SERVICE_HOST env var. */
    const char *host = getenv("KUBERNETES_SERVICE_HOST");
    cluster->available = (host && *host != '\0');
    return OC_OK;
}

OcError oc_k8s_add_pod(OcK8sCluster *cluster, const char *name,
                      const char *ip, uint16_t port)
{
    if (!cluster || !name) return OC_ERR_INVALID_ARG;
    if (cluster->n_pods >= OC_K8S_MAX_PODS) return OC_ERR_OOM;

    OcK8sPod *pod = &cluster->pods[cluster->n_pods];
    memset(pod, 0, sizeof(*pod));
    copy_str(pod->name, sizeof(pod->name), name);
    if (ip) copy_str(pod->ip, sizeof(pod->ip), ip);
    pod->port = port;
    pod->ready = false;
    pod->restarts = 0;
    pod->age_sec = 0;
    cluster->n_pods++;
    return OC_OK;
}

OcError oc_k8s_get_pods(const OcK8sCluster *cluster, const OcK8sPod **out, uint32_t *count)
{
    if (!cluster || !out || !count) return OC_ERR_INVALID_ARG;
    *out = cluster->pods;
    *count = cluster->n_pods;
    return OC_OK;
}

OcError oc_k8s_get_ready_pods(const OcK8sCluster *cluster, OcK8sPod *out,
                              uint32_t cap, uint32_t *count)
{
    if (!cluster || !out || !count) return OC_ERR_INVALID_ARG;
    uint32_t n_ready = oc_k8s_n_ready(cluster);
    if (n_ready > cap) {
        /* Report the required size so the caller can size a buffer. */
        *count = n_ready;
        return OC_ERR_OOM;
    }
    uint32_t w = 0;
    for (uint32_t i = 0; i < cluster->n_pods; i++) {
        if (cluster->pods[i].ready) out[w++] = cluster->pods[i];
    }
    *count = w;
    return OC_OK;
}

uint32_t oc_k8s_n_pods(const OcK8sCluster *cluster)
{
    return cluster ? cluster->n_pods : 0;
}

uint32_t oc_k8s_n_ready(const OcK8sCluster *cluster)
{
    if (!cluster) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < cluster->n_pods; i++)
        if (cluster->pods[i].ready) count++;
    return count;
}

bool oc_k8s_is_available(const OcK8sCluster *cluster)
{
    return cluster ? cluster->available : false;
}

static bool host_char_ok(unsigned char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
        || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == ':'
        || c == '[' || c == ']' || c == '%';
}

static bool copy_validated_host(const char *src, size_t len, char *dst,
                                  size_t cap)
{
    if (!src || len == 0 || !dst || cap == 0 || len >= cap) return false;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)src[i];
        if (!host_char_ok(c)) return false;
        dst[i] = (char)c;
    }
    dst[len] = '\0';
    return true;
}

static bool copy_validated_port(const char *src, size_t len, char *dst,
                                  size_t cap)
{
    if (!src || len == 0 || !dst || cap < 2) return false;
    unsigned long v = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c < '0' || c > '9') return false;
        v = v * 10ul + (unsigned)(c - '0');
        if (v > 65535ul) return false;
    }
    if (v == 0) return false;
    int n = snprintf(dst, cap, "%lu", v);
    return n > 0 && (size_t)n < cap;
}

static bool strip_ipv6_brackets(char *host)
{
    size_t n = strlen(host);
    if (n < 2 || host[0] != '[' || host[n - 1] != ']') return host[0] != '\0';
    memmove(host, host + 1, n - 2);
    host[n - 2] = '\0';
    return host[0] != '\0';
}

static bool parse_url_hostport(const char *p, char *host, size_t host_cap,
                                 char *port, size_t port_cap)
{
    const char *host_start;
    const char *host_end;
    const char *port_start = NULL;

    if (*p == '[') {
        const char *rbr = strchr(p, ']');
        if (!rbr || rbr == p + 1) return false;
        host_start = p + 1;
        host_end = rbr;
        if (rbr[1] == ':')
            port_start = rbr + 2;
        else if (rbr[1] != '\0' && rbr[1] != '/')
            return false;
    } else {
        const char *slash = strchr(p, '/');
        const char *colon = strchr(p, ':');
        if (slash && colon && colon > slash) colon = NULL;
        host_start = p;
        host_end = colon ? colon : (slash ? slash : p + strlen(p));
        if (colon) port_start = colon + 1;
    }

    size_t hlen = (size_t)(host_end - host_start);
    if (!copy_validated_host(host_start, hlen, host, host_cap)) return false;
    if (port_start) {
        size_t plen = 0;
        while (port_start[plen] && port_start[plen] != '/') plen++;
        if (!copy_validated_port(port_start, plen, port, port_cap)) return false;
    } else if (snprintf(port, port_cap, "80") < 0) {
        return false;
    }
    return port[0] != '\0';
}

/* Resolve the plaintext API base as host + port. Returns false if no
 * endpoint is configured. Env values are allowlisted so they cannot be
 * copied into the HTTP request as-is (CodeQL CWE-497). */
static bool k8s_api_endpoint(char *host, size_t host_cap, char *port,
                             size_t port_cap)
{
    const char *url = getenv("OC_K8S_API_URL");
    if (url && *url) {
        const char *p = strstr(url, "://");
        p = p ? p + 3 : url;
        return parse_url_hostport(p, host, host_cap, port, port_cap);
    }

    const char *h = getenv("KUBERNETES_SERVICE_HOST");
    const char *pt = getenv("KUBERNETES_SERVICE_PORT");
    if (!h || !*h) return false;
    if (!copy_validated_host(h, strlen(h), host, host_cap)) return false;
    if (!strip_ipv6_brackets(host)) return false;
    if (pt && *pt) {
        if (!copy_validated_port(pt, strlen(pt), port, port_cap)) return false;
    } else if (snprintf(port, port_cap, "443") < 0) {
        return false;
    }
    return true;
}

static bool peer_is_loopback(int fd)
{
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &len) != 0) return false;
    if (ss.ss_family == AF_INET) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)&ss;
        return ntohl(in->sin_addr.s_addr) == INADDR_LOOPBACK;
    }
    if (ss.ss_family == AF_INET6) {
        const struct sockaddr_in6 *in6 = (const struct sockaddr_in6 *)&ss;
        return IN6_IS_ADDR_LOOPBACK(&in6->sin6_addr);
    }
    return false;
}

/* HTTP Host from the connected peer, not from getenv. */
static bool format_http_host(int fd, char *out, size_t cap)
{
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &len) != 0) return false;
    char hbuf[NI_MAXHOST], pbuf[NI_MAXSERV];
    if (getnameinfo((struct sockaddr *)&ss, len, hbuf, sizeof(hbuf), pbuf,
                    sizeof(pbuf), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return false;
    }
    int n;
    if (ss.ss_family == AF_INET6)
        n = snprintf(out, cap, "[%s]:%s", hbuf, pbuf);
    else
        n = snprintf(out, cap, "%s:%s", hbuf, pbuf);
    return n > 0 && (size_t)n < cap;
}

/* Read the in-cluster service account token, if mounted. Returns the
 * number of bytes read (0 if unavailable). */
static size_t k8s_read_token(char *out, size_t cap)
{
    if (cap == 0) return 0;
    out[0] = '\0';
    FILE *f = fopen("/var/run/secrets/kubernetes.io/serviceaccount/token", "r");
    if (!f) return 0;
    size_t n = fread(out, 1, cap - 1, f);
    fclose(f);
    /* Trim trailing whitespace/newlines — they would corrupt the header. */
    while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r' ||
                     out[n - 1] == ' ')) n--;
    out[n] = '\0';
    return n;
}

static int k8s_connect(const char *host, const char *port)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res) return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static bool write_all(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        off += (size_t)n;
    }
    return true;
}

static void secure_wipe(void *p, size_t n)
{
    volatile unsigned char *v = (volatile unsigned char *)p;
    while (n > 0) {
        *v++ = 0;
        n--;
    }
}

OcError oc_k8s_scale(const OcK8sCluster *cluster, uint32_t target_replicas)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    if (!cluster->available) return OC_ERR_MODEL;
    if (cluster->service_name[0] == '\0') return OC_ERR_INVALID_ARG;

    char host[256], port[16];
    if (!k8s_api_endpoint(host, sizeof(host), port, sizeof(port)))
        return OC_ERR_NETWORK;

    int fd = k8s_connect(host, port);
    if (fd < 0) return OC_ERR_NETWORK;

    char http_host[NI_MAXHOST + NI_MAXSERV + 8];
    if (!format_http_host(fd, http_host, sizeof(http_host))) {
        close(fd);
        return OC_ERR_NETWORK;
    }

    char body[64];
    int body_len = snprintf(body, sizeof(body),
                            "{\"spec\":{\"replicas\":%u}}",
                            (unsigned)target_replicas);

    char token[8192];
    char req[16384];
    size_t token_len = 0;
    OcError err = OC_OK;
    /* The API server is HTTPS; this client is plaintext HTTP. Only attach
     * the in-cluster token when talking to a loopback proxy. */
    if (peer_is_loopback(fd))
        token_len = k8s_read_token(token, sizeof(token));
    else
        token[0] = '\0';

    /* Deployment name defaults to the service name — the common 1:1
     * Deployment/Service naming for an inference StatefulSet or Deployment. */
    int n = snprintf(req, sizeof(req),
        "PATCH /apis/apps/v1/namespaces/%s/deployments/%s/scale HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Content-Type: application/merge-patch+json\r\n"
        "Accept: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "%s%s%s"
        "\r\n"
        "%s",
        cluster->namespace, cluster->service_name, http_host, body_len,
        token_len ? "Authorization: Bearer " : "",
        token_len ? token : "",
        token_len ? "\r\n" : "",
        body);
    if (n < 0 || (size_t)n >= sizeof(req)) {
        err = OC_ERR_INVALID_ARG;
        goto wipe;
    }

    if (!write_all(fd, req, (size_t)n)) {
        err = OC_ERR_NETWORK;
        goto wipe;
    }

    /* Only the status line matters. */
    char resp[256];
    ssize_t got = read(fd, resp, sizeof(resp) - 1);
    if (got <= 0) {
        err = OC_ERR_NETWORK;
        goto wipe;
    }
    resp[got] = '\0';

    unsigned status = 0;
    if (sscanf(resp, "HTTP/1.%*u %u", &status) != 1)
        err = OC_ERR_NETWORK;
    else
        err = (status >= 200 && status < 300) ? OC_OK : OC_ERR_MODEL;

wipe:
    secure_wipe(token, sizeof(token));
    secure_wipe(req, sizeof(req));
    close(fd);
    return err;
}

OcError oc_k8s_mark_pod_ready(OcK8sCluster *cluster, const char *name)
{
    if (!cluster || !name) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < cluster->n_pods; i++) {
        if (strcmp(cluster->pods[i].name, name) == 0) {
            cluster->pods[i].ready = true;
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

void oc_k8s_free(OcK8sCluster *cluster)
{
    if (!cluster) return;
    memset(cluster, 0, sizeof(*cluster));
}
