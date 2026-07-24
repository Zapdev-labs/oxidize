/*
 * k8s.c — Kubernetes integration.
 *
 * Pod bookkeeping is in-process; oc_k8s_scale() issues a real HTTP request
 * to the Kubernetes API server (a merge-patch on the Deployment's /scale
 * subresource).
 *
 * The API server speaks HTTPS and this port links only libc, so the
 * endpoint must be plaintext HTTP. In practice that means either
 * `kubectl proxy` (OC_K8S_API_URL=http://127.0.0.1:8001) or an in-cluster
 * sidecar/proxy. Without one, scaling reports OC_ERR_NETWORK rather than
 * silently claiming success.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/k8s.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

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

/* Resolve the plaintext API base as host + port. Returns false if no
 * endpoint is configured. */
static bool k8s_api_endpoint(char *host, size_t host_cap, char *port,
                             size_t port_cap)
{
    const char *url = getenv("OC_K8S_API_URL");
    if (url && *url) {
        /* Accept "http://host:port", "host:port", or "host". */
        const char *p = strstr(url, "://");
        p = p ? p + 3 : url;
        const char *colon = strrchr(p, ':');
        const char *slash = strchr(p, '/');
        if (slash && colon && colon > slash) colon = NULL; /* colon in path */
        size_t hlen = colon ? (size_t)(colon - p)
                            : (slash ? (size_t)(slash - p) : strlen(p));
        if (hlen == 0 || hlen >= host_cap) return false;
        memcpy(host, p, hlen);
        host[hlen] = '\0';
        if (colon) {
            size_t plen = 0;
            const char *q = colon + 1;
            while (q[plen] && q[plen] != '/' && plen + 1 < port_cap) plen++;
            memcpy(port, q, plen);
            port[plen] = '\0';
        } else {
            snprintf(port, port_cap, "80");
        }
        return port[0] != '\0';
    }

    const char *h = getenv("KUBERNETES_SERVICE_HOST");
    const char *pt = getenv("KUBERNETES_SERVICE_PORT");
    if (!h || !*h) return false;
    snprintf(host, host_cap, "%s", h);
    snprintf(port, port_cap, "%s", (pt && *pt) ? pt : "443");
    return true;
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

    char body[64];
    int body_len = snprintf(body, sizeof(body),
                            "{\"spec\":{\"replicas\":%u}}",
                            (unsigned)target_replicas);

    char token[8192];
    size_t token_len = k8s_read_token(token, sizeof(token));

    /* Deployment name defaults to the service name — the common 1:1
     * Deployment/Service naming for an inference StatefulSet or Deployment. */
    char req[16384];
    int n = snprintf(req, sizeof(req),
        "PATCH /apis/apps/v1/namespaces/%s/deployments/%s/scale HTTP/1.1\r\n"
        "Host: %s:%s\r\n"
        "Content-Type: application/merge-patch+json\r\n"
        "Accept: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "%s%s%s"
        "\r\n"
        "%s",
        cluster->namespace, cluster->service_name, host, port, body_len,
        token_len ? "Authorization: Bearer " : "",
        token_len ? token : "",
        token_len ? "\r\n" : "",
        body);
    if (n < 0 || (size_t)n >= sizeof(req)) {
        close(fd);
        return OC_ERR_INVALID_ARG;
    }

    if (!write_all(fd, req, (size_t)n)) {
        close(fd);
        return OC_ERR_NETWORK;
    }

    /* Only the status line matters. */
    char resp[256];
    ssize_t got = read(fd, resp, sizeof(resp) - 1);
    close(fd);
    if (got <= 0) return OC_ERR_NETWORK;
    resp[got] = '\0';

    unsigned status = 0;
    if (sscanf(resp, "HTTP/1.%*u %u", &status) != 1) return OC_ERR_NETWORK;
    return (status >= 200 && status < 300) ? OC_OK : OC_ERR_MODEL;
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
