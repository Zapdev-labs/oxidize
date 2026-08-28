/* test_k8s.c — Kubernetes integration tests. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* INADDR_LOOPBACK is behind the BSD surface on macOS */
#endif
#include <criterion/criterion.h>
#include "oxidize/k8s.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

Test(k8s, init)
{
    OcK8sCluster cluster;
    cr_assert_eq(oc_k8s_init(&cluster, "default", "oxidize"), OC_OK);
    cr_assert_str_eq(cluster.namespace, "default");
    cr_assert_str_eq(cluster.service_name, "oxidize");
    cr_assert_eq(cluster.n_pods, 0);
    cr_assert(!cluster.available);
    oc_k8s_free(&cluster);
}

Test(k8s, init_null_ns)
{
    OcK8sCluster cluster;
    cr_assert_eq(oc_k8s_init(&cluster, NULL, NULL), OC_OK);
    cr_assert_str_eq(cluster.namespace, "default");
    oc_k8s_free(&cluster);
}

Test(k8s, init_null)
{
    cr_assert_neq(oc_k8s_init(NULL, NULL, NULL), OC_OK);
}

Test(k8s, detect)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_detect(&cluster), OC_OK);
    oc_k8s_free(&cluster);
}

Test(k8s, add_pod)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_add_pod(&cluster, "pod-1", "10.0.0.1", 8080), OC_OK);
    cr_assert_eq(cluster.n_pods, 1);
    cr_assert_str_eq(cluster.pods[0].name, "pod-1");
    cr_assert_str_eq(cluster.pods[0].ip, "10.0.0.1");
    cr_assert_eq(cluster.pods[0].port, 8080);
    oc_k8s_free(&cluster);
}

Test(k8s, add_pod_null)
{
    cr_assert_neq(oc_k8s_add_pod(NULL, NULL, NULL, 0), OC_OK);
}

Test(k8s, get_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip1", 80);
    oc_k8s_add_pod(&cluster, "p2", "ip2", 80);
    const OcK8sPod *pods;
    uint32_t count;
    cr_assert_eq(oc_k8s_get_pods(&cluster, &pods, &count), OC_OK);
    cr_assert_eq(count, 2);
    oc_k8s_free(&cluster);
}

Test(k8s, get_ready_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip1", 80);
    oc_k8s_mark_pod_ready(&cluster, "p1");
    OcK8sPod pods[4];
    uint32_t count;
    cr_assert_eq(oc_k8s_get_ready_pods(&cluster, pods, 4, &count), OC_OK);
    cr_assert_eq(count, 1);
    cr_assert_str_eq(pods[0].name, "p1");
    oc_k8s_free(&cluster);
}

Test(k8s, get_ready_pods_excludes_not_ready)
{
    /* The ready filter must actually filter: routing a request to a pod
     * that has not passed its readiness probe is a dropped request. */
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "cold", "ip1", 80);
    oc_k8s_add_pod(&cluster, "warm", "ip2", 80);
    oc_k8s_add_pod(&cluster, "colder", "ip3", 80);
    oc_k8s_mark_pod_ready(&cluster, "warm");

    OcK8sPod pods[8];
    uint32_t count = 0;
    cr_assert_eq(oc_k8s_get_ready_pods(&cluster, pods, 8, &count), OC_OK);
    cr_assert_eq(count, 1);
    cr_assert_str_eq(pods[0].name, "warm");
    cr_assert(pods[0].ready);
    oc_k8s_free(&cluster);
}

Test(k8s, get_ready_pods_reports_required_capacity)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip1", 80);
    oc_k8s_add_pod(&cluster, "p2", "ip2", 80);
    oc_k8s_mark_pod_ready(&cluster, "p1");
    oc_k8s_mark_pod_ready(&cluster, "p2");

    OcK8sPod one[1];
    uint32_t count = 0;
    cr_assert_eq(oc_k8s_get_ready_pods(&cluster, one, 1, &count), OC_ERR_OOM);
    cr_assert_eq(count, 2, "must report how many entries are needed");
    oc_k8s_free(&cluster);
}

Test(k8s, n_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_eq(oc_k8s_n_pods(&cluster), 0);
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    cr_assert_eq(oc_k8s_n_pods(&cluster), 1);
    oc_k8s_free(&cluster);
}

Test(k8s, n_ready)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    oc_k8s_add_pod(&cluster, "p2", "ip", 80);
    oc_k8s_mark_pod_ready(&cluster, "p1");
    cr_assert_eq(oc_k8s_n_ready(&cluster), 1);
    oc_k8s_free(&cluster);
}

Test(k8s, is_available)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert(!oc_k8s_is_available(&cluster));
    oc_k8s_free(&cluster);
}

Test(k8s, scale)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    /* Not available, should return error. */
    cr_assert_neq(oc_k8s_scale(&cluster, 3), OC_OK);
    /* Available but no API endpoint configured: scaling cannot succeed. */
    unsetenv("OC_K8S_API_URL");
    unsetenv("KUBERNETES_SERVICE_HOST");
    cluster.available = true;
    cr_assert_eq(oc_k8s_scale(&cluster, 3), OC_ERR_NETWORK);
    oc_k8s_free(&cluster);
}

/* Fake API server: accepts one request, records it, replies 200. */
typedef struct {
    int      listen_fd;
    char     request[8192];
    size_t   request_len;
} FakeApi;

static void *fake_api_main(void *arg)
{
    FakeApi *api = (FakeApi *)arg;
    int cfd = accept(api->listen_fd, NULL, NULL);
    if (cfd < 0) return NULL;
    ssize_t n = read(cfd, api->request, sizeof(api->request) - 1);
    if (n > 0) {
        api->request_len = (size_t)n;
        api->request[n] = '\0';
    }
    const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
    ssize_t w = write(cfd, resp, strlen(resp));
    (void)w;
    close(cfd);
    return NULL;
}

static char *dup_env(const char *name)
{
    const char *v = getenv(name);
    return v ? strdup(v) : NULL;
}

static void restore_env(const char *name, char *saved)
{
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

Test(k8s, scale_issues_merge_patch_to_api)
{
    /* Bind an ephemeral port and point the client at it. */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    cr_assert_geq(lfd, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    cr_assert_eq(bind(lfd, (struct sockaddr *)&addr, sizeof(addr)), 0);
    cr_assert_eq(listen(lfd, 1), 0);
    socklen_t alen = sizeof(addr);
    cr_assert_eq(getsockname(lfd, (struct sockaddr *)&addr, &alen), 0);

    char url[64];
    snprintf(url, sizeof(url), "http://127.0.0.1:%u", ntohs(addr.sin_port));
    char *saved_url = dup_env("OC_K8S_API_URL");
    setenv("OC_K8S_API_URL", url, 1);

    FakeApi api;
    memset(&api, 0, sizeof(api));
    api.listen_fd = lfd;
    pthread_t server;
    cr_assert_eq(pthread_create(&server, NULL, fake_api_main, &api), 0);

    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "infer", "oxidize");
    cluster.available = true;
    cr_assert_eq(oc_k8s_scale(&cluster, 7), OC_OK);
    oc_k8s_free(&cluster);

    cr_assert_eq(pthread_join(server, NULL), 0);
    close(lfd);
    restore_env("OC_K8S_API_URL", saved_url);

    cr_assert_gt(api.request_len, 0);
    cr_assert_not_null(strstr(api.request,
        "PATCH /apis/apps/v1/namespaces/infer/deployments/oxidize/scale"));
    cr_assert_not_null(strstr(api.request,
        "Content-Type: application/merge-patch+json"));
    cr_assert_not_null(strstr(api.request, "{\"spec\":{\"replicas\":7}}"));
    cr_assert_not_null(strstr(api.request, "Host: 127.0.0.1:"));
}

Test(k8s, scale_rejects_header_injection_in_api_url)
{
    char *saved_host = dup_env("KUBERNETES_SERVICE_HOST");
    char *saved_url = dup_env("OC_K8S_API_URL");
    unsetenv("KUBERNETES_SERVICE_HOST");
    setenv("OC_K8S_API_URL", "http://127.0.0.1\r\nX-Injected: 1:80", 1);
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "infer", "oxidize");
    cluster.available = true;
    OcError scaled = oc_k8s_scale(&cluster, 1);
    oc_k8s_free(&cluster);
    restore_env("KUBERNETES_SERVICE_HOST", saved_host);
    restore_env("OC_K8S_API_URL", saved_url);
    cr_assert_eq(scaled, OC_ERR_NETWORK);
}

Test(k8s, scale_accepts_bracketed_ipv6_url)
{
    int lfd = socket(AF_INET6, SOCK_STREAM, 0);
    if (lfd < 0) cr_skip("IPv6 sockets unavailable");
    int v6only = 1;
    (void)setsockopt(lfd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));
    struct sockaddr_in6 addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_loopback;
    addr.sin6_port = 0;
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(lfd);
        cr_skip("IPv6 loopback bind failed");
    }
    cr_assert_eq(listen(lfd, 1), 0);
    socklen_t alen = sizeof(addr);
    cr_assert_eq(getsockname(lfd, (struct sockaddr *)&addr, &alen), 0);

    char url[80];
    snprintf(url, sizeof(url), "http://[::1]:%u", ntohs(addr.sin6_port));
    char *saved_url = dup_env("OC_K8S_API_URL");
    setenv("OC_K8S_API_URL", url, 1);

    FakeApi api;
    memset(&api, 0, sizeof(api));
    api.listen_fd = lfd;
    pthread_t server;
    cr_assert_eq(pthread_create(&server, NULL, fake_api_main, &api), 0);

    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "infer", "oxidize");
    cluster.available = true;
    OcError scaled = oc_k8s_scale(&cluster, 3);
    oc_k8s_free(&cluster);

    cr_assert_eq(pthread_join(server, NULL), 0);
    close(lfd);
    restore_env("OC_K8S_API_URL", saved_url);

    cr_assert_eq(scaled, OC_OK);
    cr_assert_gt(api.request_len, 0);
    cr_assert_not_null(strstr(api.request, "Host: [::1]:"));
}

Test(k8s, mark_pod_ready)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    oc_k8s_add_pod(&cluster, "p1", "ip", 80);
    cr_assert_eq(oc_k8s_mark_pod_ready(&cluster, "p1"), OC_OK);
    cr_assert(cluster.pods[0].ready);
    oc_k8s_free(&cluster);
}

Test(k8s, mark_pod_not_found)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    cr_assert_neq(oc_k8s_mark_pod_ready(&cluster, "nonexistent"), OC_OK);
    oc_k8s_free(&cluster);
}

Test(k8s, free_null)
{
    oc_k8s_free(NULL);
}

Test(k8s, add_multiple_pods)
{
    OcK8sCluster cluster;
    oc_k8s_init(&cluster, "default", "svc");
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "pod-%d", i);
        oc_k8s_add_pod(&cluster, name, "10.0.0.1", 8080);
    }
    cr_assert_eq(cluster.n_pods, 10);
    oc_k8s_free(&cluster);
}
