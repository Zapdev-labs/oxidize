/*
 * k8s.c — Kubernetes integration stub implementation.
 */
#include "oxidize/k8s.h"

#include <stdlib.h>
#include <string.h>

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

OcError oc_k8s_get_ready_pods(const OcK8sCluster *cluster, const OcK8sPod **out, uint32_t *count)
{
    if (!cluster || !out || !count) return OC_ERR_INVALID_ARG;
    /* Return all pods since we can't filter a const array in-place.
     * In a real implementation we'd copy to a caller-provided buffer. */
    *out = cluster->pods;
    *count = cluster->n_pods;
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

OcError oc_k8s_scale(const OcK8sCluster *cluster, uint32_t target_replicas)
{
    if (!cluster) return OC_ERR_INVALID_ARG;
    /* Stub: in a real implementation, this would call the Kubernetes API. */
    (void)target_replicas;
    return cluster->available ? OC_OK : OC_ERR_MODEL;
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
