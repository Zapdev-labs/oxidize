/*
 * k8s.h — Kubernetes integration stubs.
 *
 * Provides a minimal API for Kubernetes-based deployment coordination.
 * This is a stub implementation that doesn't actually talk to Kubernetes.
 * Port from oxidize-core/src/mesh/k8s.rs.
 */
#ifndef OXIDIZE_K8S_H
#define OXIDIZE_K8S_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_K8S_MAX_PODS 128
#define OC_K8S_MAX_NAME 256

typedef struct {
    char name[OC_K8S_MAX_NAME];
    char ip[64];
    uint16_t port;
    bool ready;
    uint32_t restarts;
    uint64_t age_sec;
} OcK8sPod;

typedef struct {
    char namespace[128];
    char service_name[OC_K8S_MAX_NAME];
    OcK8sPod pods[OC_K8S_MAX_PODS];
    uint32_t n_pods;
    bool available;
} OcK8sCluster;

OcError oc_k8s_init(OcK8sCluster *cluster, const char *namespace,
                   const char *service_name);
OcError oc_k8s_detect(OcK8sCluster *cluster);
OcError oc_k8s_add_pod(OcK8sCluster *cluster, const char *name,
                      const char *ip, uint16_t port);
OcError oc_k8s_get_pods(const OcK8sCluster *cluster, const OcK8sPod **out, uint32_t *count);
OcError oc_k8s_get_ready_pods(const OcK8sCluster *cluster, const OcK8sPod **out, uint32_t *count);
uint32_t oc_k8s_n_pods(const OcK8sCluster *cluster);
uint32_t oc_k8s_n_ready(const OcK8sCluster *cluster);
bool oc_k8s_is_available(const OcK8sCluster *cluster);
OcError oc_k8s_scale(const OcK8sCluster *cluster, uint32_t target_replicas);
OcError oc_k8s_mark_pod_ready(OcK8sCluster *cluster, const char *name);
void oc_k8s_free(OcK8sCluster *cluster);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_K8S_H */
