#ifndef OXIDIZE_DISCOVERY_H
#define OXIDIZE_DISCOVERY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DISCOVERY_MAX_PEERS 128
#define OC_DISCOVERY_MAX_ADDR 64

typedef struct {
    uint64_t id;
    char addr[OC_DISCOVERY_MAX_ADDR];
    uint16_t port;
    uint32_t capabilities;
    uint64_t last_seen_ms;
    bool alive;
} OcPeerInfo;

typedef struct {
    OcPeerInfo peers[OC_DISCOVERY_MAX_PEERS];
    uint32_t n_peers;
    uint64_t self_id;
    char multicast_addr[OC_DISCOVERY_MAX_ADDR];
    uint16_t port;
    uint32_t discovery_interval_ms;
    uint32_t timeout_ms;
    uint64_t current_ms;
} OcDiscoveryState;

OcError oc_discovery_init(OcDiscoveryState *state, uint64_t self_id,
                         const char *multicast_addr, uint16_t port);
OcError oc_discovery_add_seed(OcDiscoveryState *state, uint64_t id,
                             const char *addr, uint16_t port);
OcError oc_discovery_announce(OcDiscoveryState *state);
OcError oc_discovery_receive_peer(OcDiscoveryState *state, uint64_t id,
                                  const char *addr, uint16_t port,
                                  uint32_t capabilities);
OcError oc_discovery_tick(OcDiscoveryState *state, uint64_t current_ms);
OcError oc_discovery_get_peers(const OcDiscoveryState *state,
                              const OcPeerInfo **out, uint32_t *count);
OcError oc_discovery_get_alive_peers(const OcDiscoveryState *state,
                                    const OcPeerInfo **out, uint32_t *count);
uint32_t oc_discovery_n_peers(const OcDiscoveryState *state);
uint32_t oc_discovery_n_alive(const OcDiscoveryState *state);
bool oc_discovery_has_peer(const OcDiscoveryState *state, uint64_t id);
OcError oc_discovery_remove_peer(OcDiscoveryState *state, uint64_t id);
void oc_discovery_free(OcDiscoveryState *state);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DISCOVERY_H */
