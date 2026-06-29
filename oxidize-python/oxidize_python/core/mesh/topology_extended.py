"""Capability-aware topology mirroring oxidize-golang/core/mesh/topology_extended.go.

Extends the simple ``TopologyGraph`` with per-peer capability tracking, stale
node eviction, capability aggregation and shardable-node filtering.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class NodeCapabilities:
    """Capabilities advertised during discovery (mirrors NodeCapabilities)."""

    device_type: str = ""
    memory_bytes: int = 0
    cpu_threads: int = 0
    can_shard: bool = False
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class AggregateCapabilities:
    """Mesh-wide capability summary (mirrors AggregateCapabilities)."""

    node_count: int = 0
    total_memory_bytes: int = 0
    total_cpu_threads: int = 0
    can_shard_nodes: int = 0
    device_types: list[str] = field(default_factory=list)


@dataclass
class CapTopologyNode:
    """Capability-tracked node with liveness metadata (mirrors TopologyNode)."""

    peer_id: str
    capabilities: NodeCapabilities
    commands_seen: int = 0
    seniority: int = 0
    last_seen: float = 0.0
    joined_at: float = 0.0

    def is_stale(self, timeout: float) -> bool:
        if self.last_seen <= 0.0:
            return True
        return (time.monotonic() - self.last_seen) > timeout


class CapTopologyGraph:
    """Capability-aware topology graph (mirrors CapTopologyGraph).

    Additive to the existing ``TopologyGraph``; tracks per-peer
    ``NodeCapabilities`` for aggregation and stale eviction.
    """

    def __init__(self, local_peer_id: str = "") -> None:
        self._mu = threading.Lock()
        self._nodes: dict[str, CapTopologyNode] = {}
        self.local_peer_id = local_peer_id

    def add_or_update_node(self, peer_id: str, caps: NodeCapabilities) -> None:
        with self._mu:
            now = time.monotonic()
            existing = self._nodes.get(peer_id)
            if existing is not None:
                existing.capabilities = caps
                existing.last_seen = now
                return
            self._nodes[peer_id] = CapTopologyNode(
                peer_id=peer_id,
                capabilities=caps,
                last_seen=now,
                joined_at=now,
            )

    def remove_node(self, peer_id: str) -> None:
        with self._mu:
            self._nodes.pop(peer_id, None)

    def heartbeat(self, peer_id: str) -> None:
        with self._mu:
            n = self._nodes.get(peer_id)
            if n is not None:
                n.last_seen = time.monotonic()

    def evict_stale(self, timeout: float) -> list[str]:
        """Remove nodes not seen within ``timeout`` and return their sorted IDs."""
        with self._mu:
            stale = [pid for pid, n in self._nodes.items() if n.is_stale(timeout)]
            for pid in stale:
                del self._nodes[pid]
        stale.sort()
        return stale

    def peer_count(self) -> int:
        with self._mu:
            return len(self._nodes)

    def peer_ids(self) -> list[str]:
        """Return all known peer IDs excluding the local peer, sorted."""
        with self._mu:
            out = [
                pid
                for pid in self._nodes
                if not (self.local_peer_id and pid == self.local_peer_id)
            ]
        out.sort()
        return out

    def capabilities_of(self, peer_id: str) -> tuple[NodeCapabilities | None, bool]:
        with self._mu:
            n = self._nodes.get(peer_id)
        if n is None:
            return None, False
        return n.capabilities, True

    def aggregate_capabilities(self) -> AggregateCapabilities:
        """Summarize capabilities across all peers (mirrors aggregate_capabilities)."""
        with self._mu:
            agg = AggregateCapabilities(node_count=len(self._nodes))
            seen: set[str] = set()
            for n in self._nodes.values():
                agg.total_memory_bytes += n.capabilities.memory_bytes
                agg.total_cpu_threads += n.capabilities.cpu_threads
                if n.capabilities.can_shard:
                    agg.can_shard_nodes += 1
                if n.capabilities.device_type not in seen:
                    seen.add(n.capabilities.device_type)
                    agg.device_types.append(n.capabilities.device_type)
        agg.device_types.sort()
        return agg

    def shardable_nodes(self) -> list[str]:
        """Return IDs of peers that can act as shard workers, sorted."""
        with self._mu:
            out = [pid for pid, n in self._nodes.items() if n.capabilities.can_shard]
        out.sort()
        return out

    def set_seniority(self, peer_id: str, seniority: int) -> None:
        with self._mu:
            n = self._nodes.get(peer_id)
            if n is not None:
                n.seniority = seniority

    # Internal accessor used by run_election_round (holds no lock; callers must).
    def _snapshot_nodes(self) -> dict[str, CapTopologyNode]:
        with self._mu:
            return dict(self._nodes)
