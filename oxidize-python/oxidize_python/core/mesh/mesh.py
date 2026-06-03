"""Distributed mesh mirroring oxidize-golang/core/mesh/mesh.go."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class MeshConfig:
    node_id: str = ""
    heartbeat_period: float = 0.25
    election_timeout: float = 1.0
    discovery_url: str = ""
    bind_addr: str = ""


def default_mesh_config() -> MeshConfig:
    return MeshConfig()


@dataclass
class MeshNode:
    id: str
    addr: str = ""
    role: str = ""
    last_seen: float = 0.0
    healthy: bool = True
    layer_hint: int = 0


class GossipRouter:
    def __init__(self) -> None:
        self._mu = threading.RLock()
        self._peers: dict[str, MeshNode] = {}
        self._callbacks: list = []

    def on(self, cb) -> None:
        with self._mu:
            self._callbacks.append(cb)

    def update(self, node: MeshNode) -> None:
        with self._mu:
            node.last_seen = time.time()
            self._peers[node.id] = node
            cbs = list(self._callbacks)
        for cb in cbs:
            cb(node)

    def peers(self) -> list[MeshNode]:
        with self._mu:
            out = list(self._peers.values())
        return sorted(out, key=lambda n: n.id)

    def peer_by_id(self, node_id: str) -> tuple[MeshNode, bool]:
        with self._mu:
            n = self._peers.get(node_id)
        return (n, n is not None) if n else (MeshNode(""), False)

    def remove(self, node_id: str) -> None:
        with self._mu:
            self._peers.pop(node_id, None)


class RingTransport:
    def __init__(self, peers: list[str]) -> None:
        self._mu = threading.Lock()
        self._peers = list(peers)

    def next(self, self_addr: str) -> str:
        with self._mu:
            if not self._peers:
                return ""
            for i, p in enumerate(self._peers):
                if p == self_addr:
                    return self._peers[(i + 1) % len(self._peers)]
            return self._peers[0]


class ChannelTransport:
    def __init__(self) -> None:
        import queue
        self.in_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self.out_q: queue.Queue[bytes] = queue.Queue(maxsize=64)

    def send(self, msg: bytes) -> None:
        self.out_q.put(msg)

    def recv(self) -> bytes | None:
        try:
            return self.in_q.get_nowait()
        except Exception:
            try:
                return self.out_q.get_nowait()
            except Exception:
                return None


@dataclass
class TcpTransport:
    addr: str


@dataclass
class MeshShard:
    node_id: str
    layer_start: int
    layer_end: int
    device_id: int = 0
    backend: str = ""


@dataclass
class ShardPlan:
    shards: list[MeshShard] = field(default_factory=list)
    layer_range: tuple[int, int] = (0, 0)
    replica_index: int = 0

    def validate(self) -> None:
        if not self.shards:
            raise ValueError("mesh: empty shard plan")
        if self.layer_range[0] >= self.layer_range[1]:
            raise ValueError("mesh: invalid layer range")
        for s in self.shards:
            if s.layer_start >= s.layer_end:
                raise ValueError("mesh: invalid shard range")


class DiscoveryService:
    def __init__(self, router: GossipRouter) -> None:
        self._mu = threading.Lock()
        self.router = router
        self._known: set[str] = set()

    def announce(self, node: MeshNode) -> None:
        with self._mu:
            self._known.add(node.id)
        self.router.update(node)

    def forget(self, node_id: str) -> None:
        with self._mu:
            self._known.discard(node_id)
        self.router.remove(node_id)


class BullyElection:
    def __init__(self, peers: list[str]) -> None:
        self._mu = threading.Lock()
        self._peers = peers
        self._leader = peers[0] if peers else ""

    def leader(self) -> str:
        with self._mu:
            return self._leader

    def set_leader(self, leader_id: str) -> None:
        with self._mu:
            self._leader = leader_id


class TopologyGraph:
    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._nodes: dict[str, MeshNode] = {}
        self._edges: dict[str, list[str]] = {}

    def add_node(self, node: MeshNode) -> None:
        with self._mu:
            self._nodes[node.id] = node
            self._edges.setdefault(node.id, [])

    def add_edge(self, a: str, b: str) -> None:
        with self._mu:
            self._edges.setdefault(a, []).append(b)

    def neighbors(self, node_id: str) -> list[str]:
        with self._mu:
            return sorted(self._edges.get(node_id, []))

    def nodes(self) -> list[MeshNode]:
        with self._mu:
            return list(self._nodes.values())


@dataclass
class LoadProgressReport:
    node_id: str = ""
    bytes_loaded: int = 0
    bytes_total: int = 0
    percent_done: float = 0.0
    layers_ready: int = 0
    layers_total: int = 0
    elapsed_millis: int = 0


@dataclass
class MeshValidationReport:
    total_nodes: int = 0
    healthy_nodes: int = 0
    layers_covered: int = 0
    missing_shards: list[str] = field(default_factory=list)


class MeshChatEngine:
    def __init__(self, local: MeshNode) -> None:
        self._mu = threading.Lock()
        self.router = GossipRouter()
        self.shards: dict[str, ShardPlan] = {}
        self.local_node = local

    def register_shard(self, key: str, plan: ShardPlan) -> None:
        with self._mu:
            self.shards[key] = plan

    def lookup_shard(self, key: str) -> tuple[ShardPlan | None, bool]:
        with self._mu:
            p = self.shards.get(key)
        return (p, p is not None)


def validate(plan: ShardPlan | None, healthy: list[MeshNode]) -> MeshValidationReport:
    rep = MeshValidationReport(total_nodes=len(healthy))
    healthy_ids = set()
    for n in healthy:
        if n.healthy:
            rep.healthy_nodes += 1
        healthy_ids.add(n.id)
    if plan is None:
        return rep
    for s in plan.shards:
        if s.node_id in healthy_ids:
            rep.layers_covered += s.layer_end - s.layer_start
        else:
            rep.missing_shards.append(s.node_id)
    return rep
