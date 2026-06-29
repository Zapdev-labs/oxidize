"""Bully election mirroring oxidize-golang/core/mesh/election_priority.go.

Implements a deterministic ``Priority`` tuple ordering and a Bully election
state machine that selects a master peer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .topology_extended import CapTopologyGraph


@dataclass(frozen=True)
class Priority:
    """Deterministic priority tuple used to rank nodes (mirrors Priority).

    Ordering: higher ``clock`` wins; if equal, higher ``seniority``; if equal,
    higher ``commands_seen``; if equal, lexicographically larger ``peer_id``.
    """

    clock: int = 0
    seniority: int = 0
    commands_seen: int = 0
    peer_id: str = ""

    def _key(self) -> tuple[int, int, int, str]:
        return (self.clock, self.seniority, self.commands_seen, self.peer_id)

    def less(self, other: "Priority") -> bool:
        """Report whether ``self`` ranks below ``other``."""
        return self._key() < other._key()


class BullyElectionEngine:
    """Bully election state machine with deterministic priority winner.

    Additive alongside the simpler existing ``BullyElection`` type.
    """

    def __init__(self, local_peer_id: str, local_seniority: int = 0) -> None:
        self.local_peer_id = local_peer_id
        self.local_seniority = local_seniority
        self.local_commands = 0
        self.clock = 0
        self._electing = False
        self._elected = False
        self._master = ""
        self._declares: dict[str, Priority] = {}
        self.completed = 0

    def start_election(self) -> None:
        """Begin a new round with an incremented clock."""
        self.clock += 1
        self._declares = {}
        self._electing = True
        self._elected = False

    def record_declare(
        self, clock: int, seniority: int, commands_seen: int, peer_id: str
    ) -> None:
        """Record a remote candidacy if it belongs to the current round."""
        if not self._electing or clock != self.clock:
            return
        self._declares[peer_id] = Priority(clock, seniority, commands_seen, peer_id)

    def finalize_election(self) -> tuple[str, bool]:
        """Compute the deterministic winner and transition to elected."""
        if not self._electing:
            return "", False
        best = Priority(
            self.clock, self.local_seniority, self.local_commands, self.local_peer_id
        )
        for p in self._declares.values():
            if best.less(p):
                best = p
        self._electing = False
        self._elected = True
        self._master = best.peer_id
        self.completed += 1
        return best.peer_id, True

    def is_master(self) -> bool:
        return self._elected and self._master == self.local_peer_id

    def current_master(self) -> tuple[str, bool]:
        if self._elected:
            return self._master, True
        return "", False

    def accept_result(self, clock: int, master: str) -> None:
        """Adopt a result broadcast if its clock is current or newer."""
        if clock >= self.clock:
            self.clock = clock
            self._electing = False
            self._elected = True
            self._master = master

    def inc_local_commands(self) -> None:
        self.local_commands += 1


def run_election_round(
    engine: BullyElectionEngine, graph: CapTopologyGraph | None
) -> tuple[str, bool]:
    """Run a full deterministic election using a capability graph.

    Starts the round, injects virtual declares for every peer in the graph and
    finalizes. Mirrors run_election_round.
    """
    engine.start_election()
    if graph is not None:
        for peer_id, node in graph._snapshot_nodes().items():  # noqa: SLF001
            if peer_id == engine.local_peer_id:
                continue
            engine.record_declare(
                engine.clock, node.seniority, node.commands_seen, peer_id
            )
    return engine.finalize_election()
