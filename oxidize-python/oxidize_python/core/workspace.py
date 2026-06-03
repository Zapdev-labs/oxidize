"""Mirrors oxidize_core workspace_health / benchmark_input / wasm_workspace_status."""

from dataclasses import dataclass

_READY = "ready"


@dataclass(frozen=True)
class WorkspaceHealth:
    status: str


def health() -> WorkspaceHealth:
    return WorkspaceHealth(status=_READY)


def benchmark_input() -> WorkspaceHealth:
    return health()


def wasm_status() -> str:
    return _READY
