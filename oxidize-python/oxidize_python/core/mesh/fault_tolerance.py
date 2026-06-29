"""Fault tolerance mirroring oxidize-golang/core/mesh/fault_tolerance.go.

Implements timeout-enforced collective evaluation with a ``TimedResult`` enum
and ``RunnerStatusUpdated`` notification events.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")

# Default collective timeout in seconds (mirrors DEFAULT_COLLECTIVE_TIMEOUT).
DEFAULT_COLLECTIVE_TIMEOUT = 60.0


class RunnerStatusKind(str, Enum):
    """Runner status discriminant (mirrors RunnerStatus)."""

    HEALTHY = "healthy"
    FAILED = "runner_failed"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


@dataclass
class RunnerStatus:
    kind: RunnerStatusKind = RunnerStatusKind.HEALTHY
    reason: str = ""


@dataclass
class RunnerStatusUpdated:
    peer_id: str = ""
    status: RunnerStatus = field(default_factory=RunnerStatus)
    clock: int = 0


@dataclass
class ShutdownTask:
    instance_id: str = ""
    reason: str = ""
    clock: int = 0


class TimedResultKind(Enum):
    """TimedResult discriminant (mirrors TimedResultKind)."""

    OK = 0
    TIMED_OUT = 1
    ERR = 2


@dataclass
class TimedResult(Generic[T]):
    """Result of a timeout-enforced evaluation (mirrors TimedResult<T>).

    ``value`` is valid only when ``kind == TimedResultKind.OK``.
    """

    kind: TimedResultKind
    value: T | None = None
    err: str = ""

    def is_ok(self) -> bool:
        return self.kind == TimedResultKind.OK


def eval_with_timeout(
    deadline: float, fut: Callable[[], T]
) -> TimedResult[T]:
    """Evaluate ``fut`` with a hard deadline (in seconds).

    If ``fut`` does not complete within ``deadline``, ``TIMED_OUT`` is returned.
    Note: the worker thread is left running (daemon) on timeout since Python
    cannot forcibly cancel a thread; this mirrors the Go context-cancel
    semantics at the result level. Mirrors eval_with_timeout.
    """
    result: dict[str, object] = {}

    def _run() -> None:
        try:
            result["val"] = fut()
        except Exception as exc:  # noqa: BLE001
            result["err"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=deadline)
    if worker.is_alive():
        return TimedResult(kind=TimedResultKind.TIMED_OUT)
    if "err" in result:
        return TimedResult(kind=TimedResultKind.ERR, err=str(result["err"]))
    return TimedResult(kind=TimedResultKind.OK, value=result.get("val"))  # type: ignore[arg-type]


def eval_with_timeout_and_notify(
    deadline: float,
    peer_id: str,
    clock: int,
    on_status: Callable[[RunnerStatusUpdated], None] | None,
    fut: Callable[[], T],
) -> TimedResult[T]:
    """Wrap ``eval_with_timeout`` and emit a ``RunnerStatusUpdated`` on timeout.

    Mirrors eval_with_timeout_and_notify.
    """
    res = eval_with_timeout(deadline, fut)
    if res.kind == TimedResultKind.TIMED_OUT and on_status is not None:
        on_status(
            RunnerStatusUpdated(
                peer_id=peer_id,
                status=RunnerStatus(
                    kind=RunnerStatusKind.FAILED,
                    reason=f"collective timed out after {int(deadline)}s",
                ),
                clock=clock,
            )
        )
    return res
