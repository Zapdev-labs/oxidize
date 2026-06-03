"""Row-parallel helpers mirroring oxidize-golang/core/tensor/gemv.go."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from oxidize_python.core.tensor.constants import PARALLEL_GEMV_MIN_OPS


def parallel_workers(n: int) -> int:
    if n < 1:
        return 1
    if n < 256:
        return 1
    return min(8, os.cpu_count() or 8)


def parallelize_rows(rows: int, fn: Callable[[int, int], None]) -> None:
    if rows <= PARALLEL_GEMV_MIN_OPS // 4 or rows < 32:
        fn(0, rows)
        return
    workers = parallel_workers(rows)
    if workers <= 1:
        fn(0, rows)
        return
    chunk = (rows + workers - 1) // workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for w in range(workers):
            start = w * chunk
            if start >= rows:
                break
            end = min(start + chunk, rows)
            futures.append(pool.submit(fn, start, end))
        for f in futures:
            f.result()


def parallelize_range(n: int, fn: Callable[[int, int], None]) -> None:
    parallelize_rows(n, fn)
