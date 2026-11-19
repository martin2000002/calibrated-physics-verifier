"""Process-level map whose result does not depend on the worker count.

``ordered_map`` hands the items to forked worker processes one index at a time and returns the
results in input order, so a caller that folds them in that order gets the serial result. With
one worker, or at most one item, it runs in the calling process. Workers are forked: the function
and the items are inherited, never pickled. Each worker has its own task pipe and answer pipe (one
writer per pipe, so large results never interleave) and pickles its result itself, so an
unpicklable result fails loudly in the worker. A worker holds at most one task: the next index is
sent only after its answer arrives, so no write to a task pipe can fill its buffer and block.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import pickle
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path

from physverify.core.log import get_logger

_log = get_logger("runtime.parallel")

STOP = -1
"""Index that tells a worker to exit."""


class WorkerError(RuntimeError):
    """A worker raised, died or returned something it could not pickle."""


@dataclass(frozen=True)
class _Worker:
    process: BaseProcess
    tasks: Connection
    answers: Connection


def ordered_map[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """``[fn(item) for item in items]``, computed by up to ``workers`` forked processes."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers == 1 or len(items) <= 1:
        return [fn(item) for item in items]
    return _fan_out(fn, items, min(workers, len(items)))


def _fan_out[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    context = multiprocessing.get_context("fork")
    _log.info(
        "forking %d workers from a process with %d threads (%d Python)",
        workers,
        _native_threads(),
        threading.active_count(),
    )
    pool: list[_Worker] = []
    try:
        for _ in range(workers):
            take, tasks = context.Pipe(duplex=False)
            answers, send = context.Pipe(duplex=False)
            inherited = tuple(end for worker in pool for end in (worker.tasks, worker.answers))
            process = context.Process(
                target=_serve, args=(fn, items, take, send, inherited), daemon=True
            )
            process.start()
            take.close()
            send.close()
            pool.append(_Worker(process, tasks, answers))
        return _collect(pool, len(items))
    finally:
        for worker in pool:
            if worker.process.is_alive():
                worker.process.terminate()
            worker.process.join()
            worker.tasks.close()
            worker.answers.close()


def _native_threads() -> int:
    """Threads of this process as the OS sees them (BLAS pools, injected agents included)."""
    tasks = Path("/proc/self/task")
    return sum(1 for _ in tasks.iterdir()) if tasks.is_dir() else threading.active_count()


def _serve[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    take: Connection,
    send: Connection,
    inherited: Sequence[Connection],
) -> None:
    for end in inherited:
        end.close()
    while (index := take.recv()) != STOP:
        try:
            payload = pickle.dumps(fn(items[index]), pickle.HIGHEST_PROTOCOL)
        except BaseException:  # noqa: BLE001 - any failure goes back to the parent, with its trace
            send.send((index, False, traceback.format_exc()))
            return
        send.send((index, True, payload))


def _dispatch(worker: _Worker, pending: Iterator[int]) -> None:
    with contextlib.suppress(BrokenPipeError):  # a gone worker is reported by its answer pipe
        worker.tasks.send(next(pending, STOP))


def _collect[R](pool: Sequence[_Worker], count: int) -> list[R]:
    results: dict[int, R] = {}
    pending = iter(range(count))
    for worker in pool:
        _dispatch(worker, pending)
    live = list(pool)
    while len(results) < count:
        if not live:
            raise WorkerError(f"every worker exited with {count - len(results)} items pending")
        ready = wait([worker.answers for worker in live])
        for worker in [worker for worker in live if worker.answers in ready]:
            try:
                index, ok, payload = worker.answers.recv()
            except EOFError:
                live.remove(worker)
                worker.process.join()
                if worker.process.exitcode != 0:
                    raise WorkerError(
                        f"a worker died with exit code {worker.process.exitcode}"
                    ) from None
                continue
            if not ok:
                raise WorkerError(f"item {index} failed in a worker:\n{payload}")
            results[index] = pickle.loads(payload)
            _dispatch(worker, pending)
    return [results[index] for index in range(count)]
