"""Process-level map whose result does not depend on the worker count.

``ordered_map`` hands the items to forked worker processes one index at a time and returns the
results in input order, so a caller that folds them in that order gets the serial result. With
one worker, or at most one item, it runs in the calling process. Workers are forked: the function
and the items are inherited, never pickled. Each worker answers on its own pipe (one writer per
pipe, so large results never interleave) and pickles its result itself, so an unpicklable result
fails loudly in the worker.
"""

from __future__ import annotations

import multiprocessing
import pickle
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from multiprocessing.queues import SimpleQueue

STOP = -1
"""Index that tells a worker to exit."""


class WorkerError(RuntimeError):
    """A worker raised, died or returned something it could not pickle."""


@dataclass(frozen=True)
class _Worker:
    process: BaseProcess
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
    tasks: SimpleQueue[int] = context.SimpleQueue()
    for index in range(len(items)):
        tasks.put(index)
    for _ in range(workers):
        tasks.put(STOP)
    pool: list[_Worker] = []
    try:
        for _ in range(workers):
            answers, send = context.Pipe(duplex=False)
            process = context.Process(target=_serve, args=(fn, items, tasks, send), daemon=True)
            process.start()
            send.close()
            pool.append(_Worker(process, answers))
        return _collect(pool, len(items))
    finally:
        for worker in pool:
            if worker.process.is_alive():
                worker.process.terminate()
            worker.process.join()
            worker.answers.close()


def _serve[T, R](
    fn: Callable[[T], R], items: Sequence[T], tasks: SimpleQueue[int], send: Connection
) -> None:
    while (index := tasks.get()) != STOP:
        try:
            payload = pickle.dumps(fn(items[index]), pickle.HIGHEST_PROTOCOL)
        except BaseException:  # noqa: BLE001 - any failure goes back to the parent, with its trace
            send.send((index, False, traceback.format_exc()))
            return
        send.send((index, True, payload))


def _collect[R](pool: Sequence[_Worker], count: int) -> list[R]:
    results: dict[int, R] = {}
    open_pipes = {worker.answers: worker.process for worker in pool}
    while len(results) < count:
        if not open_pipes:
            raise WorkerError(f"every worker exited with {count - len(results)} items pending")
        for pipe in wait(list(open_pipes)):
            assert isinstance(pipe, Connection)
            try:
                index, ok, payload = pipe.recv()
            except EOFError:
                process = open_pipes.pop(pipe)
                process.join()
                if process.exitcode != 0:
                    raise WorkerError(f"a worker died with exit code {process.exitcode}") from None
                continue
            if not ok:
                raise WorkerError(f"item {index} failed in a worker:\n{payload}")
            results[index] = pickle.loads(payload)
    return [results[index] for index in range(count)]
