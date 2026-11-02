"""ordered_map: input order, worker-count independence and loud failures."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from physverify.diagnosis.measurement_cases import MetricOutcome, outcome_of
from physverify.runtime.parallel import WorkerError, ordered_map


def square_with_pid(item: int) -> tuple[int, int]:
    return item * item, os.getpid()


def fail_on_three(item: int) -> int:
    if item == 3:
        raise ValueError("three")
    return item


def exit_on_two(item: int) -> int:
    if item == 2:
        os._exit(7)
    return item


def unpicklable(item: int) -> object:
    return lambda: item


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_results_come_back_in_input_order(workers: int) -> None:
    items = list(range(23))
    assert [v for v, _ in ordered_map(square_with_pid, items, workers)] == [i * i for i in items]


def test_several_workers_run_in_other_processes() -> None:
    pids = {pid for _, pid in ordered_map(square_with_pid, list(range(12)), 3)}
    assert os.getpid() not in pids


def test_one_worker_runs_in_the_calling_process() -> None:
    assert {pid for _, pid in ordered_map(square_with_pid, [1, 2, 3], 1)} == {os.getpid()}


def test_one_and_three_workers_give_byte_identical_results() -> None:
    def noisy(seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=50).cumsum()

    serial = [a.tobytes() for a in ordered_map(noisy, list(range(10)), 1)]
    assert [a.tobytes() for a in ordered_map(noisy, list(range(10)), 3)] == serial


MANY_ITEMS = """
from physverify.runtime.parallel import ordered_map
assert ordered_map(abs, list(range(-10_000, 0)), 2) == list(range(10_000, 0, -1))
"""


def test_many_items_never_block_on_a_full_pipe() -> None:
    """Thousands of indices overflow any pipe buffer if they are all queued before dispatch."""
    done = subprocess.run([sys.executable, "-c", MANY_ITEMS], timeout=60, check=False)
    assert done.returncode == 0


def test_no_items_gives_no_results() -> None:
    assert ordered_map(square_with_pid, [], 4) == []


def test_fewer_than_one_worker_is_refused() -> None:
    with pytest.raises(ValueError, match="workers"):
        ordered_map(square_with_pid, [1], 0)


def test_an_exception_in_a_worker_is_raised_with_its_traceback() -> None:
    with pytest.raises(WorkerError, match="ValueError: three"):
        ordered_map(fail_on_three, list(range(6)), 3)


def test_a_worker_that_dies_is_reported() -> None:
    with pytest.raises(WorkerError, match="exit code 7"):
        ordered_map(exit_on_two, list(range(6)), 2)


def test_an_unpicklable_result_fails_in_the_worker() -> None:
    with pytest.raises(WorkerError, match="item"):
        ordered_map(unpicklable, [1, 2], 2)


def test_a_metric_outcome_survives_pickling() -> None:
    outcome = outcome_of({"x": np.array([1.0, np.nan, -2.5])}, {"hit": (3.0, 4.0)})
    back = pickle.loads(pickle.dumps(outcome))
    assert isinstance(back, MetricOutcome)
    np.testing.assert_array_equal(back.samples["x"], outcome.samples["x"])
    assert dict(back.ratios) == {"hit": (3.0, 4.0)}
