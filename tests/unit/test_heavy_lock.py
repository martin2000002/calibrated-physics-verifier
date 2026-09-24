from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from cli_runner import CliRunner

from physverify.core.errors import LockBusyError
from physverify.runtime.heavy import HeavyLock, describe_holder, holder_path, lock_holder_pids


@pytest.fixture
def flock_holder(lock_path: Path) -> Iterator[subprocess.Popen[str]]:
    """flock(1) holding the heavy lock, the way `flock <lock> uv sync` would."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["flock", str(lock_path), "-c", "echo ready; sleep 30"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None and process.stdout.readline().strip() == "ready"
    yield process
    process.kill()
    process.wait()


def test_cli_exits_75_and_names_the_holder(
    cli: CliRunner, flock_holder: subprocess.Popen[str]
) -> None:
    result = cli("--lock-timeout", "0", "doctor", "--lock-probe", "1")
    assert result.code == 75
    assert f"pid {flock_holder.pid}" in result.err and "flock" in result.err


def test_cli_takes_a_free_lock(cli: CliRunner, lock_path: Path) -> None:
    result = cli("--lock-timeout", "0", "doctor", "--lock-probe", "1")
    assert result.code == 0, result.err
    assert "lock-probe: released" in result.out
    assert not holder_path(lock_path).exists()


def test_lock_held_by_the_caller_is_skipped(
    cli: CliRunner, flock_holder: subprocess.Popen[str]
) -> None:
    result = cli(
        "--lock-timeout", "0", "doctor", "--lock-probe", "0.1", env={"PHYSVERIFY_LOCK_HELD": "1"}
    )
    assert result.code == 0, result.err
    assert "disabled (PHYSVERIFY_LOCK_HELD=1" in result.out


def test_no_lock_option(cli: CliRunner, flock_holder: subprocess.Popen[str]) -> None:
    result = cli("--no-lock", "--lock-timeout", "0", "doctor", "--lock-probe", "0.1")
    assert result.code == 0 and "disabled (--no-lock)" in result.out


def test_doctor_reports_the_holder(cli: CliRunner, flock_holder: subprocess.Popen[str]) -> None:
    result = cli("doctor")
    assert result.code == 0 and f"held by pid {flock_holder.pid}" in result.out


def test_waiting_announces_the_holder_then_times_out(
    lock_path: Path, flock_holder: subprocess.Popen[str]
) -> None:
    messages: list[str] = []
    clock = iter([0.0, 0.0, 1.0, 2.5])
    lock = HeavyLock(
        lock_path, 2.0, ["x"], messages.append, sleep=lambda _: None, clock=lambda: next(clock)
    )
    with pytest.raises(LockBusyError):
        lock.acquire()
    assert len(messages) == 1 and "waiting for heavy lock" in messages[0]
    assert lock_holder_pids(lock_path) == [flock_holder.pid]


def test_lock_creates_parents_writes_holder_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "dir" / "heavy.lock"
    with HeavyLock(path, 0.0, ["physverify", "doctor"], print):
        assert holder_path(path).exists()
        assert "physverify doctor" in describe_holder(path)
        assert lock_holder_pids(path)
    assert not holder_path(path).exists()
    assert lock_holder_pids(path) == []
