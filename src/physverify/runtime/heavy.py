"""One heavy job at a time: an advisory ``flock`` shared with flock(1).

``HeavyLock`` and ``flock <lock> <cmd>`` exclude each other because both use flock(2) on the
same file. Never wrap ``physverify`` itself with flock(1); set PHYSVERIFY_LOCK_HELD=1 instead.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from physverify.core.errors import LockBusyError
from physverify.core.output import Echo

LOCK_HELD_ENV = "PHYSVERIFY_LOCK_HELD"
PROC_LOCKS = Path("/proc/locks")


def holder_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".holder")


class NullLock:
    """Stand-in when locking is disabled (empty path, --no-lock, or held by a parent)."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __enter__(self) -> NullLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class HeavyLock:
    """Exclusive lock retried every ``poll_s`` until ``timeout_s`` (None waits forever)."""

    def __init__(
        self,
        path: Path,
        timeout_s: float | None,
        argv: Sequence[str],
        notify: Echo,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 1.0,
    ) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self.argv = tuple(argv)
        self._notify = notify
        self._sleep = sleep
        self._clock = clock
        self._poll_s = poll_s
        self._fd: int | None = None

    def __enter__(self) -> HeavyLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._wait(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        try:
            self._write_holder()
        except BaseException:
            self.release()
            raise

    def _wait(self, fd: int) -> None:
        deadline = None if self.timeout_s is None else self._clock() + self.timeout_s
        announced = False
        while not _try_lock(fd):
            holder = describe_holder(self.path)
            if deadline is not None and self._clock() >= deadline:
                raise LockBusyError(f"heavy lock {self.path} is busy; held by {holder}")
            if not announced:
                self._notify(f"waiting for heavy lock {self.path}; held by {holder}")
                announced = True
            self._sleep(self._poll_s)

    def _write_holder(self) -> None:
        record = {
            "pid": os.getpid(),
            "argv": list(self.argv),
            "since": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        holder_path(self.path).write_text(json.dumps(record), encoding="utf-8")

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            holder_path(self.path).unlink(missing_ok=True)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def describe_holder(lock: Path) -> str:
    """Who holds ``lock``: the holder file written by physverify, else /proc/locks."""
    pids = lock_holder_pids(lock)
    recorded = _read_holder(lock)
    if recorded is not None and (not pids or recorded.get("pid") in pids):
        command = recorded.get("argv")
        argv = " ".join(map(str, command)) if isinstance(command, list) else ""
        return f"pid {recorded.get('pid')} ({argv}) since {recorded.get('since')}"
    if pids:
        return ", ".join(f"pid {pid} ({_cmdline(pid)})" for pid in pids)
    return "an unknown process"


def _read_holder(lock: Path) -> dict[str, object] | None:
    try:
        data = json.loads(holder_path(lock).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def lock_holder_pids(lock: Path, proc_locks: Path = PROC_LOCKS) -> list[int]:
    """PIDs with a granted flock on the inode of ``lock`` according to /proc/locks."""
    try:
        stat = lock.stat()
        lines = proc_locks.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    target = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    pids: list[int] = []
    for line in lines:
        fields = line.split()
        if "->" in fields or len(fields) < 6 or fields[1] != "FLOCK":
            continue
        if fields[5] == target and fields[4].isdigit():
            pids.append(int(fields[4]))
    return sorted(set(pids))


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "?"
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
