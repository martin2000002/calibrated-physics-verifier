"""Resources a pipeline run holds: the lazily entered heavy lock and the per-stage RSS probe."""

from __future__ import annotations

import resource
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Final, Protocol

KIB_PER_MIB: Final = 1024.0


class LazyLock:
    """Enters the heavy lock on the first ``ensure()``; a fully cached run never builds it."""

    def __init__(self, factory: Callable[[], AbstractContextManager[object]]) -> None:
        self._factory = factory
        self._held: AbstractContextManager[object] | None = None
        self._acquisitions = 0

    def ensure(self) -> None:
        if self._held is not None:
            return
        manager = self._factory()
        manager.__enter__()
        self._held = manager
        self._acquisitions += 1

    def release(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            held.__exit__(None, None, None)

    @property
    def acquisitions(self) -> int:
        return self._acquisitions

    @property
    def held(self) -> bool:
        return self._held is not None


class RssProbe(Protocol):
    def reset(self) -> None: ...

    def peak_mb(self) -> float: ...


class ProcRssProbe:
    """Per-stage peak RSS from ``VmHWM`` after resetting it through ``clear_refs``.

    ``exact`` turns False when the reset or the read fails; the fallback ``ru_maxrss`` is the
    process-lifetime peak, an upper bound of the stage peak."""

    def __init__(self, proc: Path = Path("/proc/self")) -> None:
        self._proc = proc
        self.exact = True

    def reset(self) -> None:
        try:
            (self._proc / "clear_refs").write_text("5", encoding="ascii")
        except OSError:
            self.exact = False

    def peak_mb(self) -> float:
        if self.exact:
            hwm_kb = self._vm_hwm_kb()
            if hwm_kb is not None:
                return hwm_kb / KIB_PER_MIB
            self.exact = False
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB

    def _vm_hwm_kb(self) -> float | None:
        try:
            lines = (self._proc / "status").read_text(encoding="ascii").splitlines()
        except OSError:
            return None
        for line in lines:
            if line.startswith("VmHWM:"):
                return float(line.split()[1])
        return None
