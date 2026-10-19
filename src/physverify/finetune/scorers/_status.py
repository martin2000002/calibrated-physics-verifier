"""Optional scorer extra: the calibration state behind its scores (not in the frozen protocol)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CalibrationStatusSource(Protocol):
    def calibration_status(self) -> Mapping[str, Any] | None: ...
