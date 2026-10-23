"""Leaves every result UNSCORED: raw magnitudes for the calibration chain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from physverify.core.models import FrozenModel
from physverify.schema.checks import CheckResult


class RawParams(FrozenModel):
    pass


class RawPolicy:
    name: ClassVar[str] = "raw"
    Params: ClassVar[type[RawParams]] = RawParams

    def __init__(self, params: RawParams | None = None) -> None:
        self._params = params or RawParams()

    def decide(self, result: CheckResult, thresholds: Mapping[str, Any]) -> CheckResult:
        return result


PLUGINS = (RawPolicy,)
