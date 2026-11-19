"""Leaves every result UNSCORED: raw magnitudes for the calibration chain."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.core.models import FrozenModel
from physverify.schema.checks import CheckResult
from physverify.verification.thresholding import ThresholdTable


class RawParams(FrozenModel):
    pass


class RawPolicy:
    name: ClassVar[str] = "raw"
    Params: ClassVar[type[BaseModel]] = RawParams

    def __init__(self, params: RawParams | None = None) -> None:
        self._params = params or RawParams()

    def decide(self, result: CheckResult, thresholds: ThresholdTable) -> CheckResult:
        return result


PLUGINS = (RawPolicy,)
