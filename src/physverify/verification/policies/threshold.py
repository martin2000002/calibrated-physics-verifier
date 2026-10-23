"""PASS/FAIL when the k-sigma interval of a magnitude clears its threshold.

A calibrated entry also corrects the magnitude by its measured bias and scales its
sigma by the measured factor before the comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar, Literal

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.runner import ThresholdEntry, entry_of

Decision = tuple[Verdict, AbstainReason | None]


class ThresholdParams(FrozenModel):
    k_sigma: float = Field(default=2.0, ge=0.0)
    on_unknown_sigma: Literal["abstain", "point"] = "abstain"


class ThresholdPolicy:
    name: ClassVar[str] = "threshold"
    Params: ClassVar[type[ThresholdParams]] = ThresholdParams

    def __init__(self, params: ThresholdParams | None = None) -> None:
        self._params = params or ThresholdParams()

    def decide(self, result: CheckResult, thresholds: Mapping[str, Any]) -> CheckResult:
        if result.verdict is Verdict.ABSTAIN or result.magnitude is None:
            return result
        entry = entry_of(thresholds, result.check_id)
        if entry is None:
            return result
        verdict, reason = self.judge(result.magnitude, result.sigma, entry)
        return replace(result, verdict=verdict, abstain_reason=reason, threshold=entry.tau)

    def judge(self, magnitude: float, sigma: float, entry: ThresholdEntry) -> Decision:
        """Verdict of m - bias against tau with the interval k * factor * sigma."""
        factor = math.nan if entry.sigma_factor is None else entry.sigma_factor
        return self._judge(magnitude - entry.bias, factor * sigma, entry.tau)

    def _judge(self, magnitude: float, sigma: float, tau: float) -> Decision:
        if math.isnan(sigma):
            if self._params.on_unknown_sigma == "abstain":
                return Verdict.ABSTAIN, AbstainReason.UNKNOWN_UNCERTAINTY
            sigma = 0.0
        margin = self._params.k_sigma * sigma
        if magnitude - margin > tau:
            return Verdict.FAIL, None
        if magnitude + margin <= tau:
            return Verdict.PASS, None
        return Verdict.ABSTAIN, AbstainReason.INTERVAL_CROSSES_THRESHOLD


PLUGINS = (ThresholdPolicy,)
