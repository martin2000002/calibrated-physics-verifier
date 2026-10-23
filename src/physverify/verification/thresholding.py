"""The threshold table every verdict reads, and the k-sigma rule that turns a magnitude into one.

A calibrated entry also corrects the magnitude by its measured bias and scales its sigma by the
measured factor before the comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from physverify.config.models import validated
from physverify.core.errors import ConfigError
from physverify.core.jsonio import loads
from physverify.core.models import FrozenModel
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

PROVISIONAL = "provisional"
CALIBRATED = "calibrated"
PROVISIONAL_SOURCE = "configs/checks.toml [verification.thresholds]"

Decision = tuple[Verdict, AbstainReason | None]


class ThresholdEntry(FrozenModel):
    """``tau`` in check units. The calibration chain may add the measured bias of the
    magnitude and the factor of its sigma; ``sigma_factor`` None: the error was not measured."""

    tau: float
    bias: float = 0.0
    sigma_factor: float | None = Field(default=1.0, ge=0.0)


class ArtifactRef(FrozenModel):
    path: str
    sha256: str


class ThresholdTable(FrozenModel):
    """Shape of ``paths.thresholds`` (written by the calibration chain) and of the provisional
    set. ``dump`` keeps only the fields the table was built with: a provisional or raw table
    carries its taus alone, a table read from a file every field."""

    status: Literal["provisional", "calibrated"]
    source: str
    checks: dict[str, ThresholdEntry]
    artifact: ArtifactRef | None = None

    @classmethod
    def provisional(cls, taus: Mapping[str, float]) -> Self:
        checks = {check_id: ThresholdEntry(tau=float(tau)) for check_id, tau in taus.items()}
        return cls(status=PROVISIONAL, source=PROVISIONAL_SOURCE, checks=checks)

    @classmethod
    def complete(cls, data: object, where: str) -> Self:
        """A validated table with every field set, defaults included."""
        return cls.model_validate(validated(cls, data, where).model_dump())

    def raw(self) -> ThresholdTable:
        """The same taus with no calibration correction (bias 0, sigma factor 1)."""
        checks = {check_id: ThresholdEntry(tau=e.tau) for check_id, e in self.checks.items()}
        return ThresholdTable(status=self.status, source=self.source, checks=checks, artifact=None)

    def entry_of(self, check_id: str) -> ThresholdEntry | None:
        return self.checks.get(check_id)

    def tau_of(self, check_id: str) -> float | None:
        entry = self.checks.get(check_id)
        return None if entry is None else entry.tau

    @property
    def calibrated(self) -> bool:
        return self.status == CALIBRATED

    @property
    def label(self) -> str:
        return f"{CALIBRATED}:{self.source}" if self.calibrated else PROVISIONAL

    def dump(self) -> dict[str, Any]:
        """What is hashed and reported."""
        return self.model_dump(exclude_unset=True)


def load_thresholds(path: Path, provisional: Mapping[str, float]) -> ThresholdTable:
    """The calibrated table at ``path`` when it exists, else the provisional thresholds."""
    if not path.is_file():
        return ThresholdTable.provisional(provisional)
    data = loads(path.read_text(encoding="utf-8"), path, ConfigError)
    return ThresholdTable.complete(data, str(path))


class ThresholdParams(FrozenModel):
    k_sigma: float = Field(default=2.0, ge=0.0)
    on_unknown_sigma: Literal["abstain", "point"] = "abstain"


class ThresholdRule:
    """PASS/FAIL when the k-sigma interval of a magnitude clears its threshold."""

    def __init__(self, params: ThresholdParams | None = None) -> None:
        self._params = params or ThresholdParams()

    def decide(self, result: CheckResult, thresholds: ThresholdTable) -> CheckResult:
        if result.verdict is Verdict.ABSTAIN or result.magnitude is None:
            return result
        entry = thresholds.entry_of(result.check_id)
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
