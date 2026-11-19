"""Contract of a clip scorer: verifier results of one candidate -> one score in [0, 1].

A higher score is more physically consistent; an abstention is ``score = None``. ``only`` restricts
the score to a set of checks, so two candidates of one prompt are compared on the checks decided
in both and abstaining never pays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.config.plugin_context import PluginKind
from physverify.schema.checks import CheckResult


@dataclass(frozen=True)
class AggregateScore:
    score: float | None
    sigma: float | None
    abstained: bool
    reason: str | None
    n_decided: int
    n_failed: int
    coverage: float
    decided: frozenset[str] = frozenset()
    per_check: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class AggregateScorer(Protocol):
    """Constructed as ``cls(settings, checks)`` (the check registry); ``unavailable`` names what
    is missing (exit 2), or None."""

    name: ClassVar[str]
    version: ClassVar[str]
    calibrated: ClassVar[bool]

    def __init__(self, settings: Settings, checks: Mapping[str, type[Check]]) -> None: ...

    def unavailable(self) -> str | None: ...

    def thresholds(self) -> str: ...

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore: ...


@runtime_checkable
class CalibrationStatusSource(Protocol):
    """Optional scorer extra, not part of ``AggregateScorer``: the calibration state behind its
    scores."""

    def calibration_status(self) -> Mapping[str, Any] | None: ...


SCORERS = PluginKind[type[AggregateScorer]]("scorers")
