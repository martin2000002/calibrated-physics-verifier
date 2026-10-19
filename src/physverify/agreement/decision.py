"""Clip-level decision of the verifier from one verify run (ADR 0011 D5, Juicio change 1).

``score`` is always an implausibility (higher = more evidence of a violation). The provisional
rule decides by itself; the calibrated rule only scores, and the frozen dev cut decides.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from physverify.core.errors import ConfigError, PhysverifyError
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.verification.runner import RowStatus, TableRow, tau_of, thresholds_label

if TYPE_CHECKING:
    from pathlib import Path

    from physverify.calibration.loader import Stratum
    from physverify.config.models import Settings

ERROR_REASON = "check_error"
PROVISIONAL_NAME = "provisional_any_fail"
CALIBRATED_NAME = "calibrated"
RULES = (PROVISIONAL_NAME, CALIBRATED_NAME)


class ClipDecision(StrEnum):
    PLAUSIBLE = "plausible"
    IMPLAUSIBLE = "implausible"
    NO_EVIDENCE = "no_evidence"
    ABSTAIN = "abstain"


DECIDED = (ClipDecision.PLAUSIBLE, ClipDecision.IMPLAUSIBLE)


@dataclass(frozen=True)
class Decision:
    """``fixed`` is the rule's own outcome; None means the cut on ``score`` decides."""

    clip_id: str
    fixed: ClipDecision | None
    score: float
    deciding_check: str | None
    reasons: tuple[str, ...]

    def outcome(self, cut: float | None) -> ClipDecision:
        if self.fixed is not None:
            return self.fixed
        if cut is None:
            raise ConfigError("the calibrated rule needs the dev cut frozen by `agreement freeze`")
        return ClipDecision.IMPLAUSIBLE if self.score > cut else ClipDecision.PLAUSIBLE


class ClipDecisionRule(Protocol):
    name: str

    @property
    def label(self) -> str: ...

    @property
    def kubric_threshold(self) -> float | None:
        """A cut on the score fixed outside VideoPhy (A8's clean-scene quantile), if any."""
        ...

    def tau(self, check_id: str) -> float | None: ...

    def excluding(self, excluded: Collection[str]) -> ClipDecisionRule:
        """The same rule with ``excluded`` checks also left out."""
        ...

    def decide(
        self, clip_id: str, rows: Sequence[TableRow], results: Sequence[CheckResult]
    ) -> Decision: ...


def _ratio(row: TableRow, thresholds: Mapping[str, Any]) -> float:
    tau = tau_of(thresholds, row.check_id)
    if row.magnitude is None or tau is None or tau <= 0:
        return math.nan
    return row.magnitude / tau


def _reason(row: TableRow) -> str | None:
    if row.status is RowStatus.ERROR:
        return ERROR_REASON
    return None if row.abstain_reason is None else row.abstain_reason.value


def _reasons(rows: Sequence[TableRow]) -> tuple[str, ...]:
    """The distinct abstention reasons of the clip's rows (errors as ``check_error``)."""
    return tuple(sorted({reason for r in rows if (reason := _reason(r)) is not None}))


class ProvisionalAnyFail:
    """IMPLAUSIBLE on any FAIL row; PLAUSIBLE with no FAIL and at least one PASS; NO_EVIDENCE when
    every row is no_event; ABSTAIN otherwise. Score: max over scored rows of magnitude / tau."""

    name = PROVISIONAL_NAME

    def __init__(self, thresholds: Mapping[str, Any], excluded: Collection[str] = ()) -> None:
        self._thresholds = thresholds
        self._excluded = frozenset(excluded)

    @property
    def label(self) -> str:
        return f"{self.name}/{thresholds_label(self._thresholds)}"

    @property
    def kubric_threshold(self) -> float | None:
        return None

    def tau(self, check_id: str) -> float | None:
        return tau_of(self._thresholds, check_id)

    def excluding(self, excluded: Collection[str]) -> ProvisionalAnyFail:
        return ProvisionalAnyFail(self._thresholds, self._excluded | frozenset(excluded))

    def decide(
        self, clip_id: str, rows: Sequence[TableRow], results: Sequence[CheckResult]
    ) -> Decision:
        kept = [r for r in rows if r.check_id not in self._excluded]
        ratios = {r.check_id: _ratio(r, self._thresholds) for r in kept}
        finite = {k: v for k, v in ratios.items() if math.isfinite(v)}
        score = max(finite.values(), default=math.nan)
        failed = [r for r in kept if r.verdict is Verdict.FAIL and r.status is not RowStatus.ERROR]
        deciding = max(failed, key=lambda r: finite.get(r.check_id, -math.inf), default=None)
        return Decision(
            clip_id=clip_id,
            fixed=_provisional_outcome(kept, bool(failed)),
            score=score,
            deciding_check=None if deciding is None else deciding.check_id,
            reasons=_reasons(kept),
        )


def _provisional_outcome(rows: Sequence[TableRow], failed: bool) -> ClipDecision:
    if failed:
        return ClipDecision.IMPLAUSIBLE
    if any(r.verdict is Verdict.PASS and r.status is RowStatus.SCORED for r in rows):
        return ClipDecision.PLAUSIBLE
    if rows and all(r.status is RowStatus.NO_EVENT for r in rows):
        return ClipDecision.NO_EVIDENCE
    return ClipDecision.ABSTAIN


class CalibrationStratum(Protocol):
    """What the rule reads from A8's stratum (ADR 0009 §2: per-check tau, aggregate cut, and
    the artefact#level/profile@run label)."""

    @property
    def source(self) -> str: ...

    def tau(self, check_id: str) -> float | None: ...

    def score_threshold(self) -> float | None: ...

    def check_status(self) -> dict[str, Any]: ...


class UnusableCalibrationError(PhysverifyError):
    """The calibrated stratum matches the run but cannot score yet (item 1.3): the workflow
    falls back to the provisional rule or stops, per ``[agreement].on_unusable_stratum``."""


class ScoredClip(Protocol):
    """A8's clip score (ADR 0009 §3): S in [0, 1] or None when it abstains."""

    @property
    def score(self) -> float | None: ...

    @property
    def reason(self) -> str | None: ...

    @property
    def abstained_checks(self) -> Sequence[str]: ...

    @property
    def per_check(self) -> Mapping[str, float]: ...


class ScoreClip[S: CalibrationStratum](Protocol):
    def __call__(
        self,
        results: Sequence[CheckResult],
        stratum: S,
        k_sigma: float | None = None,
        *,
        only: Collection[str] | None = None,
    ) -> ScoredClip: ...


class OpenStratum[S: CalibrationStratum](Protocol):
    """A8's ``open_stratum``: the one gate in front of a stratum."""

    def __call__(
        self,
        path: Path,
        level: InputLevel,
        profile: str,
        *,
        checks_hash: str,
        row_step: int,
        require_calibrated: bool = False,
    ) -> S: ...


class CalibratedRule[S: CalibrationStratum]:
    """A8's clip score S in [0, 1] (1 = no violation evidence) on the PERCEIVED stratum of the
    run's profile; the implausibility is 1 - S, and S = None is an abstention. The interval
    width is the stratum's own fitted k_sigma."""

    name = CALIBRATED_NAME

    def __init__(
        self,
        stratum: S,
        score_clip: ScoreClip[S],
        excluded: Collection[str] = (),
    ) -> None:
        self._stratum, self._score_clip = stratum, score_clip
        self._excluded = frozenset(excluded)

    @property
    def label(self) -> str:
        return f"{self.name}:{self._stratum.source}"

    @property
    def status(self) -> dict[str, Any]:
        """The stratum's state per check, for ``metrics["calibration"]``."""
        return self._stratum.check_status()

    @property
    def kubric_threshold(self) -> float | None:
        value = self._stratum.score_threshold()
        return None if value is None else float(value)

    def tau(self, check_id: str) -> float | None:
        value = self._stratum.tau(check_id)
        return None if value is None else float(value)

    def excluding(self, excluded: Collection[str]) -> CalibratedRule[S]:
        return CalibratedRule(self._stratum, self._score_clip, self._excluded | frozenset(excluded))

    def decide(
        self, clip_id: str, rows: Sequence[TableRow], results: Sequence[CheckResult]
    ) -> Decision:
        checks = {r.check_id for r in rows} | {r.check_id for r in results}
        only = None if not self._excluded else checks - self._excluded
        scored = self._score_clip(results, self._stratum, only=only)
        if scored.score is None:
            reason = str(scored.reason or "no_decided_check")
            return Decision(clip_id, ClipDecision.ABSTAIN, math.nan, None, (reason,))
        per_check = scored.per_check
        deciding = max(per_check, key=lambda k: per_check[k], default=None)
        reasons = tuple(sorted(str(c) for c in scored.abstained_checks))
        return Decision(clip_id, None, 1.0 - float(scored.score), deciding, reasons)


@dataclass(frozen=True)
class CalibrationApi[S: CalibrationStratum]:
    """The A8 entry points, injected so tests can pass a fake. ``error`` is a configuration
    problem (missing or stale artifact, another stratum or time base); ``unusable_error``, a
    subclass of it, a stratum that cannot score yet."""

    open_stratum: OpenStratum[S]
    default_artifact_path: Callable[[Settings], Path]
    active_checks_hash: Callable[[Settings], str]
    require_calibrated: Callable[[Settings], bool]
    score_clip: ScoreClip[S]
    error: type[Exception]
    unusable_error: type[Exception]


def import_calibration() -> CalibrationApi[Stratum]:
    """A8's loader and scorer, imported only when the calibrated rule is asked for."""
    try:
        from physverify.calibration.apply import score_clip
        from physverify.calibration.loader import (
            CalibrationError,
            UnusableStratumError,
            default_artifact_path,
            open_stratum,
        )
        from physverify.calibration.settings import active_checks_hash, calibration_settings
    except ImportError as exc:
        raise ConfigError(f"the calibrated rule needs A8's calibration package: {exc}") from exc
    return CalibrationApi(
        open_stratum=open_stratum,
        default_artifact_path=default_artifact_path,
        active_checks_hash=active_checks_hash,
        require_calibrated=lambda s: calibration_settings(s).require_calibrated_artifact,
        score_clip=score_clip,
        error=CalibrationError,
        unusable_error=UnusableStratumError,
    )


@dataclass(frozen=True)
class StratumRequest:
    """What the run asks of the stratum: its profile, time base and check parameters."""

    artifact: Path
    profile: str
    row_step: int
    checks_hash: str
    require_calibrated: bool = False


def calibrated_rule[S: CalibrationStratum](
    api: CalibrationApi[S], request: StratumRequest, excluded: Collection[str]
) -> CalibratedRule[S]:
    """A configuration problem is a ``ConfigError`` (exit 2), never a silent fallback; a
    stratum that cannot score yet is ``UnusableCalibrationError``, which the caller handles."""
    try:
        stratum = api.open_stratum(
            request.artifact,
            InputLevel.PERCEIVED,
            request.profile,
            checks_hash=request.checks_hash,
            row_step=request.row_step,
            require_calibrated=request.require_calibrated,
        )
    except api.unusable_error as exc:
        raise UnusableCalibrationError(str(exc)) from exc
    except api.error as exc:
        raise ConfigError(f"calibration {request.artifact}: {exc}") from exc
    return CalibratedRule(stratum, api.score_clip, excluded)
