"""Clip-level decision of the verifier from one verify run.

``score`` is always an implausibility (higher = more evidence of a violation). The provisional
rule decides by itself; the calibrated rule only scores, and the frozen dev cut decides; the
calibrated fail-only rule decides by itself on the calibrated per-check verdicts.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from physverify.calibration.api import CalibrationApi, CalibrationStratum, ScoreClip, ScoredClip
from physverify.core.errors import ConfigError, PhysverifyError
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.verification.runner import RowStatus, TableRow
from physverify.verification.thresholding import ThresholdTable

if TYPE_CHECKING:
    from pathlib import Path

ERROR_REASON = "check_error"
PROVISIONAL_NAME = "provisional_any_fail"
CALIBRATED_NAME = "calibrated"
CALIBRATED_FAIL_ONLY_NAME = "calibrated_fail_only"
CALIBRATED_RULES = (CALIBRATED_NAME, CALIBRATED_FAIL_ONLY_NAME)
RULES = (PROVISIONAL_NAME, *CALIBRATED_RULES)
NO_FAILED_REASON = "no_failed_check"
NO_EVENT_REASON = "no_event"


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
    decided_checks: tuple[str, ...] = ()
    """Active checks with a calibrated PASS or FAIL; empty under the provisional rule."""

    def outcome(self, cut: float | None) -> ClipDecision:
        if self.fixed is not None:
            return self.fixed
        if cut is None:
            raise ConfigError("the calibrated rule needs the dev cut frozen by `agreement freeze`")
        return ClipDecision.IMPLAUSIBLE if self.score > cut else ClipDecision.PLAUSIBLE


class ClipDecisionRule(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def kubric_threshold(self) -> float | None:
        """A cut on the score fixed outside VideoPhy (the calibration chain's clean-scene quantile),
        if any."""
        ...

    def tau(self, check_id: str) -> float | None: ...

    def excluding(self, excluded: Collection[str]) -> ClipDecisionRule:
        """The same rule with ``excluded`` checks also left out."""
        ...

    def decide(
        self, clip_id: str, rows: Sequence[TableRow], results: Sequence[CheckResult]
    ) -> Decision: ...


def _ratio(row: TableRow, thresholds: ThresholdTable) -> float:
    tau = thresholds.tau_of(row.check_id)
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

    def __init__(self, thresholds: ThresholdTable, excluded: Collection[str] = ()) -> None:
        self._thresholds = thresholds
        self._excluded = frozenset(excluded)

    @property
    def label(self) -> str:
        return f"{self.name}/{self._thresholds.label}"

    @property
    def kubric_threshold(self) -> float | None:
        return None

    def tau(self, check_id: str) -> float | None:
        return self._thresholds.tau_of(check_id)

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


class UnusableCalibrationError(PhysverifyError):
    """The calibrated stratum matches the run but cannot score yet: the workflow falls back to the
    provisional rule or stops, per ``[agreement].on_unusable_stratum``."""


class CalibratedRule[S: CalibrationStratum]:
    """The calibration chain's clip score S in [0, 1] (1 = no violation evidence) on the PERCEIVED
    stratum of the run's profile; the implausibility is 1 - S, and S = None is an abstention. The
    interval width is the stratum's own fitted k_sigma (``k_sigma`` pins it: another value fails).

    ``fail_only``: IMPLAUSIBLE when a voting check (tau and weight > 0) FAILs and ABSTAIN otherwise
    (NO_EVIDENCE without any event). A PASS is no evidence of plausibility, since a check that sees
    no violation of its own law says nothing about the others; the score is still reported."""

    def __init__(
        self,
        stratum: S,
        score_clip: ScoreClip[S],
        excluded: Collection[str] = (),
        *,
        k_sigma: float | None = None,
        fail_only: bool = False,
    ) -> None:
        self._stratum, self._score_clip = stratum, score_clip
        self._excluded = frozenset(excluded)
        self._k_sigma, self._fail_only = k_sigma, fail_only

    @property
    def name(self) -> str:
        return CALIBRATED_FAIL_ONLY_NAME if self._fail_only else CALIBRATED_NAME

    @property
    def label(self) -> str:
        return f"{self.name}:{self._stratum.source}"

    @property
    def status(self) -> dict[str, Any]:
        """The stratum's state per check, for ``metrics["calibration"]``."""
        return self._stratum.check_status()

    @property
    def kubric_threshold(self) -> float | None:
        value = None if self._fail_only else self._stratum.score_threshold()
        return None if value is None else float(value)

    def tau(self, check_id: str) -> float | None:
        value = self._stratum.tau(check_id)
        return None if value is None else float(value)

    def excluding(self, excluded: Collection[str]) -> CalibratedRule[S]:
        return CalibratedRule(
            self._stratum,
            self._score_clip,
            self._excluded | frozenset(excluded),
            k_sigma=self._k_sigma,
            fail_only=self._fail_only,
        )

    def decide(
        self, clip_id: str, rows: Sequence[TableRow], results: Sequence[CheckResult]
    ) -> Decision:
        checks = {r.check_id for r in rows} | {r.check_id for r in results}
        only = None if not self._excluded else checks - self._excluded
        scored = self._score_clip(results, self._stratum, self._k_sigma, only=only)
        if self._fail_only:
            return _fail_only(clip_id, scored)
        if scored.score is None:
            reason = str(scored.reason or "no_decided_check")
            return Decision(clip_id, ClipDecision.ABSTAIN, math.nan, None, (reason,))
        per_check = scored.per_check
        deciding = max(per_check, key=lambda k: per_check[k], default=None)
        reasons = tuple(sorted(str(c) for c in scored.abstained_checks))
        return Decision(
            clip_id, None, 1.0 - float(scored.score), deciding, reasons, _decided(scored)
        )


def _decided(scored: ScoredClip) -> tuple[str, ...]:
    return tuple(sorted(str(c) for c in scored.decided_checks))


def _fail_only(clip_id: str, scored: ScoredClip) -> Decision:
    score = math.nan if scored.score is None else 1.0 - float(scored.score)
    reasons = tuple(sorted(str(c) for c in scored.abstained_checks))
    if scored.failed_checks:
        per_check = scored.per_check
        deciding = max(scored.failed_checks, key=lambda c: per_check.get(c, -math.inf))
        return Decision(
            clip_id, ClipDecision.IMPLAUSIBLE, score, deciding, reasons, _decided(scored)
        )
    events = (*scored.decided_checks, *scored.abstained_checks, *scored.unweighted_checks)
    if not events:
        return Decision(clip_id, ClipDecision.NO_EVIDENCE, score, None, (NO_EVENT_REASON,))
    why = reasons or (NO_FAILED_REASON,)
    return Decision(clip_id, ClipDecision.ABSTAIN, score, None, why, _decided(scored))


@dataclass(frozen=True)
class StratumRequest:
    """What the run asks of the stratum: its profile, time base and check parameters."""

    artifact: Path
    profile: str
    row_step: int
    checks_hash: str
    require_calibrated: bool = False


def calibrated_rule[S: CalibrationStratum](
    api: CalibrationApi[S],
    request: StratumRequest,
    excluded: Collection[str],
    *,
    k_sigma: float | None = None,
    fail_only: bool = False,
) -> CalibratedRule[S]:
    """A configuration problem is a ``ConfigError`` (exit 2), never a silent fallback; a
    stratum that cannot score yet is ``UnusableCalibrationError``, which the caller handles. A
    ``k_sigma`` other than the stratum's fails here, before any clip is scored."""
    try:
        stratum = api.open_stratum(
            request.artifact,
            InputLevel.PERCEIVED,
            request.profile,
            checks_hash=request.checks_hash,
            row_step=request.row_step,
            require_calibrated=request.require_calibrated,
        )
        api.score_clip((), stratum, k_sigma)
    except api.unusable_error as exc:
        raise UnusableCalibrationError(str(exc)) from exc
    except api.error as exc:
        raise ConfigError(f"calibration {request.artifact}: {exc}") from exc
    return CalibratedRule(stratum, api.score_clip, excluded, k_sigma=k_sigma, fail_only=fail_only)
