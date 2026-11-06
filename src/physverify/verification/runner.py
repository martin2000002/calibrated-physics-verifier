"""Runs the checks over check inputs, applies the verdict policy, writes the run."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum, StrEnum
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import Field

from physverify.checks._base import Check, CheckInputs, abstain, missing_inputs, row_step_or_none
from physverify.config.models import Settings, validated
from physverify.core.errors import ConfigError
from physverify.runtime.parallel import ordered_map
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.io import write_jsonl
from physverify.verification._base import VerdictPolicy
from physverify.verification.check_routes import VERIFICATION_SECTION as VERIFICATION_SECTION
from physverify.verification.check_routes import CheckRouting
from physverify.verification.check_set import (
    CHECKS_SECTION,
    ChecksSettings,
    active_checks_hash,
    build_checks,
)
from physverify.verification.decisive import Judged, applicable, decisive
from physverify.verification.inputs import ClipInputs, InputSource
from physverify.verification.sweep import CheckUnit, SpecificitySettings
from physverify.verification.thresholding import (
    THRESHOLD_POLICY,
    ThresholdTable,
    load_thresholds,
    threshold_problem,
)

CHECKS_FILE = "checks.jsonl"
TABLE_FILE = "table.csv"
CHECKS_HASH_FILE = "checks_hash.txt"
"""The run's ``checks_hash`` (verification.check_set.active_checks_hash): its check parameters."""
THRESHOLDS_KEY = "thresholds"
UNKNOWN_ROW_STEP = 0
"""``row_step`` of a clip whose table has no valid time axis."""


def run_checks_hash(run_dir: Path) -> str:
    """The ``checks_hash`` a ``verify`` run recorded; empty when it records none (an older run)."""
    path = run_dir / CHECKS_HASH_FILE
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


class VerificationSettings(CheckRouting):
    policy: str = THRESHOLD_POLICY
    thresholds: dict[str, float] = Field(default_factory=dict)
    threshold_units: dict[str, str] = Field(default_factory=dict)
    policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    specificity: SpecificitySettings = Field(default_factory=SpecificitySettings)


def build_policy(
    registry: Mapping[str, type[VerdictPolicy]], settings: VerificationSettings
) -> VerdictPolicy:
    if settings.policy not in registry:
        raise ConfigError(f"unknown verdict policy {settings.policy!r}; known: {sorted(registry)}")
    policy = registry[settings.policy]
    raw = settings.policies.get(settings.policy, {})
    return policy(validated(policy.Params, raw, f"verification.policies.{settings.policy}"))


class RowStatus(StrEnum):
    SCORED = "scored"
    ABSTAINED = "abstained"
    NO_EVENT = "no_event"
    ERROR = "error"


@dataclass(frozen=True)
class TableRow:
    clip_id: str
    check_id: str
    status: RowStatus
    magnitude: float | None
    sigma: float
    verdict: Verdict | None
    abstain_reason: AbstainReason | None
    n_subjects: int
    n_abstained: int
    unit: str
    row_step: int


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[CheckResult, ...]
    table: tuple[TableRow, ...]
    clips: tuple[ClipManifest, ...]


class VerificationRunner:
    """Every check on every clip: 9 table rows per clip, whatever happens inside a check."""

    def __init__(
        self,
        checks: Sequence[Check],
        policy: VerdictPolicy,
        thresholds: ThresholdTable,
        workers: int = 1,
    ) -> None:
        self._checks = tuple(checks)
        self._policy = policy
        self._thresholds = thresholds
        self._label = thresholds.label
        self._workers = workers

    def run(self, source: InputSource, clips: Sequence[ClipManifest]) -> VerificationReport:
        """With several workers each clip is loaded and checked alone in a worker process and
        the reports are joined in clip order."""
        if self._workers == 1:
            return self._run(source, clips)
        parts = ordered_map(partial(self._run_clip, source), clips, self._workers)
        return VerificationReport(
            tuple(result for part in parts for result in part.results),
            tuple(row for part in parts for row in part.table),
            tuple(clip for part in parts for clip in part.clips),
        )

    def _run_clip(self, source: InputSource, clip: ClipManifest) -> VerificationReport:
        return self._run(source, [clip])

    def _run(self, source: InputSource, clips: Sequence[ClipManifest]) -> VerificationReport:
        results: list[CheckResult] = []
        table: list[TableRow] = []
        seen: list[ClipManifest] = []
        for item in source.load(clips):
            seen.append(item.clip)
            step = _row_step(item.inputs)
            for check in self._checks:
                decided, failed = self._evaluate(check, item.for_check(check.check_id))
                results.extend(decided)
                row = aggregate(item.clip.clip_id, check, decided, row_step=step)
                table.append(replace(row, status=RowStatus.ERROR) if failed else row)
        return VerificationReport(tuple(results), tuple(table), tuple(seen))

    def _evaluate(self, check: Check, item: ClipInputs) -> tuple[list[CheckResult], bool]:
        if item.inputs is None:
            error = item.error or ""
            return [self._missing(check, item, {"error": error})], False
        missing = missing_inputs(check, item.inputs)
        if missing:
            return [self._missing(check, item, {"missing": missing})], False
        try:
            found = check.evaluate(item.inputs)
            return [self._decide(result) for result in found], False
        except Exception as exc:  # noqa: BLE001 - a failing check must not abort the run
            error = f"{type(exc).__name__}: {exc}"
            return [self._missing(check, item, {"error": error})], True

    def _decide(self, result: CheckResult) -> CheckResult:
        decided = self._policy.decide(result, self._thresholds)
        return replace(decided, provenance={**decided.provenance, THRESHOLDS_KEY: self._label})

    def _missing(self, check: Check, item: ClipInputs, details: Mapping[str, Any]) -> CheckResult:
        reason = AbstainReason.MISSING_INPUT
        if item.inputs is not None:
            result = abstain(type(check), item.inputs, (), reason, None, details)
        else:
            result = _abstain_without_inputs(check, item.clip, item.input_level, details)
        return replace(result, provenance={**result.provenance, THRESHOLDS_KEY: self._label})


@dataclass(frozen=True)
class ConfiguredSuite:
    """The configured checks, verdict policy and thresholds of a run."""

    verification: VerificationSettings
    checks: tuple[Check, ...]
    policy: VerdictPolicy
    thresholds: ThresholdTable

    def runner(self) -> VerificationRunner:
        return VerificationRunner(self.checks, self.policy, self.thresholds)


def configured_suite(
    settings: Settings,
    checks: Mapping[str, type[Check]],
    policies: Mapping[str, type[VerdictPolicy]],
    *,
    policy: str | None = None,
) -> ConfiguredSuite:
    """The ``[checks]`` and ``[verification]`` sections built over the registries (``policy``
    overrides the configured one), with the thresholds of ``guarded_thresholds``."""
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    if policy is not None:
        verification = verification.model_copy(update={"policy": policy})
    overrides = settings.section(CHECKS_SECTION, ChecksSettings)
    built = build_checks(checks, overrides, verification)
    decider = build_policy(policies, verification)
    return ConfiguredSuite(verification, built, decider, guarded_thresholds(settings, checks))


def guarded_thresholds(settings: Settings, checks: Mapping[str, type[Check]]) -> ThresholdTable:
    """The threshold table every consumer of ``[verification]`` thresholds loads, through
    ``guard_table``."""
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    table = load_thresholds(settings.paths.thresholds, verification.thresholds)
    return guard_table(table, settings, checks)


def guard_table(
    thresholds: ThresholdTable, settings: Settings, checks: Mapping[str, type[Check]]
) -> ThresholdTable:
    """``thresholds`` when it may judge the configured checks: a calibrated table of other checks
    (``checks_hash``) or a tau in another unit than the check reports is a ``ConfigError``."""
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    units = {c.check_id: CheckUnit.of(c) for c in checks.values()}
    written = {
        check_id: verification.threshold_units.get(check_id, unit.written_unit)
        for check_id, unit in units.items()
    }
    reported = {check_id: unit.magnitude_unit for check_id, unit in units.items()}
    current = active_checks_hash(settings, checks)
    problem = threshold_problem(thresholds, current, written, reported)
    if problem is not None:
        raise ConfigError(problem)
    return thresholds


def aggregate(
    clip_id: str,
    check: Check,
    results: Sequence[CheckResult],
    *,
    row_step: int = UNKNOWN_ROW_STEP,
) -> TableRow:
    """Magnitude and sigma of the ``decisive`` scored subject; FAIL if any fails, PASS if all do.
    Only the ``applicable`` subjects make the row."""
    kept = applicable(results)
    scored = [r for r in kept if r.magnitude is not None]
    worst = decisive(scored)
    reasons = Counter(r.abstain_reason for r in kept if r.abstain_reason is not None)
    return TableRow(
        clip_id=clip_id,
        check_id=check.check_id,
        status=_status(results, scored),
        magnitude=None if worst is None else worst.magnitude,
        sigma=math.nan if worst is None else worst.sigma,
        verdict=row_verdict(kept),
        abstain_reason=reasons.most_common(1)[0][0] if reasons and _abstains(kept) else None,
        n_subjects=len(kept),
        n_abstained=sum(1 for r in kept if r.verdict is Verdict.ABSTAIN),
        unit=check.magnitude_unit,
        row_step=row_step,
    )


def _status(results: Sequence[CheckResult], scored: Sequence[CheckResult]) -> RowStatus:
    """NO_EVENT without results or when every one is NOT_APPLICABLE."""
    if not applicable(results):
        return RowStatus.NO_EVENT
    return RowStatus.SCORED if scored else RowStatus.ABSTAINED


def row_verdict(results: Sequence[Judged]) -> Verdict | None:
    """FAIL if any ``applicable`` subject fails, PASS if all pass, else UNSCORED if any is, else
    ABSTAIN; None without applicable subjects."""
    kept = applicable(results)
    if not kept:
        return None
    verdicts = {r.verdict for r in kept}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    if verdicts == {Verdict.PASS}:
        return Verdict.PASS
    return Verdict.UNSCORED if Verdict.UNSCORED in verdicts else Verdict.ABSTAIN


def _abstains(results: Sequence[CheckResult]) -> bool:
    return row_verdict(results) is Verdict.ABSTAIN


def _row_step(inputs: CheckInputs | None) -> int:
    step = None if inputs is None else row_step_or_none(inputs)
    return UNKNOWN_ROW_STEP if step is None else step


def _abstain_without_inputs(
    check: Check, clip: ClipManifest, level: InputLevel, details: Mapping[str, Any]
) -> CheckResult:
    return CheckResult(
        check_id=check.check_id,
        clip_id=clip.clip_id,
        subject=(),
        frame_span=None,
        magnitude=None,
        sigma=math.nan,
        unit=check.magnitude_unit,
        verdict=Verdict.ABSTAIN,
        input_level=level,
        abstain_reason=AbstainReason.MISSING_INPUT,
        details={**details, "row_step": None},
    )


def write_report(run_dir: Path, report: VerificationReport) -> None:
    write_jsonl(run_dir / CHECKS_FILE, report.results)
    columns = [field.name for field in fields(TableRow)]
    with (run_dir / TABLE_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report.table:
            writer.writerow({name: _cell(getattr(row, name)) for name in columns})


def _cell(value: object) -> object:
    if value is None:
        return ""
    return value.value if isinstance(value, Enum) else value
