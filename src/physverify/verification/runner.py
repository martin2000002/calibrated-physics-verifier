"""Runs the checks over check inputs, applies the verdict policy, writes the run (ADR 0006 §3.8)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from physverify.checks._base import Check, CheckInputs
from physverify.checks._common import abstain, row_step_or_none
from physverify.core.errors import ConfigError
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.io import write_jsonl
from physverify.verification._base import VerdictPolicy
from physverify.verification.decisive import Judged, decisive
from physverify.verification.inputs import ClipInputs, InputSource
from physverify.verification.sweep import SpecificitySettings

CHECKS_SECTION = "checks"
VERIFICATION_SECTION = "verification"
CHECKS_FILE = "checks.jsonl"
TABLE_FILE = "table.csv"
PROVISIONAL = "provisional"
CALIBRATED = "calibrated"
PROVISIONAL_SOURCE = "configs/checks.toml [verification.thresholds]"
THRESHOLDS_KEY = "thresholds"
UNKNOWN_ROW_STEP = 0
"""``row_step`` of a clip whose table has no valid time axis."""


class VerificationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: str = "threshold"
    thresholds: dict[str, float] = Field(default_factory=dict)
    policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    specificity: SpecificitySettings = Field(default_factory=SpecificitySettings)


class ChecksSettings(RootModel[dict[str, dict[str, Any]]]):
    root: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ThresholdEntry(BaseModel):
    """``tau`` in check units. The H1 calibration (ADR 0009) may add the measured bias of the
    magnitude and the factor of its sigma; ``sigma_factor`` None: the error was not measured."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tau: float
    bias: float = 0.0
    sigma_factor: float | None = Field(default=1.0, ge=0.0)


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str


class ThresholdTable(BaseModel):
    """Shape of ``paths.thresholds`` (written by the H1 calibration) and of the provisional set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["provisional", "calibrated"]
    source: str
    checks: dict[str, ThresholdEntry]
    artifact: ArtifactRef | None = None


def build_checks(
    registry: Mapping[str, type[Check]], overrides: ChecksSettings
) -> tuple[Check, ...]:
    unknown = sorted(set(overrides.root) - set(registry))
    if unknown:
        raise ConfigError(f"[checks] has unknown check ids {unknown}; known: {sorted(registry)}")
    return tuple(
        check(_validated(check.Params, overrides.root.get(check_id, {}), f"checks.{check_id}"))
        for check_id, check in registry.items()
    )


def build_policy(
    registry: Mapping[str, type[VerdictPolicy]], settings: VerificationSettings
) -> VerdictPolicy:
    if settings.policy not in registry:
        raise ConfigError(f"unknown verdict policy {settings.policy!r}; known: {sorted(registry)}")
    policy = registry[settings.policy]
    params_model: type[BaseModel] | None = getattr(policy, "Params", None)
    raw = settings.policies.get(settings.policy, {})
    if params_model is None:
        return policy()
    section = f"verification.policies.{settings.policy}"
    return policy(_validated(params_model, raw, section))  # type: ignore[call-arg]


def load_thresholds(path: Path, provisional: Mapping[str, float]) -> Mapping[str, Any]:
    """The calibrated table at ``path`` when it exists, else the provisional thresholds."""
    if not path.is_file():
        checks = {check_id: {"tau": float(tau)} for check_id, tau in provisional.items()}
        return {"status": PROVISIONAL, "source": PROVISIONAL_SOURCE, "checks": checks}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return _validated(ThresholdTable, data, str(path)).model_dump()


def tau_of(thresholds: Mapping[str, Any], check_id: str) -> float | None:
    entry = entry_of(thresholds, check_id)
    return None if entry is None else entry.tau


def entry_of(thresholds: Mapping[str, Any], check_id: str) -> ThresholdEntry | None:
    entry = thresholds.get("checks", {}).get(check_id)
    return None if entry is None else _validated(ThresholdEntry, entry, f"thresholds.{check_id}")


def thresholds_label(thresholds: Mapping[str, Any]) -> str:
    if thresholds.get("status") == CALIBRATED:
        return f"{CALIBRATED}:{thresholds.get('source', '')}"
    return PROVISIONAL


def _validated[M: BaseModel](model: type[M], data: object, section: str) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"[{section}] is invalid:\n{exc}") from exc


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
        self, checks: Sequence[Check], policy: VerdictPolicy, thresholds: Mapping[str, Any]
    ) -> None:
        self._checks = tuple(checks)
        self._policy = policy
        self._thresholds = thresholds
        self._label = thresholds_label(thresholds)

    def run(self, source: InputSource, clips: Sequence[ClipManifest]) -> VerificationReport:
        results: list[CheckResult] = []
        table: list[TableRow] = []
        seen: list[ClipManifest] = []
        for item in source.load(clips):
            seen.append(item.clip)
            step = _row_step(item.inputs)
            for check in self._checks:
                decided, failed = self._evaluate(check, item)
                results.extend(decided)
                row = aggregate(item.clip.clip_id, check, decided, row_step=step)
                table.append(replace(row, status=RowStatus.ERROR) if failed else row)
        return VerificationReport(tuple(results), tuple(table), tuple(seen))

    def _evaluate(self, check: Check, item: ClipInputs) -> tuple[list[CheckResult], bool]:
        if item.inputs is None:
            error = item.error or ""
            return [self._missing(check, item, {"error": error})], False
        missing = sorted(name for name in check.requires if getattr(item.inputs, name) is None)
        if missing:
            return [self._missing(check, item, {"missing": missing})], False
        try:
            found = check.evaluate(item.inputs)
            return [self._decide(result) for result in found], False
        except Exception as exc:  # a failing check must not abort the run (ADR 0006 §3.8)
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


def aggregate(
    clip_id: str,
    check: Check,
    results: Sequence[CheckResult],
    *,
    row_step: int = UNKNOWN_ROW_STEP,
) -> TableRow:
    """Magnitude and sigma of the ``decisive`` scored subject; FAIL if any fails, PASS if all do."""
    scored = [r for r in results if r.magnitude is not None]
    worst = decisive(scored)
    reasons = Counter(r.abstain_reason for r in results if r.abstain_reason is not None)
    return TableRow(
        clip_id=clip_id,
        check_id=check.check_id,
        status=_status(results, scored),
        magnitude=None if worst is None else worst.magnitude,
        sigma=math.nan if worst is None else worst.sigma,
        verdict=row_verdict(results),
        abstain_reason=reasons.most_common(1)[0][0] if reasons and _abstains(results) else None,
        n_subjects=len(results),
        n_abstained=sum(1 for r in results if r.verdict is Verdict.ABSTAIN),
        unit=check.magnitude_unit,
        row_step=row_step,
    )


def _status(results: Sequence[CheckResult], scored: Sequence[CheckResult]) -> RowStatus:
    if not results:
        return RowStatus.NO_EVENT
    return RowStatus.SCORED if scored else RowStatus.ABSTAINED


def row_verdict(results: Sequence[Judged]) -> Verdict | None:
    """FAIL if any subject fails, PASS if all pass, else UNSCORED if any is, else ABSTAIN."""
    if not results:
        return None
    verdicts = {r.verdict for r in results}
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
