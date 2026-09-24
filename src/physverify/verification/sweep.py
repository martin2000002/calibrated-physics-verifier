"""Specificity sweep: each perturbation moves its own check and no other (ADR 0006 §3.9)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import kendalltau

from physverify.schema.checks import CheckResult
from physverify.schema.clip import ClipManifest
from physverify.schema.io import encode
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.synth.perturbations.none import NAME as NONE

if TYPE_CHECKING:
    from physverify.verification.runner import TableRow, VerificationReport

MATRIX_FILE = "sweep_matrix.csv"
SPECIFICITY_FILE = "specificity.json"
PERFECT_ORDER = 1.0


class SpecificitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    abs_tol: float = Field(default=1e-3, ge=0.0)
    rel_tol: float = Field(default=0.02, ge=0.0)
    null_tol: float = Field(default=1e-3, ge=0.0)
    couplings: list[tuple[str, str]] = Field(default_factory=list)
    """(perturbation, check) pairs that are expected to move together; reported, never failed.
    Declared in ``configs/checks.toml`` with their physical reason."""


class Rule(StrEnum):
    CHECK_UNIT = "check_unit"
    INJECTED_PARAMETER = "injected_parameter"
    NULL_BAND = "null_band"
    COUPLING = "coupling"


class CheckIdentity(Protocol):
    @property
    def check_id(self) -> str: ...

    @property
    def magnitude_unit(self) -> str: ...


@dataclass(frozen=True)
class CheckUnit:
    check_id: str
    magnitude_unit: str


@dataclass(frozen=True)
class MatrixCell:
    perturbation: str
    check_id: str
    median: float | None
    max: float | None
    n: int
    n_clips: int


@dataclass(frozen=True)
class RuleOutcome:
    rule: Rule
    perturbation: str
    check_id: str
    clips: tuple[str, ...]
    passed: bool
    expected: float | None
    observed: float | None
    detail: str = ""


@dataclass(frozen=True)
class SpecificityReport:
    outcomes: tuple[RuleOutcome, ...]
    null_band: Mapping[str, float]
    n_levels: Mapping[str, int]

    @property
    def failures(self) -> tuple[RuleOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.passed and o.rule is not Rule.COUPLING)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _Clip:
    clip: ClipManifest
    rows: Mapping[str, TableRow]
    results: tuple[CheckResult, ...]

    @property
    def record(self) -> PerturbationRecord | None:
        return self.clip.perturbation

    @property
    def name(self) -> str:
        return perturbation_name(self.clip)

    def magnitude(self, check_id: str) -> float | None:
        row = self.rows.get(check_id)
        return None if row is None else row.magnitude


def perturbation_name(clip: ClipManifest) -> str:
    return NONE if clip.perturbation is None else clip.perturbation.name


def target_check_ids(
    record: PerturbationRecord, checks: Sequence[CheckIdentity]
) -> tuple[str, ...]:
    """A4 labels name a check exactly (R1) or a family resolved by unit (R4 -> R4_jump)."""
    found: list[str] = []
    for label in record.target_checks:
        exact = [c.check_id for c in checks if c.check_id == label]
        family = [
            c.check_id
            for c in checks
            if c.check_id.startswith(f"{label}_") and c.magnitude_unit == record.magnitude_unit
        ]
        found.extend(exact or family)
    return tuple(dict.fromkeys(found))


def target_value(
    record: PerturbationRecord, check_id: str, results: Sequence[CheckResult]
) -> float | None:
    """Max magnitude of ``check_id`` over the subjects that cover the perturbed objects."""
    targets = set(record.target_objects)
    values = [
        r.magnitude
        for r in results
        if r.check_id == check_id and r.magnitude is not None and _covers(r.subject, targets)
    ]
    return max(values, default=None)


def _covers(subject: Sequence[str], targets: set[str]) -> bool:
    objects = set(subject)
    return objects <= targets or targets <= objects


def matrix(report: VerificationReport) -> list[MatrixCell]:
    """Perturbation x check: median, max and count of the clip magnitudes."""
    names = {clip.clip_id: perturbation_name(clip) for clip in report.clips}
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()
    for row in report.table:
        key = (names[row.clip_id], row.check_id)
        counts[key] += 1
        if row.magnitude is not None:
            values[key].append(row.magnitude)
    order = {check_id: index for index, check_id in enumerate(_check_ids(report.table))}
    keys = sorted(counts, key=lambda k: (k[0] != NONE, k[0], order[k[1]]))
    return [_cell(key, values.get(key, []), counts[key]) for key in keys]


def _cell(key: tuple[str, str], values: Sequence[float], n_clips: int) -> MatrixCell:
    return MatrixCell(
        perturbation=key[0],
        check_id=key[1],
        median=median(values) if values else None,
        max=max(values, default=None),
        n=len(values),
        n_clips=n_clips,
    )


def specificity(report: VerificationReport, settings: SpecificitySettings) -> SpecificityReport:
    checks = _check_units(report.table)
    clips = _clips(report)
    band = null_band(clips, [c.check_id for c in checks], settings.null_tol)
    outcomes = [
        *_check_unit_outcomes(clips, checks, settings),
        *_injected_outcomes(clips, checks),
        *_band_outcomes(clips, checks, band, settings),
    ]
    return SpecificityReport(tuple(outcomes), band, _n_levels(clips))


def null_band(
    clips: Sequence[_Clip], check_ids: Sequence[str], null_tol: float
) -> dict[str, float]:
    """Per check: max magnitude over the unperturbed variants plus ``null_tol``."""
    clean = [clip for clip in clips if clip.name == NONE]
    band: dict[str, float] = {}
    for check_id in check_ids:
        values = [m for clip in clean if (m := clip.magnitude(check_id)) is not None]
        band[check_id] = max(values, default=0.0) + null_tol
    return band


def _check_unit_outcomes(
    clips: Sequence[_Clip], checks: Sequence[CheckUnit], settings: SpecificitySettings
) -> list[RuleOutcome]:
    outcomes: list[RuleOutcome] = []
    for clip in _perturbed(clips, MagnitudeKind.CHECK_UNIT):
        record = clip.record
        assert record is not None
        targets = target_check_ids(record, checks)
        if not targets:
            outcomes.append(_unmatched(clip, record))
        units = {c.check_id: c.magnitude_unit for c in checks}
        outcomes += [_agreement(clip, record, c, units[c], settings) for c in targets]
    return outcomes


def _unmatched(clip: _Clip, record: PerturbationRecord) -> RuleOutcome:
    label = ",".join(record.target_checks)
    detail = f"no check matches {label} with unit {record.magnitude_unit}"
    return RuleOutcome(
        Rule.CHECK_UNIT,
        clip.name,
        label,
        (clip.clip.clip_id,),
        False,
        record.magnitude,
        None,
        detail,
    )


def _agreement(
    clip: _Clip, record: PerturbationRecord, check_id: str, unit: str, settings: SpecificitySettings
) -> RuleOutcome:
    value = target_value(record, check_id, clip.results)
    tolerance = max(settings.abs_tol, settings.rel_tol * abs(record.magnitude))
    if unit != record.magnitude_unit:
        passed, detail = False, f"unit {unit} != {record.magnitude_unit}"
    elif value is None:
        passed, detail = False, "the target check gave no magnitude"
    else:
        passed, detail = abs(value - record.magnitude) <= tolerance, f"tolerance {tolerance:.3g}"
    return RuleOutcome(
        Rule.CHECK_UNIT,
        clip.name,
        check_id,
        (clip.clip.clip_id,),
        passed,
        record.magnitude,
        value,
        detail,
    )


def _injected_outcomes(clips: Sequence[_Clip], checks: Sequence[CheckUnit]) -> list[RuleOutcome]:
    """Per scene, the check ranks the positive levels exactly like the injected parameter."""
    injected = [c for c in _perturbed(clips, MagnitudeKind.INJECTED_PARAMETER) if _injected(c) > 0]
    injected.sort(key=_scene_key)
    outcomes: list[RuleOutcome] = []
    for (name, _scene), group in groupby(injected, key=_scene_key):
        members = list(group)
        record = members[0].record
        assert record is not None
        outcomes += [
            _ranking(name, check_id, members) for check_id in target_check_ids(record, checks)
        ]
    return outcomes


def _ranking(name: str, check_id: str, members: Sequence[_Clip]) -> RuleOutcome:
    ids = tuple(m.clip.clip_id for m in members)
    pairs = [(_injected(m), target_value(_record(m), check_id, m.results)) for m in members]
    measured = [(level, value) for level, value in pairs if value is not None]
    if len(measured) < len(pairs):
        detail = "the target check gave no magnitude on some level"
        return RuleOutcome(
            Rule.INJECTED_PARAMETER, name, check_id, ids, False, PERFECT_ORDER, None, detail
        )
    tau = kendall(measured)
    detail = f"n_levels={len(measured)}"
    passed = math.isnan(tau) or tau == PERFECT_ORDER
    return RuleOutcome(
        Rule.INJECTED_PARAMETER, name, check_id, ids, passed, PERFECT_ORDER, tau, detail
    )


def kendall(pairs: Sequence[tuple[float, float]]) -> float:
    """Kendall tau between injected level and measured magnitude; NaN below two levels."""
    if len(pairs) < 2:
        return math.nan
    x, y = zip(*pairs, strict=True)
    return float(kendalltau(x, y).statistic)


def _band_outcomes(
    clips: Sequence[_Clip],
    checks: Sequence[CheckUnit],
    band: Mapping[str, float],
    settings: SpecificitySettings,
) -> list[RuleOutcome]:
    """Checks a perturbation must not move, plus supported levels of injected parameters."""
    couplings = {tuple(pair) for pair in settings.couplings}
    outcomes: list[RuleOutcome] = []
    for clip in clips:
        record = clip.record
        if record is None or clip.name == NONE:
            continue
        for check_id in _band_checks(record, checks):
            value = clip.magnitude(check_id)
            if value is None:
                continue
            rule = Rule.COUPLING if (clip.name, check_id) in couplings else Rule.NULL_BAND
            limit = band[check_id]
            ids = (clip.clip.clip_id,)
            outcomes.append(
                RuleOutcome(rule, clip.name, check_id, ids, value <= limit, limit, value)
            )
    return outcomes


def _band_checks(record: PerturbationRecord, checks: Sequence[CheckUnit]) -> list[str]:
    targets = set(target_check_ids(record, checks))
    supported = record.magnitude_kind is MagnitudeKind.INJECTED_PARAMETER and record.magnitude <= 0
    return [c.check_id for c in checks if c.check_id not in targets or supported]


def _perturbed(clips: Sequence[_Clip], kind: MagnitudeKind) -> list[_Clip]:
    return [
        clip
        for clip in clips
        if clip.record is not None and clip.name != NONE and clip.record.magnitude_kind is kind
    ]


def _record(clip: _Clip) -> PerturbationRecord:
    assert clip.record is not None
    return clip.record


def _injected(clip: _Clip) -> float:
    return _record(clip).magnitude


def _scene_key(clip: _Clip) -> tuple[str, str]:
    return clip.name, clip.clip.scene_id or ""


def _n_levels(clips: Sequence[_Clip]) -> dict[str, int]:
    levels: dict[str, set[int]] = defaultdict(set)
    for clip in clips:
        levels[clip.name].add(0 if clip.record is None else clip.record.level)
    return {name: len(found) for name, found in sorted(levels.items())}


def _check_ids(table: Sequence[TableRow]) -> list[str]:
    return list(dict.fromkeys(row.check_id for row in table))


def _check_units(table: Sequence[TableRow]) -> list[CheckUnit]:
    units = {row.check_id: row.unit for row in table}
    return [CheckUnit(check_id, units[check_id]) for check_id in _check_ids(table)]


def _clips(report: VerificationReport) -> list[_Clip]:
    rows: dict[str, dict[str, TableRow]] = defaultdict(dict)
    for row in report.table:
        rows[row.clip_id][row.check_id] = row
    results: dict[str, list[CheckResult]] = defaultdict(list)
    for result in report.results:
        results[result.clip_id].append(result)
    return [_Clip(clip, rows[clip.clip_id], tuple(results[clip.clip_id])) for clip in report.clips]


def write_sweep(run_dir: Path, cells: Sequence[MatrixCell], spec: SpecificityReport) -> None:
    columns = [field.name for field in fields(MatrixCell)]
    with (run_dir / MATRIX_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cell in cells:
            writer.writerow({k: "" if v is None else v for k, v in asdict(cell).items()})
    payload = {
        "passed": spec.passed,
        "n_failures": len(spec.failures),
        "null_band": dict(spec.null_band),
        "n_levels": dict(spec.n_levels),
        "outcomes": [asdict(outcome) for outcome in spec.outcomes],
    }
    text = json.dumps(encode(payload), indent=2, sort_keys=True, allow_nan=True)
    (run_dir / SPECIFICITY_FILE).write_text(text + "\n", encoding="utf-8")
