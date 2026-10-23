"""Specificity sweep: each perturbation moves its own check and no other (ADR 0006 §3.9)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.schema.clip import ClipManifest
from physverify.schema.io import encode
from physverify.schema.scene import PerturbationRecord
from physverify.synth.perturbations.none import NAME as NONE

if TYPE_CHECKING:
    from physverify.verification.runner import TableRow, VerificationReport

MATRIX_FILE = "sweep_matrix.csv"
SPECIFICITY_FILE = "specificity.json"


class SpecificitySettings(FrozenModel):
    abs_tol: float = Field(default=1e-3, ge=0.0)
    rel_tol: float = Field(default=0.02, ge=0.0)
    null_tol: float = Field(default=1e-3, ge=0.0)
    couplings: list[tuple[str, str]] = Field(default_factory=list)
    """(perturbation, check) pairs that are expected to move together; reported, never failed.
    Declared in ``configs/verifier/checks.toml`` with their physical reason."""


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


PERTURBATION_UNIT = "perturbation_unit"
"""Optional class attribute of a check: the unit of the perturbations it is the target of, when it
is not its own (its magnitude is a test statistic of a parameter injected in that unit)."""


PERTURBATION_ESTIMATE = "perturbation_estimate"
"""Optional class attribute of such a check: the ``details`` key where each result carries its
estimate of the injected value, in the perturbation unit (labelled as an estimate)."""


def target_unit(check: CheckIdentity) -> str:
    """The perturbation unit ``check`` answers to: its ``PERTURBATION_UNIT`` or its own unit."""
    declared = getattr(check, PERTURBATION_UNIT, None)
    return str(declared) if declared else check.magnitude_unit


@dataclass(frozen=True)
class CheckUnit:
    check_id: str
    magnitude_unit: str
    perturbation_unit: str | None = None
    perturbation_estimate: str | None = None

    @classmethod
    def of(cls, check: CheckIdentity) -> CheckUnit:
        """Identity of a check class (or instance), its declared perturbation unit and estimate
        key included."""
        unit = getattr(check, PERTURBATION_UNIT, None)
        estimate = getattr(check, PERTURBATION_ESTIMATE, None)
        return cls(
            check.check_id,
            check.magnitude_unit,
            None if unit is None else str(unit),
            None if estimate is None else str(estimate),
        )


def units_of(registry: Mapping[str, CheckIdentity]) -> tuple[CheckUnit, ...]:
    """The identity of every registered check, in id order."""
    return tuple(CheckUnit.of(registry[c]) for c in sorted(registry))


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
    undefined: bool = False
    """The rule cannot be judged (a ranking of one level); neither a pass nor a failure."""


@dataclass(frozen=True)
class SpecificityReport:
    outcomes: tuple[RuleOutcome, ...]
    null_band: Mapping[str, float]
    n_levels: Mapping[str, int]

    @property
    def failures(self) -> tuple[RuleOutcome, ...]:
        return tuple(
            o
            for o in self.outcomes
            if not o.passed and not o.undefined and o.rule is not Rule.COUPLING
        )

    @property
    def passed(self) -> bool:
        return not self.failures


def perturbation_name(clip: ClipManifest) -> str:
    return NONE if clip.perturbation is None else clip.perturbation.name


def target_check_ids(
    record: PerturbationRecord, checks: Sequence[CheckIdentity]
) -> tuple[str, ...]:
    """A4 labels name a check exactly (R1) or a family resolved by ``target_unit`` (R4 ->
    R4_jump)."""
    found: list[str] = []
    for label in record.target_checks:
        exact = [c.check_id for c in checks if c.check_id == label]
        family = [
            c.check_id
            for c in checks
            if c.check_id.startswith(f"{label}_") and target_unit(c) == record.magnitude_unit
        ]
        found.extend(exact or family)
    return tuple(dict.fromkeys(found))


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
    order = {check_id: index for index, check_id in enumerate(check_ids(report.table))}
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


def check_ids(table: Sequence[TableRow]) -> list[str]:
    return list(dict.fromkeys(row.check_id for row in table))


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
