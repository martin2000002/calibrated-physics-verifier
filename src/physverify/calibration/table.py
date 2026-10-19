"""One calibration row per (clip, check) with its role (ADR 0009 D3, Juicio 1)."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum

from physverify.core.errors import PhysverifyError
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.synth.perturbations.none import NAME as NONE
from physverify.verification.runner import UNKNOWN_ROW_STEP, RowStatus, VerificationReport
from physverify.verification.sweep import CheckIdentity, perturbation_name, target_check_ids

CLEAN_LEVEL = 0


class CalibrationDataError(PhysverifyError):
    """The rows cannot be calibrated as asked (wrong split, mixed time steps)."""


class Role(StrEnum):
    POSITIVE = "positive"
    CLEAN = "clean"
    OFF_TARGET = "off_target"
    COUPLED = "coupled"


@dataclass(frozen=True)
class CalibrationRow:
    """``magnitude`` is the clip's max over subjects (what tau is fitted on); ``subjects`` the
    per-subject results the clip verdict judges; ``mu`` the injected magnitude (A4)."""

    clip_id: str
    scene_id: str
    split: Split
    check_id: str
    unit: str
    level: InputLevel
    status: RowStatus
    magnitude: float | None
    sigma: float
    role: Role
    perturbation: str
    pert_level: int
    mu: float | None
    mu_unit: str
    row_step: int
    subjects: tuple[CheckResult, ...] = ()

    @property
    def has_event(self) -> bool:
        return self.status in (RowStatus.SCORED, RowStatus.ABSTAINED)

    @property
    def scored(self) -> bool:
        return self.status is RowStatus.SCORED and self.magnitude is not None


def scene_of(clip: ClipManifest) -> str:
    return clip.scene_id or clip.clip_id


def role_of(
    clip: ClipManifest,
    check_id: str,
    checks: Sequence[CheckIdentity],
    couplings: Collection[tuple[str, str]],
) -> Role:
    record = clip.perturbation
    if record is None or record.name == NONE:
        return Role.CLEAN
    if check_id in target_check_ids(record, checks):
        return Role.POSITIVE if _injected(record) else Role.OFF_TARGET
    if (record.name, check_id) in couplings:
        return Role.COUPLED
    return Role.OFF_TARGET


def _injected(record: PerturbationRecord) -> bool:
    """R2 (``injected_parameter``) with an overhang <= 0 is still supported: not a violation."""
    return record.magnitude_kind is not MagnitudeKind.INJECTED_PARAMETER or record.magnitude > 0.0


def build_rows(
    report: VerificationReport,
    checks: Sequence[CheckIdentity],
    couplings: Collection[tuple[str, str]],
    level: InputLevel,
    *,
    split: Split | None = None,
) -> tuple[CalibrationRow, ...]:
    """Rows of ``report``; with ``split`` every clip must belong to it (D12: error, not warning)."""
    clips = {clip.clip_id: clip for clip in report.clips}
    if split is not None:
        _require_split(clips.values(), split)
    subjects = results_by_row(report.results)
    rows: list[CalibrationRow] = []
    for row in report.table:
        clip = clips[row.clip_id]
        role = role_of(clip, row.check_id, checks, couplings)
        record = clip.perturbation
        rows.append(
            CalibrationRow(
                clip_id=row.clip_id,
                scene_id=scene_of(clip),
                split=clip.split,
                check_id=row.check_id,
                unit=row.unit,
                level=level,
                status=row.status,
                magnitude=row.magnitude,
                sigma=row.sigma,
                role=role,
                perturbation=perturbation_name(clip),
                pert_level=CLEAN_LEVEL if record is None else record.level,
                mu=record.magnitude if record is not None and role is Role.POSITIVE else None,
                mu_unit="" if record is None else record.magnitude_unit,
                row_step=row.row_step,
                subjects=tuple(subjects.get((row.clip_id, row.check_id), ())),
            )
        )
    return tuple(rows)


def results_by_row(results: Sequence[CheckResult]) -> dict[tuple[str, str], list[CheckResult]]:
    """(clip_id, check_id) -> its per-subject results."""
    found: dict[tuple[str, str], list[CheckResult]] = {}
    for result in results:
        found.setdefault((result.clip_id, result.check_id), []).append(result)
    return found


def _require_split(clips: Collection[ClipManifest], split: Split) -> None:
    wrong = sorted(clip.clip_id for clip in clips if clip.split is not split)
    if wrong:
        raise CalibrationDataError(
            f"{len(wrong)} clips are not split={split.value} and cannot be fitted on: {wrong[:5]}"
        )


def unique_row_step(rows: Sequence[CalibrationRow]) -> int:
    """D4: one time step per stratum (ADR 0006 R9); clips without a time axis are ignored."""
    steps = {row.row_step for row in rows if row.row_step != UNKNOWN_ROW_STEP}
    if len(steps) > 1:
        raise CalibrationDataError(f"rows mix several row_step values {sorted(steps)}")
    return steps.pop() if steps else UNKNOWN_ROW_STEP


def require_one_clean_per_scene(rows: Sequence[CalibrationRow]) -> None:
    """D5/D8 take one clean clip per scene (exchangeable scenes); two would double-count one."""
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.role is not Role.CLEAN:
            continue
        key = (row.check_id, row.scene_id)
        other = seen.setdefault(key, row.clip_id)
        if other != row.clip_id:
            raise CalibrationDataError(
                f"scene {row.scene_id} has two clean clips for {row.check_id}: "
                f"{other} and {row.clip_id}"
            )


def corrected(row: CalibrationRow, bias: float) -> float:
    """m - bias of a scored row, NaN otherwise."""
    return math.nan if row.magnitude is None else row.magnitude - bias
