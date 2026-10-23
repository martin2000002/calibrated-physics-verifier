"""Row roles over hand-made manifests, the calib-only rule and the time step."""

from __future__ import annotations

from dataclasses import replace

import pytest
from calibration_cases import CHECKS, clip, config, record, synthetic_report

from physverify.calibration.table import (
    CalibrationDataError,
    Role,
    build_rows,
    role_of,
    unique_row_step,
)
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.scene import MagnitudeKind
from physverify.synth.perturbations.none import NAME as NONE
from physverify.verification.runner import RowStatus, VerificationReport
from physverify.verification.sweep import CheckUnit

R4 = (
    CheckUnit("R4_jump", "jump_over_size"),
    CheckUnit("R4_size", "rel_size_change"),
    CheckUnit("R2", "residual_over_weight"),
    CheckUnit("R5b", "g_rel_error"),
)
COUPLINGS = config().couplings


def role(perturbation_name: str, check_id: str, **kw: object) -> Role:
    pert = record(perturbation_name, **kw)  # type: ignore[arg-type]
    return role_of(clip("c", "s", perturbation=pert), check_id, R4, COUPLINGS)


def test_a_clean_clip_is_clean_with_or_without_a_none_record() -> None:
    assert role_of(clip("c"), "R4_jump", R4, COUPLINGS) is Role.CLEAN
    none = record(NONE, (), 0.0, "none", level=0)
    assert role_of(clip("c", perturbation=none), "R4_jump", R4, COUPLINGS) is Role.CLEAN


def test_a_family_label_targets_the_check_with_the_same_unit() -> None:
    kw = {"targets": ("R4",), "magnitude": 0.3, "unit": "jump_over_size"}
    assert role("r4_teleport", "R4_jump", **kw) is Role.POSITIVE
    assert role("r4_teleport", "R4_size", **kw) is Role.OFF_TARGET


def test_a_declared_coupling_is_excluded() -> None:
    kw = {"targets": ("R5a",), "magnitude": 0.3, "unit": "accel_rel_diff"}
    assert role("r5a_gravity_switch", "R5b", **kw) is Role.COUPLED
    assert role("r5a_gravity_switch", "R4_jump", **kw) is Role.OFF_TARGET


def test_r2_with_a_non_positive_injected_overhang_is_not_a_violation() -> None:
    kind = MagnitudeKind.INJECTED_PARAMETER
    kw = {"targets": ("R2",), "unit": "overhang", "kind": kind}
    assert role("r2_overhang", "R2", magnitude=-0.1, **kw) is Role.OFF_TARGET
    assert role("r2_overhang", "R2", magnitude=0.2, **kw) is Role.POSITIVE


def test_rows_carry_role_level_and_injected_magnitude() -> None:
    report = synthetic_report(2)
    rows = build_rows(report, CHECKS, COUPLINGS, InputLevel.ORACLE, split=Split.CALIB)
    assert len(rows) == len(report.table)
    positive = next(
        r for r in rows if r.clip_id == "s000__r1_penetration_L2" and r.check_id == "R1"
    )
    assert (positive.role, positive.pert_level, positive.mu) == (Role.POSITIVE, 2, 0.1)
    no_event = next(r for r in rows if r.check_id == "R5b" and r.role is Role.CLEAN)
    assert no_event.status is RowStatus.NO_EVENT
    assert not no_event.has_event and not no_event.scored
    off = next(r for r in rows if r.clip_id == "s000__r5b_gravity_scale_L1" and r.check_id == "R1")
    assert (off.role, off.mu) == (Role.OFF_TARGET, None)


def test_an_eval_clip_cannot_be_fitted_on() -> None:
    report = synthetic_report(2)
    leaked = replace(report.clips[0], split=Split.EVAL)
    bad = VerificationReport(report.results, report.table, (leaked, *report.clips[1:]))
    with pytest.raises(CalibrationDataError, match="split=calib"):
        build_rows(bad, CHECKS, COUPLINGS, InputLevel.ORACLE, split=Split.CALIB)
    assert build_rows(bad, CHECKS, COUPLINGS, InputLevel.ORACLE)


def test_rows_must_share_one_time_step() -> None:
    rows = build_rows(synthetic_report(1), CHECKS, COUPLINGS, InputLevel.ORACLE)
    assert unique_row_step(rows) == 1
    mixed = (*rows, replace(rows[0], row_step=2))
    with pytest.raises(CalibrationDataError, match="row_step"):
        unique_row_step(mixed)
    unknown = (*rows, replace(rows[0], row_step=0))
    assert unique_row_step(unknown) == 1
