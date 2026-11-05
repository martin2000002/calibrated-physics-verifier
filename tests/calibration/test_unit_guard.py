"""Rows or sigma pairs of one check in two magnitude units never reach a threshold or sigma fit."""

from __future__ import annotations

from dataclasses import replace

import pytest
from calibration_cases import CHECKS, UNITS, config, extension_report, row, synthetic_level

from physverify.calibration.fit import fit_checks
from physverify.calibration.sigma_recalibration import SigmaPair, fit_sigma, require_pair_units
from physverify.calibration.table import CalibrationDataError, CalibrationRow, require_one_unit
from physverify.calibration.thresholds import cv_verdicts, fit_tau
from physverify.calibration.verdict import Rule
from physverify.schema.checks import InputLevel
from physverify.verification.policies.threshold import ThresholdPolicy

CONFIG = config(alpha=0.1, folds=3, bootstrap=30)
RULE = Rule(ThresholdPolicy(), bias=0.0, factor=1.0)
HOST_UNIT = "dE_over_E"


def clean(n: int, unit_of: dict[int, str] | None = None) -> list[CalibrationRow]:
    rows = [row(0.01 * i, scene=f"s{i:03d}", check_id="R3b") for i in range(n)]
    return [replace(r, unit=(unit_of or {}).get(i, r.unit)) for i, r in enumerate(rows)]


def test_one_unit_passes_and_an_empty_fit_has_nothing_to_mix() -> None:
    require_one_unit(clean(5))
    require_one_unit(clean(5), "u")
    require_one_unit([], "u")
    assert fit_tau(clean(20), 0.0, 0.1, 0.0).n == 20


def test_fit_tau_refuses_a_row_in_another_unit() -> None:
    with pytest.raises(CalibrationDataError, match=r"R3b rows mix magnitude units .*dE_over_E"):
        fit_tau(clean(20, {7: HOST_UNIT}), 0.0, 0.1, 0.0)


def test_every_fold_of_the_cross_validated_tau_is_guarded() -> None:
    with pytest.raises(CalibrationDataError, match="mix magnitude units"):
        cv_verdicts(clean(30, {3: HOST_UNIT}), RULE, CONFIG.calibration, 0.0)


def test_rows_in_one_unit_that_is_not_the_check_s_are_refused() -> None:
    with pytest.raises(CalibrationDataError, match=r"the check reports log_normal_speed_ratio"):
        require_one_unit(clean(5), "log_normal_speed_ratio")


def test_rows_spliced_in_another_implementation_s_unit_stop_the_fit() -> None:
    """The calib_f3_extra splice defect: some rows of a check kept in the unit of another
    implementation, fitted together with the recomputed ones."""
    data = synthetic_level(extension_report(), InputLevel.ORACLE)
    assert fit_checks(data, CHECKS, CONFIG)["R5b"].status == "ok"
    host = {r.clip_id for r in data.rows if r.scene_id in ("x003", "x011")}
    rows = tuple(
        replace(r, unit=HOST_UNIT) if r.check_id == "R5b" and r.clip_id in host else r
        for r in data.rows
    )
    with pytest.raises(CalibrationDataError, match=r"R5b rows mix magnitude units"):
        fit_checks(replace(data, rows=rows), CHECKS, CONFIG)


def sigma_pairs(n: int, unit_of: dict[int, str] | None = None) -> list[SigmaPair]:
    unit = UNITS["R1"]
    return [
        SigmaPair(f"s{i:03d}", "R1", 0.1, 0.1 + 0.01 * (i % 5 - 2), 0.01, unit, unit)
        if i not in (unit_of or {})
        else SigmaPair(f"s{i:03d}", "R1", 0.1, 0.1, 0.01, unit, (unit_of or {})[i])
        for i in range(n)
    ]


def test_sigma_pairs_in_one_unit_pass_and_none_have_nothing_to_mix() -> None:
    require_pair_units(sigma_pairs(5), UNITS["R1"])
    require_pair_units([], "u")
    assert fit_sigma(sigma_pairs(20), CONFIG.calibration, UNITS["R1"]).n_pairs == 20


def test_the_sigma_fit_refuses_a_pair_whose_side_is_in_another_unit() -> None:
    with pytest.raises(CalibrationDataError, match=r"R1 sigma pairs mix magnitude units"):
        fit_sigma(sigma_pairs(20, {4: HOST_UNIT}), CONFIG.calibration)


def test_sigma_pairs_in_one_unit_that_is_not_the_check_s_stop_the_fit() -> None:
    """The pairs' bias enters fit_tau, so the sigma fit is guarded before the rows are."""
    data = synthetic_level(extension_report(), InputLevel.PERCEIVED)
    pairs = tuple(
        replace(p, oracle_unit=HOST_UNIT, perceived_unit=HOST_UNIT) for p in sigma_pairs(20)
    )
    assert fit_checks(replace(data, sigma_pairs=tuple(sigma_pairs(20))), CHECKS, CONFIG)
    with pytest.raises(CalibrationDataError, match=r"the check reports pen_over_size"):
        fit_checks(replace(data, sigma_pairs=pairs), CHECKS, CONFIG)
