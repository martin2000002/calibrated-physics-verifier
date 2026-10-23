"""plan-congelado F4 (J7, J8, AC-4, AC-17): H1 on the held-out rows with the frozen artifact."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from calibration_cases import (
    CHECKS,
    config,
    dataset_ref,
    fit_meta,
    synthetic_level,
    synthetic_report,
)

from physverify.calibration.apply import policy_for, rule_for
from physverify.calibration.artifact import EvaluationReport
from physverify.calibration.artifact_io import Calibration, parse_calibration
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
    fit_settings,
)
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.heldout_monotonicity import heldout_h1, n_met
from physverify.calibration.monotonicity import LevelDelta, deltas, monotonicity
from physverify.calibration.report import evaluation_markdown
from physverify.calibration.scenes import SceneScope
from physverify.calibration.table import CLEAN_LEVEL, CalibrationRow, Role, corrected
from physverify.calibration.thresholds import fails
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.verification.policies.threshold import ThresholdParams
from physverify.verification.runner import VerificationReport

PRIMARY = ("s011-s039",)


def _joined(*reports: VerificationReport) -> VerificationReport:
    return VerificationReport(
        results=tuple(r for report in reports for r in report.results),
        table=tuple(t for report in reports for t in report.table),
        clips=tuple(c for report in reports for c in report.clips),
    )


@pytest.fixture(scope="module")
def frozen() -> Calibration:
    calib = synthetic_report(40, prefix="fam-s")
    level = synthetic_level(calib, InputLevel.ORACLE)
    artifact = fit_calibration(FitInputs((level,), CHECKS, fit_meta()), config())
    return parse_calibration(artifact.to_json())


def _rows(*reports: VerificationReport) -> tuple[CalibrationRow, ...]:
    return level_data(InputLevel.ORACLE, _joined(*reports), CHECKS, config(), split=None).rows


@pytest.fixture(scope="module")
def mixed() -> tuple[CalibrationRow, ...]:
    calib = synthetic_report(40, prefix="fam-s")
    held = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    return _rows(calib, held)


def _check_ids(calibration: Calibration) -> list[str]:
    return sorted(calibration.stratum(InputLevel.ORACLE, "fast").model.checks)


def _meta(primary: PrimaryScope | None = None) -> EvaluationMeta:
    return EvaluationMeta(
        "2026-10-21T00:00:00Z", "eval-h", "calibration.json", dataset_ref(), "fast", primary=primary
    )


def _evaluate(
    calibration: Calibration, rows_of: tuple[VerificationReport, ...], **update: object
) -> EvaluationReport:
    synthetic = level_data(InputLevel.ORACLE, _joined(*rows_of), CHECKS, config(), split=None)
    primary = PrimaryScope(SceneScope(primary_scenes=PRIMARY), dataset_ref())
    cfg = config(**update)
    return evaluate(calibration, (EvaluationInputs(synthetic, {}),), _meta(primary), cfg)


def test_only_calib_rows_give_no_heldout_h1(frozen: Calibration) -> None:
    """J8: ``evaluate --subset calib`` leaves the held-out rows empty."""
    stratum = frozen.stratum(InputLevel.ORACLE, "fast")
    calib_only = _rows(synthetic_report(10, prefix="fam-s"))
    settings = fit_settings(frozen, config())
    assert heldout_h1(stratum, calib_only, _check_ids(frozen), settings) == ()
    assert heldout_h1(stratum, (), _check_ids(frozen), settings) == ()
    report = _evaluate(frozen, (synthetic_report(40, prefix="fam-s"),))
    assert report.h1_primary is not None
    assert [s.heldout_h1 for s in (*report.strata, *report.h1_primary.strata)] == [(), ()]
    assert "Held-out H1" not in evaluation_markdown(report)


def test_heldout_h1_is_monotonicity_and_j_of_the_frozen_rule_by_hand(
    frozen: Calibration, mixed: tuple[CalibrationRow, ...]
) -> None:
    """AC-4: tau_b from the held-out deltas only, J = TPR - FPR of the frozen tau and sigma."""
    stratum = frozen.stratum(InputLevel.ORACLE, "fast")
    settings = fit_settings(frozen, config())
    found = {c.check_id: c for c in heldout_h1(stratum, mixed, _check_ids(frozen), settings)}
    assert set(found) == set(_check_ids(frozen))
    for check_id, result in found.items():
        held = [r for r in mixed if r.check_id == check_id and r.split is Split.EVAL]
        bias = stratum.sigma_correction(check_id)[0]
        clean = {r.scene_id: corrected(r, bias) for r in held if r.role is Role.CLEAN and r.scored}
        pairs = [
            LevelDelta(r.scene_id, r.pert_level, corrected(r, bias) - clean[r.scene_id])
            for r in held
            if r.role is Role.POSITIVE and r.scored and r.scene_id in clean
        ]
        pairs += [LevelDelta(s, CLEAN_LEVEL, 0.0) for s in {p.scene_id for p in pairs}]
        by_hand = monotonicity(sorted(pairs, key=lambda p: (p.scene_id, p.level)), settings)
        expected = by_hand.kendall_tau_b
        if expected is None or math.isnan(expected):
            assert result.monotonicity.kendall_tau_b is None
        else:
            assert result.monotonicity.kendall_tau_b == pytest.approx(expected)
        assert result.monotonicity.n_pairs == len(deltas(held, bias))
        tau = stratum.model.checks[check_id].tau
        rule = rule_for(stratum, check_id, policy_for(stratum))
        positives = [r for r in held if r.role is Role.POSITIVE and r.has_event]
        negatives = [r for r in held if r.role is Role.CLEAN and r.scored]
        assert (result.j.n_positive, result.j.n_negative) == (len(positives), len(negatives))
        if tau is not None and positives and negatives:
            tpr = np.mean([fails(r, rule, tau) for r in positives])
            fpr = np.mean([fails(r, rule, tau) for r in negatives])
            assert result.j.informedness == pytest.approx(tpr - fpr)
        low_t, low_j = result.monotonicity.ci.low, result.j.informedness_ci.low
        assert result.meets_h1 is (low_t is not None and low_t > 0.0)
        assert result.meets_j is (low_j is not None and low_j > 0.0)
        assert result.meets_both is (result.meets_h1 and result.meets_j)
    assert found["R1"].meets_both
    assert n_met(tuple(found.values())) >= 1


def test_calib_rows_never_enter(frozen: Calibration, mixed: tuple[CalibrationRow, ...]) -> None:
    stratum = frozen.stratum(InputLevel.ORACLE, "fast")
    settings = fit_settings(frozen, config())
    held_only = tuple(r for r in mixed if r.split is not Split.CALIB)
    ids = _check_ids(frozen)
    assert heldout_h1(stratum, mixed, ids, settings) == heldout_h1(
        stratum, held_only, ids, settings
    )


def test_no_parameter_comes_from_outside_the_artifact(frozen: Calibration) -> None:
    """AC-17: other evaluation settings (bootstrap, seed, level, policy) change nothing."""
    reports = (
        synthetic_report(40, prefix="fam-s"),
        synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s"),
    )
    shipped = _evaluate(frozen, reports)
    moved = _evaluate(frozen, reports, bootstrap=17, seed=5, ci=0.8)
    cfg = config(bootstrap=17, seed=5, ci=0.8)
    other_policy = type(cfg)(
        calibration=cfg.calibration,
        policy=ThresholdParams(k_sigma=7.0),
        couplings=cfg.couplings,
        min_span_iou=cfg.min_span_iou,
    )
    synthetic = level_data(InputLevel.ORACLE, _joined(*reports), CHECKS, config(), split=None)
    policy_moved = evaluate(frozen, (EvaluationInputs(synthetic, {}),), _meta(), other_policy)
    assert shipped.strata[0].heldout_h1
    assert moved.strata[0].heldout_h1 == shipped.strata[0].heldout_h1
    assert policy_moved.strata[0].heldout_h1 == shipped.strata[0].heldout_h1
    assert moved.strata[0].checks != shipped.strata[0].checks


def test_the_primary_view_has_its_own_heldout_h1(frozen: Calibration) -> None:
    reports = (
        synthetic_report(40, prefix="fam-s"),
        synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s"),
    )
    report = _evaluate(frozen, reports)
    assert report.h1_primary is not None
    (primary,) = report.h1_primary.strata
    (whole,) = report.strata
    p_r1 = next(c for c in primary.heldout_h1 if c.check_id == "R1")
    w_r1 = next(c for c in whole.heldout_h1 if c.check_id == "R1")
    assert w_r1.monotonicity.n_scenes == 40
    assert p_r1.monotonicity.n_scenes == 29
    markdown = evaluation_markdown(report)
    assert "Held-out H1 over the primary scenes (oracle)" in markdown
    assert "Held-out H1 over every scene (secondary" in markdown


def test_an_evaluation_without_heldout_h1_still_reads(frozen: Calibration) -> None:
    reports = (
        synthetic_report(20, prefix="fam-s"),
        synthetic_report(20, seed=7, split=Split.EVAL, prefix="held-s"),
    )
    data = json.loads(_evaluate(frozen, reports).to_json())
    data["version"] = "1.5.0"
    for stratum in (*data["strata"], *data["h1_primary"]["strata"]):
        del stratum["heldout_h1"]
    old = EvaluationReport.model_validate_json(json.dumps(data))
    assert old.strata[0].heldout_h1 == ()
    assert old.h1_primary is not None and old.h1_primary.strata[0].heldout_h1 == ()
