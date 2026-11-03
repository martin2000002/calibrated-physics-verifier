"""ADR 0036 clause (ii): the held-out n_met and the dev score count a check by the same rule."""

from __future__ import annotations

import pytest
from calibration_cases import CHECKS, config, fit_meta, settings, synthetic_level, synthetic_report

from physverify.calibration.artifact import CheckCalibration
from physverify.calibration.artifact_io import Calibration, parse_calibration
from physverify.calibration.check_weights import Flag
from physverify.calibration.collect import level_data
from physverify.calibration.dev_score import check_score
from physverify.calibration.evaluate import fit_settings
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.heldout_monotonicity import heldout_h1, judged, n_met
from physverify.calibration.stratified import Families, Valued, family_map, unit_scores
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.synth.plan import family_of, scene_id_of
from physverify.verification.runner import VerificationReport

SETTINGS = settings()
SCENES = 30


@pytest.fixture(scope="module")
def frozen() -> Calibration:
    level = synthetic_level(synthetic_report(40, prefix="fam-s"), InputLevel.ORACLE)
    artifact = fit_calibration(FitInputs((level,), CHECKS, fit_meta()), config())
    return parse_calibration(artifact.to_json())


@pytest.fixture(scope="module")
def r1(frozen: Calibration) -> CheckCalibration:
    found = frozen.stratum(InputLevel.ORACLE, "fast").model.checks["R1"]
    assert found.monotonicity.h1_pass
    return found


def _toy(f2_positives: bool) -> tuple[list[Flag], list[Valued], Families]:
    """Family f1: every positive and every clean flagged; family f2: no clean flagged and, when
    ``f2_positives``, every positive flagged. Global J = 1 - 1/2 in both."""
    flags: list[Flag] = []
    valued: list[Valued] = []
    for i in range(SCENES):
        a, b = scene_id_of("f1", i), scene_id_of("f2", i)
        flags += [Flag(a, True, True), Flag(a, False, True), Flag(b, False, False)]
        valued += [Valued(a, 2.0, True), Valued(a, 1.0, False), Valued(b, 0.0, False)]
        if f2_positives:
            flags.append(Flag(b, True, True))
            valued.append(Valued(b, 2.0, True))
    families = family_map((v.scene_id for v in valued), family_of)
    assert families is not None
    return flags, valued, families


def test_a_check_that_only_separates_families_is_not_met(r1: CheckCalibration) -> None:
    """The reviewer's case: global J 0.5 and stratified J 0.0 meet the old rule, not the clause."""
    flags, valued, families = _toy(f2_positives=False)
    found = judged("R1", r1.monotonicity, flags, valued, families, SETTINGS)
    assert found.j.informedness == pytest.approx(0.5)
    assert found.stratified is not None and found.stratified.j_cv is not None
    assert found.stratified.j_cv.informedness == pytest.approx(0.0)
    assert (found.meets_h1, found.meets_j, found.meets_both) == (True, True, True)
    assert found.meets_stratified is False
    assert found.meets_clause is False
    assert n_met((found,)) == 0


def test_a_check_that_separates_within_families_is_met(r1: CheckCalibration) -> None:
    flags, valued, families = _toy(f2_positives=True)
    found = judged("R1", r1.monotonicity, flags, valued, families, SETTINGS)
    assert found.stratified is not None and found.stratified.j_cv is not None
    assert found.stratified.j_cv.informedness == pytest.approx(0.5)
    assert (found.meets_both, found.meets_stratified, found.meets_clause) == (True, True, True)
    assert n_met((found,)) == 1


def _joined(*reports: VerificationReport) -> VerificationReport:
    return VerificationReport(
        results=tuple(r for report in reports for r in report.results),
        table=tuple(t for report in reports for t in report.table),
        clips=tuple(c for report in reports for c in report.clips),
    )


def test_heldout_scenes_with_families_get_the_stratified_block(frozen: Calibration) -> None:
    """Held-out scenes of two families: the evaluation computes the block and counts by it."""
    held = [
        synthetic_report(20, seed=seed, split=Split.EVAL, prefix=f"{family}-s")
        for seed, family in ((7, "f1"), (8, "f2"))
    ]
    rows = level_data(InputLevel.ORACLE, _joined(*held), CHECKS, config(), split=None).rows
    stratum = frozen.stratum(InputLevel.ORACLE, "fast")
    ids = sorted(stratum.model.checks)
    found = heldout_h1(stratum, rows, ids, fit_settings(frozen, config()), family_of)
    by_id = {c.check_id: c for c in found}
    assert by_id["R1"].stratified is not None
    assert by_id["R1"].meets_clause is True
    assert all(c.meets_clause is (c.meets_both and c.meets_stratified is not False) for c in found)
    assert n_met(found) == sum(bool(c.meets_clause) for c in found)


@pytest.mark.parametrize("f2_positives", [False, True])
def test_the_dev_score_and_the_heldout_evaluation_agree(
    r1: CheckCalibration, f2_positives: bool
) -> None:
    """Same monotonicity, verdicts and magnitudes: the same J, block and verdict on both paths."""
    flags, valued, families = _toy(f2_positives)
    unit = unit_scores(valued, flags, families, SETTINGS)
    dev = check_score("R1", r1, (), flags, SETTINGS, unit)
    held = judged("R1", r1.monotonicity, flags, valued, families, SETTINGS)
    assert (held.j, held.stratified, held.auroc_readings) == (
        dev.j_cv,
        dev.stratified,
        dev.auroc_cv_readings,
    )
    verdict = ("meets_h1", "meets_j", "meets_both", "meets_stratified", "meets_clause")
    assert [getattr(held, k) for k in verdict] == [getattr(dev, k) for k in verdict]
    assert held.counted is dev.meets_clause is f2_positives
