"""The stratified devscore figures of a real campaign reproduce from its out-of-fold values.

Fixture ``golden/devscore_oof_kv4_mf.json.gz``: the perceived level of ``devscore_oof.json`` for
arm Kv4 of the MegaSaM + FoundationPose chain on the 426 calib clips of ``kubric_synthetic``
(shards of tree 2f1e5e5, scored by devscore 1.2.0; no sealed split). Expected figures are the
ones its ``devscore.json`` reported.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from physverify.calibration.check_weights import Flag
from physverify.calibration.oof import AGGREGATE, DevScoreOof, LevelOof, units
from physverify.calibration.stratified import (
    Reading,
    UnitScores,
    family_map,
    meets_stratified,
    unit_scores,
)
from physverify.calibration.table import Role
from physverify.schema.checks import InputLevel
from physverify.synth.plan import family_of

FIXTURE = Path(__file__).parent / "golden" / "devscore_oof_kv4_mf.json.gz"
CLAUSE_MET = {"R4_jump", "R4_missing"}


def flags(level: LevelOof, unit: str) -> list[Flag]:
    """Positives first, as ``oof_flags`` orders them."""
    values = level.checks.get(unit, ())
    judged = [(v, v.flag) for v in values if v.flag is not None]
    found = [Flag(v.scene_id, v.role is Role.POSITIVE, flag) for v, flag in judged]
    return sorted(found, key=lambda flag: not flag.positive)


@pytest.fixture(scope="module")
def scores() -> dict[str, UnitScores]:
    oof = DevScoreOof.model_validate_json(gzip.decompress(FIXTURE.read_bytes()))
    level = oof.level(InputLevel.PERCEIVED)
    assert level is not None
    families = family_map([v.scene_id for v in level.aggregate], family_of)
    return {
        unit: unit_scores(list(values.values()), flags(level, unit), families, oof)
        for unit, values in units(level).items()
    }


def test_the_clip_score_auroc(scores: dict[str, UnitScores]) -> None:
    found = scores[AGGREGATE]
    scored = found.auroc_cv[Reading.SCORED]
    assert (scored.value, scored.ci.low, scored.ci.high) == pytest.approx(
        (0.6189, 0.5802, 0.6551), abs=1e-4
    )
    assert found.stratified is not None
    assert found.stratified.auroc_cv[Reading.SCORED].value == pytest.approx(0.6265, abs=1e-4)


def test_r4_jump_beats_its_family_floor(scores: dict[str, UnitScores]) -> None:
    found = scores["R4_jump"].stratified
    assert found is not None and found.j_cv is not None
    auroc = found.auroc_cv[Reading.SCORED]
    assert (auroc.value, auroc.ci.low, auroc.ci.high) == pytest.approx(
        (0.8283, 0.7505, 0.8966), abs=1e-4
    )
    assert found.family_floor[Reading.SCORED] == pytest.approx(0.7567, abs=1e-4)
    j = found.j_cv
    assert (j.informedness, j.informedness_ci.low, j.informedness_ci.high) == pytest.approx(
        (0.3670, 0.2260, 0.5228), abs=1e-4
    )


def test_r4_missing_stratified_informedness(scores: dict[str, UnitScores]) -> None:
    found = scores["R4_missing"].stratified
    assert found is not None and found.j_cv is not None
    j = found.j_cv
    assert (j.informedness, j.informedness_ci.low, j.informedness_ci.high) == pytest.approx(
        (0.7391, 0.6413, 0.8222), abs=1e-4
    )


def test_only_the_jump_and_missing_checks_meet_the_stratified_clause(
    scores: dict[str, UnitScores],
) -> None:
    met = {
        unit
        for unit, found in scores.items()
        if unit != AGGREGATE and found.stratified is not None and meets_stratified(found.stratified)
    }
    assert met == CLAUSE_MET
