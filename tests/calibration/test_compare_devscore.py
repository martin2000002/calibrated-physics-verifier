"""``calibration.devscore_compare``: paired deltas b - a of two dev-score runs."""

from __future__ import annotations

import numpy as np
import pytest

from physverify.calibration.artifact import READINGS
from physverify.calibration.devscore_compare import compare_oof
from physverify.calibration.fit import ClipValue
from physverify.calibration.oof import AGGREGATE, CheckValue, DevScoreOof, LevelOof
from physverify.calibration.table import Role
from physverify.core.errors import UsageError
from physverify.schema.checks import InputLevel
from physverify.synth.plan import family_of, scene_id_of

SCENES, BOOTSTRAP = 30, 200


def run(run_id: str = "a", shift: float = 0.0, seed: int = 0, skip: int = 0) -> DevScoreOof:
    """Two families, a clean and a positive clip per scene; positives lifted by ``shift``."""
    rng = np.random.default_rng(seed)
    checks: list[CheckValue] = []
    clips: list[ClipValue] = []
    for s in range(skip, SCENES):
        scene = scene_id_of("fa" if s % 2 else "fb", s)
        for positive in (False, True):
            clip = f"{scene}__{'pert' if positive else 'none'}"
            value = float(rng.normal()) + (shift if positive else 0.0)
            role = Role.POSITIVE if positive else Role.CLEAN
            checks.append(
                CheckValue(clip_id=clip, scene_id=scene, role=role, value=value, flag=False)
            )
            clips.append(ClipValue(clip_id=clip, scene_id=scene, value=value, perturbed=positive))
    level = LevelOof(
        level=InputLevel.PERCEIVED, checks={"R1": tuple(checks)}, aggregate=tuple(clips)
    )
    return DevScoreOof(run_id=run_id, bootstrap=BOOTSTRAP, seed=0, ci=0.95, levels=(level,))


def test_a_run_against_itself_moves_nothing() -> None:
    found = compare_oof(run(), run("b"), InputLevel.PERCEIVED, family_of)
    assert set(found.units) == {"R1", AGGREGATE} and found.stratification.applies
    for unit in found.units.values():
        assert unit.stratified is not None
        for r in READINGS:
            for paired in (unit.auroc_cv[r], unit.stratified[r]):
                assert paired.delta == 0.0 and (paired.ci.low, paired.ci.high) == (0.0, 0.0)
            assert not unit.improves[r] and not unit.breaks[r]


def test_a_separating_arm_improves_on_a_random_one() -> None:
    found = compare_oof(run(), run("b", shift=4.0), InputLevel.PERCEIVED, family_of)
    assert all(found.units["R1"].improves.values())
    assert not any(found.units["R1"].breaks.values())


def test_runs_of_other_clips_are_refused() -> None:
    with pytest.raises(UsageError, match="different labelled clips"):
        compare_oof(run(), run("b", skip=1), InputLevel.PERCEIVED, family_of)


def test_runs_of_other_bootstrap_settings_are_refused() -> None:
    other = run("b").model_copy(update={"seed": 1})
    with pytest.raises(UsageError, match="bootstrap"):
        compare_oof(run(), other, InputLevel.PERCEIVED, family_of)


def test_a_missing_level_is_refused() -> None:
    with pytest.raises(UsageError, match="no oracle level"):
        compare_oof(run(), run("b"), InputLevel.ORACLE, family_of)


def test_a_clip_that_changes_scene_is_refused() -> None:
    a = run()
    level = a.levels[0]
    moved = level.aggregate[0].model_copy(update={"scene_id": "elsewhere"})
    b = a.model_copy(
        update={"levels": (level.model_copy(update={"aggregate": (moved, *level.aggregate[1:])}),)}
    )
    with pytest.raises(UsageError, match="change scene"):
        compare_oof(a, b, InputLevel.PERCEIVED, family_of)
