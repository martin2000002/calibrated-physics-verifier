"""``calibration.stratified`` and ``calibration.paired``: family-stratified AUROC, floor and J."""

from __future__ import annotations

import numpy as np
import pytest
from calibration_cases import settings

from physverify.calibration.artifact import READINGS, Reading
from physverify.calibration.check_weights import Flag, raw_weight
from physverify.calibration.clause import meets_stratified
from physverify.calibration.paired import paired_deltas
from physverify.calibration.stats import auroc
from physverify.calibration.stratified import (
    Families,
    Valued,
    columns,
    family_floor,
    family_map,
    stratification,
    stratified_auroc,
    stratified_informedness,
    unit_scores,
)
from physverify.synth.plan import family_of, scene_id_of

SETTINGS = settings(bootstrap=200)
SCENES, CLIPS = 100, 4
SIGNAL = "a"


def everything(items: list[Valued]) -> np.ndarray:
    return np.arange(len(items))


def confounded(shift: float, seed: int = 0) -> tuple[list[Valued], list[Flag]]:
    """Positives only in family ``a`` (half its clips); score = family positive rate + noise,
    plus ``shift`` noise sds on the positives. Flags: a coin in ``a``, never elsewhere."""
    rng = np.random.default_rng(seed)
    items: list[Valued] = []
    flags: list[Flag] = []
    for family in (SIGNAL, "b", "c"):
        for s in range(SCENES):
            scene = scene_id_of(family, s)
            for k in range(CLIPS):
                positive = family == SIGNAL and k % 2 == 0
                rate = 0.5 if family == SIGNAL else 0.0
                noise = 0.1 * (rng.normal() + (shift if positive else 0.0))
                items.append(Valued(scene, rate + noise, positive))
                flags.append(Flag(scene, positive, family == SIGNAL and bool(rng.random() < 0.5)))
    return items, flags


def families(items: list[Valued]) -> Families:
    found = family_map((i.scene_id for i in items), family_of)
    assert found is not None
    return found


def test_a_family_only_score_is_high_globally_and_chance_within_family() -> None:
    items, flags = confounded(shift=0.0)
    scores = unit_scores(items, flags, families(items), SETTINGS)
    assert scores.stratified is not None
    global_auroc = scores.auroc_cv[Reading.SCORED].value
    floor = scores.stratified.family_floor[Reading.SCORED]
    assert global_auroc is not None and floor is not None
    assert global_auroc >= 0.85
    assert scores.stratified.auroc_cv[Reading.SCORED].value == pytest.approx(0.5, abs=0.05)
    assert floor == pytest.approx(global_auroc, abs=0.02)
    j = scores.stratified.j_cv
    assert j is not None and j.informedness_ci.low is not None and j.informedness_ci.high
    assert j.informedness_ci.low <= 0.0 <= j.informedness_ci.high
    assert raw_weight(flags, SETTINGS).value > 0.0  # globally the coin looks informative
    assert not meets_stratified(scores.stratified)


def test_a_within_family_shift_is_kept_by_the_stratified_auroc() -> None:
    items, flags = confounded(shift=2.0)
    scores = unit_scores(items, flags, families(items), SETTINGS)
    assert scores.stratified is not None
    strat = scores.stratified.auroc_cv[Reading.SCORED].value
    global_auroc = scores.auroc_cv[Reading.SCORED].value
    floor = scores.stratified.family_floor[Reading.SCORED]
    assert strat is not None and strat >= 0.9
    assert global_auroc is not None and floor is not None and global_auroc >= floor


def hand_case() -> list[Valued]:
    """x: AUROC 0.5 with 1 positive; y: AUROC 1 with 3; w: a positive without negatives (no
    AUROC, left out); z: negatives only."""
    rows = {
        "x": [(3.0, True), (1.0, False), (5.0, False)],
        "y": [(2.0, True), (4.0, True), (6.0, True), (1.0, False)],
        "w": [(0.0, True)],
        "z": [(9.0, False)],
    }
    return [
        Valued(scene_id_of(family, i), value, positive)
        for family, found in rows.items()
        for i, (value, positive) in enumerate(found)
    ]


def test_the_stratified_auroc_weights_families_by_positives() -> None:
    items = hand_case()
    c = columns(items, families(items))
    assert stratified_auroc(c, everything(items)) == pytest.approx((1 * 0.5 + 3 * 1.0) / 4)


def test_the_floor_is_the_auroc_of_the_family_rate() -> None:
    items = hand_case()
    rate = {"x": 1 / 3, "y": 3 / 4, "w": 1.0, "z": 0.0}
    score = [rate[str(family_of(i.scene_id))] for i in items]
    expected = auroc(
        [s for s, i in zip(score, items, strict=True) if i.positive],
        [s for s, i in zip(score, items, strict=True) if not i.positive],
    )
    c = columns(items, families(items))
    assert family_floor(c, everything(items)) == pytest.approx(expected)


def test_the_stratified_informedness_weights_families_by_positives() -> None:
    positive = np.array([True, False, False, True, True, True, False])
    flag = np.array([True, False, True, True, False, True, False])
    family = np.array([0, 0, 0, 1, 1, 1, 1])
    assert stratified_informedness(positive, flag, family) == pytest.approx(
        (1 * (1 - 0.5) + 3 * (2 / 3 - 0)) / 4
    )


def test_unscored_rows_read_zero_in_the_operational_reading() -> None:
    items = [
        *hand_case(),
        Valued(scene_id_of("x", 9), None, True),
        Valued(scene_id_of("z", 9), float("nan"), False),
    ]
    scores = unit_scores(items, (), families(items), SETTINGS)
    assert scores.auroc_cv[Reading.SCORED].n_positive == 5
    assert scores.auroc_cv[Reading.OPERATIONAL].n_positive == 6
    assert scores.auroc_cv[Reading.OPERATIONAL].n_negative == 5
    scored = [(i.value, i.positive) for i in items if i.value is not None and i.scored]
    assert scores.auroc_cv[Reading.SCORED].value == auroc(
        [v for v, p in scored if p], [v for v, p in scored if not p]
    )
    assert scores.stratified is not None and scores.stratified.j_cv is None


@pytest.mark.parametrize(
    "scenes",
    [
        ["s000", "s001"],
        [scene_id_of("a", 0), "s001"],
        [scene_id_of("a", 0), scene_id_of("a", 1)],
    ],
)
def test_without_two_families_on_every_scene_nothing_is_stratified(scenes: list[str]) -> None:
    assert family_map(scenes, family_of) is None
    found = stratification(scenes, family_of)
    assert not found.applies
    assert found.n_scenes_with_family == sum(family_of(s) is not None for s in scenes)
    items = [Valued(s, float(i), i % 2 == 0) for i, s in enumerate(scenes)]
    assert unit_scores(items, (), None, SETTINGS).stratified is None


def test_an_arm_against_itself_moves_nothing() -> None:
    items, _ = confounded(shift=1.0)
    arm = {f"c{i:04d}": item for i, item in enumerate(items)}
    unit = paired_deltas(arm, arm, families(items), SETTINGS)
    assert unit.stratified is not None
    for reading in READINGS:
        for found in (unit.auroc_cv[reading], unit.stratified[reading]):
            assert found.delta == 0.0
            assert (found.ci.low, found.ci.high) == (0.0, 0.0)
        assert not unit.improves[reading] and not unit.breaks[reading]


def test_a_paired_gain_inside_the_family_improves_both_deltas() -> None:
    (before, _), (after, _) = confounded(shift=0.0), confounded(shift=3.0)
    a = {f"c{i:04d}": item for i, item in enumerate(before)}
    b = {f"c{i:04d}": item for i, item in enumerate(after)}
    unit = paired_deltas(a, b, families(before), SETTINGS)
    assert unit.improves[Reading.SCORED] and unit.improves[Reading.OPERATIONAL]
    assert unit.auroc_cv[Reading.SCORED].n == len(before)
