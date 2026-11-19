"""Pair accuracy, its paired difference, u_B1, detection and false alarms over clip rows."""

from __future__ import annotations

import pytest

from physverify.baselines.records import ClipRow
from physverify.baselines.settings import BootstrapSettings
from physverify.baselines.stats import (
    TIE,
    b1_flag,
    b1_score,
    clean_cut,
    detection,
    false_alarms,
    flagged_clean,
    judge_flag,
    judge_score,
    pair_accuracy,
    pair_outcome,
    paired_pairs,
    scene_pairs,
    unread_share,
)
from physverify.stats.intervals import clopper_pearson

BOOT = BootstrapSettings(n=200, seed=0, ci=0.95)


def row(clip: str, scene: str, perturbed: bool, b1: float | None, judge: float | None) -> ClipRow:
    return ClipRow(clip_id=clip, scene=scene, perturbed=perturbed, b1=b1, judge=judge)


def scene(name: str, clean: tuple[float, float], *perturbed: tuple[float, float]) -> list[ClipRow]:
    rows = [row(f"{name}_c", name, False, *clean)]
    return rows + [row(f"{name}_p{i}", name, True, *p) for i, p in enumerate(perturbed)]


def test_a_scene_pairs_its_one_clean_clip_with_each_perturbed_one() -> None:
    rows = [
        *scene("a", (0.1, 5.0), (0.9, 1.0), (0.8, 2.0)),
        row("b_c", "b", False, 0.1, 5.0),
        *scene("c", (0.1, 5.0)),
        row("d_c1", "d", False, 0.1, 5.0),
        *scene("d", (0.1, 5.0), (0.9, 1.0)),
    ]
    (found,) = scene_pairs(rows)
    assert found.scene == "a" and found.clean.clip_id == "a_c" and len(found.perturbed) == 2


@pytest.mark.parametrize(
    ("clean", "perturbed", "expected"),
    [(0.1, 0.9, 1.0), (0.9, 0.1, 0.0), (0.5, 0.5, TIE), (None, 0.9, TIE), (0.1, None, TIE)],
)
def test_a_pair_counts_a_tie_or_a_missing_score_as_half(
    clean: float | None, perturbed: float | None, expected: float
) -> None:
    assert pair_outcome(clean, perturbed) == expected


def test_the_judge_enters_negated_so_a_lower_plausibility_is_a_hit() -> None:
    scenes = scene_pairs(scene("a", (0.1, 5.0), (0.2, 1.0)))
    assert pair_accuracy(scenes, judge_score) == 1.0
    assert pair_accuracy(scenes, b1_score) == 1.0


def test_the_difference_is_b1_minus_the_judge() -> None:
    rows = [
        *scene("a", (0.1, 1.0), (0.9, 5.0)),
        *scene("b", (0.1, 1.0), (0.9, 5.0)),
        *scene("c", (0.1, 1.0), (0.9, 1.0)),
    ]
    found = paired_pairs(scene_pairs(rows), BOOT)
    assert found["b1"].value == 1.0
    assert found["judge"].value == pytest.approx(0.5 / 3)
    assert found["delta"].value == pytest.approx(1.0 - 0.5 / 3) and found["delta"].n == 3
    assert found["delta"].ci.low is not None and found["delta"].ci.low > 0


def test_u_b1_is_the_smallest_value_above_which_at_most_the_target_share_lies() -> None:
    values = [i / 10 for i in range(1, 11)]
    assert clean_cut(values, 0.10) == 0.9
    assert clean_cut(values, 0.0) == 1.0
    assert clean_cut([0.5, 0.5, 0.5, 0.7], 0.10) == 0.7
    assert clean_cut([0.5, 0.5, 0.5, 0.7], 0.25) == 0.5
    assert clean_cut([], 0.10) is None


def test_detection_is_tpr_minus_the_false_alarm_rate() -> None:
    """An unscored perturbed clip is missed; an unscored clean one is out of the rate."""
    rows = [
        row("c1", "c1", False, 0.1, 5.0),
        row("c2", "c2", False, 0.9, 1.0),
        row("c3", "c3", False, 0.1, None),
        row("p1", "p1", True, 0.9, 1.0),
        row("p2", "p2", True, 0.8, None),
    ]
    assert detection(rows, b1_flag(0.5), b1_score)["value"] == pytest.approx(1.0 - 1 / 3)
    assert detection(rows, judge_flag(4.0), judge_score)["value"] == pytest.approx(0.5 - 0.5)
    for flag, score in ((b1_flag(0.5), b1_score), (judge_flag(4.0), judge_score)):
        alarms = false_alarms(rows, flag, score, 0.95)
        assert flagged_clean(rows, flag, score) == (alarms.k, alarms.n)


def test_false_alarms_count_flagged_clean_clips_with_a_score_clopper_pearson() -> None:
    rows = [row(f"c{i}", f"c{i}", False, 0.9 if i == 0 else 0.1, None) for i in range(10)]
    rows.append(row("x", "x", False, None, None))
    found = false_alarms(rows, b1_flag(0.5), b1_score, 0.95)
    assert (found.k, found.n, found.value) == (1, 10, 0.1)
    assert found.ci == clopper_pearson(1, 10, 0.95)
    assert false_alarms(rows, judge_flag(4.0), judge_score, 0.95).n == 0


def test_the_unread_share_counts_clips_without_a_judge_score() -> None:
    rows = [row("a", "a", False, 0.1, 5.0), row("b", "b", True, 0.1, None)]
    assert unread_share(rows) == 0.5
