"""The comparison's statistics over labelled clip rows: pair accuracy (PA) and its paired
difference, the clean-quantile cut u_B1, detection at a cut and the false-alarm rate.

Every score orders clips the same way: higher means more likely perturbed. A judge scores
plausibility, so its value enters negated.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from physverify.baselines.records import ClipRow
from physverify.baselines.settings import BootstrapSettings
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.stats.intervals import Ci, Rate, rate

Score = Callable[[ClipRow], float | None]
Flag = Callable[[ClipRow], bool]
TIE: float = 0.5


def b1_score(row: ClipRow) -> float | None:
    return row.b1


def judge_score(row: ClipRow) -> float | None:
    return None if row.judge is None else -row.judge


def b1_flag(cut: float) -> Flag:
    return lambda row: row.b1 is not None and row.b1 > cut


def judge_flag(cut: float) -> Flag:
    """The published reading: flagged when scored below ``cut``; an unread score flags nothing."""
    return lambda row: row.judge is not None and row.judge < cut


@dataclass(frozen=True)
class ScenePairs:
    scene: str
    clean: ClipRow
    perturbed: tuple[ClipRow, ...]


@dataclass(frozen=True)
class Estimate:
    value: float | None
    ci: Ci
    n: int


def scene_pairs(rows: Sequence[ClipRow]) -> tuple[ScenePairs, ...]:
    """The scenes with exactly one clean clip and at least one perturbed one, by scene."""
    by_scene: dict[str, list[ClipRow]] = {}
    for row in rows:
        by_scene.setdefault(row.scene, []).append(row)
    found = []
    for scene, members in sorted(by_scene.items()):
        clean = [r for r in members if not r.perturbed]
        perturbed = tuple(r for r in members if r.perturbed)
        if len(clean) == 1 and perturbed:
            found.append(ScenePairs(scene, clean[0], perturbed))
    return tuple(found)


def pair_outcome(clean: float | None, perturbed: float | None) -> float:
    """1 when the perturbed clip scores above the clean one, 0.5 on a tie or a missing score."""
    if clean is None or perturbed is None or clean == perturbed:
        return TIE
    return 1.0 if perturbed > clean else 0.0


def pair_accuracy(scenes: Sequence[ScenePairs], score: Score) -> float:
    outcomes = [pair_outcome(score(s.clean), score(p)) for s in scenes for p in s.perturbed]
    return sum(outcomes) / len(outcomes) if outcomes else math.nan


def pair_difference(scenes: Sequence[ScenePairs]) -> Mapping[str, float]:
    b1, judge = pair_accuracy(scenes, b1_score), pair_accuracy(scenes, judge_score)
    return {"b1": b1, "judge": judge, "delta": b1 - judge}


def paired_pairs(scenes: Sequence[ScenePairs], boot: BootstrapSettings) -> dict[str, Estimate]:
    """PA of B1, of the judge and their difference, each with its whole-scene interval."""
    point = pair_difference(scenes)
    blocks = [ClusterBlock(s.scene, s) for s in scenes]
    cis = scene_bootstrap(blocks, pair_difference, boot.n, boot.seed, boot.ci)
    n = sum(len(s.perturbed) for s in scenes)
    return {key: Estimate(_finite(value), cis.get(key, Ci()), n) for key, value in point.items()}


def clean_cut(values: Sequence[float], clean_fpr: float) -> float | None:
    """The smallest value u of ``values`` with frac(values > u) <= ``clean_fpr``."""
    ordered = sorted(values)
    for u in ordered:
        if (len(ordered) - bisect_right(ordered, u)) / len(ordered) <= clean_fpr:
            return u
    return None


def flagged_clean(rows: Sequence[ClipRow], flag: Flag, score: Score) -> tuple[int, int]:
    """The one false-alarm count: flagged clean clips, over the clean clips with a score."""
    scored = [r for r in rows if not r.perturbed and score(r) is not None]
    return sum(map(flag, scored)), len(scored)


def detection(rows: Sequence[ClipRow], flag: Flag, score: Score) -> Mapping[str, float]:
    """TPR over every perturbed clip (an unscored one is missed) minus the false-alarm rate."""
    positives = [r for r in rows if r.perturbed]
    alarms, clean = flagged_clean(rows, flag, score)
    if not positives or not clean:
        return {"value": math.nan}
    return {"value": sum(map(flag, positives)) / len(positives) - alarms / clean}


def scene_detection(
    rows: Sequence[ClipRow], flag: Flag, score: Score, boot: BootstrapSettings
) -> Estimate:
    blocks = [ClusterBlock(r.scene, r) for r in rows]
    cis = scene_bootstrap(
        blocks, lambda sample: detection(sample, flag, score), boot.n, boot.seed, boot.ci
    )
    point = detection(rows, flag, score)["value"]
    return Estimate(_finite(point), cis.get("value", Ci()), len(rows))


def false_alarms(rows: Sequence[ClipRow], flag: Flag, score: Score, ci: float) -> Rate:
    """``flagged_clean`` with its Clopper-Pearson interval."""
    return rate(*flagged_clean(rows, flag, score), ci)


def unread_share(rows: Sequence[ClipRow]) -> float:
    return sum(r.judge is None for r in rows) / len(rows) if rows else 1.0


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None
