"""Family-stratified validity of a score, where every scene carries a family.

A score can rank the positives of one family above the negatives of another without separating
anything inside a family. Per family with positives: the AUROC of its positives against its own
negatives, averaged with the family's positive count as weight; the same for the informedness of
out-of-fold flags. The family floor is the AUROC of the in-sample family positive rate alone. Two
readings: the scored rows, and every labelled row with an unscored one read as 0.0. Pure, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from physverify.calibration.artifact import READINGS, Auroc, Reading, Stratified
from physverify.calibration.check_weights import Flag, weight
from physverify.calibration.stats import auroc
from physverify.core.models import ArtifactModel
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.stats.intervals import Ci

FamilyOf = Callable[[str], str | None]
Index = NDArray[np.int64]
UNSCORED_VALUE = 0.0
J_KEY = "j/stratified"


class Resampling(Protocol):
    """The scene bootstrap settings (``CalibrationSettings``, a ``DevScoreOof``)."""

    @property
    def bootstrap(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def ci(self) -> float: ...


class Stratification(ArtifactModel):
    """Applies when every scene carries a family and there are at least two families."""

    applies: bool
    families: tuple[str, ...]
    n_scenes: int
    n_scenes_with_family: int


@dataclass(frozen=True)
class Families:
    by_scene: Mapping[str, str]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.by_scene.values())))


@dataclass(frozen=True)
class Valued:
    """A labelled row; ``value`` None or not finite when it has nothing to rank."""

    scene_id: str
    value: float | None
    positive: bool

    @property
    def scored(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


@dataclass(frozen=True)
class UnitScores:
    auroc_cv: dict[Reading, Auroc]
    """Global, out of fold."""
    stratified: Stratified | None


def _named(scene_ids: Iterable[str], family_of: FamilyOf) -> tuple[int, dict[str, str]]:
    scenes = set(scene_ids)
    found = {scene: family_of(scene) for scene in scenes}
    return len(scenes), {scene: family for scene, family in found.items() if family is not None}


def _applies(n_scenes: int, named: Mapping[str, str]) -> bool:
    return len(named) == n_scenes and len(set(named.values())) > 1


def stratification(scene_ids: Iterable[str], family_of: FamilyOf) -> Stratification:
    n_scenes, named = _named(scene_ids, family_of)
    families = tuple(sorted(set(named.values())))
    return Stratification(
        applies=_applies(n_scenes, named),
        families=families,
        n_scenes=n_scenes,
        n_scenes_with_family=len(named),
    )


def family_map(scene_ids: Iterable[str], family_of: FamilyOf) -> Families | None:
    """None ("does not apply") when a scene has no family or there is a single family: a set
    with only some scenes in families is not stratified halfway."""
    n_scenes, named = _named(scene_ids, family_of)
    return Families(named) if _applies(n_scenes, named) else None


@dataclass(frozen=True)
class Columns:
    """Rows as arrays; ``family`` indexes ``Families.names`` (all 0 without families)."""

    value: NDArray[np.float64]
    scored: NDArray[np.bool_]
    positive: NDArray[np.bool_]
    family: Index
    scene: tuple[str, ...]

    def rows(self, reading: Reading, index: Index) -> Index:
        return index if reading is Reading.OPERATIONAL else index[self.scored[index]]


def columns(items: Sequence[Valued], families: Families | None) -> Columns:
    code = {} if families is None else {name: i for i, name in enumerate(families.names)}
    by_scene = {} if families is None else families.by_scene
    return Columns(
        value=np.array([i.value if i.scored else UNSCORED_VALUE for i in items], dtype=np.float64),
        scored=np.array([i.scored for i in items], dtype=bool),
        positive=np.array([i.positive for i in items], dtype=bool),
        family=np.array([code.get(by_scene.get(i.scene_id, ""), 0) for i in items], dtype=np.int64),
        scene=tuple(i.scene_id for i in items),
    )


def global_auroc(c: Columns, rows: Index) -> float:
    value, positive = c.value[rows], c.positive[rows]
    return auroc(value[positive], value[~positive])


def _weighted(values: Iterable[tuple[float, float]]) -> float:
    """Mean of the finite values weighted by their weight; NaN when none is finite."""
    kept = [(w, v) for w, v in values if math.isfinite(v)]
    total = sum(w for w, _ in kept)
    return sum(w * v for w, v in kept) / total if total else math.nan


def stratified_auroc(c: Columns, rows: Index) -> float:
    value, positive, family = c.value[rows], c.positive[rows], c.family[rows]
    return _weighted(
        (
            float(np.count_nonzero(positive & (family == g))),
            auroc(value[positive & (family == g)], value[~positive & (family == g)]),
        )
        for g in np.unique(family[positive])
    )


def family_floor(c: Columns, rows: Index) -> float:
    positive, family = c.positive[rows], c.family[rows]
    counts = np.bincount(family)
    rate = np.bincount(family, weights=positive) / np.maximum(counts, 1)
    score = rate[family]
    return auroc(score[positive], score[~positive])


def stratified_informedness(
    positive: NDArray[np.bool_], flag: NDArray[np.bool_], family: Index
) -> float:
    """TPR - FPR per family with positives and negatives, weighted by positives."""
    parts = []
    for g in np.unique(family[positive]):
        pos, neg = positive & (family == g), ~positive & (family == g)
        if neg.any():
            rate = float(flag[pos].mean()) - float(flag[neg].mean())
            parts.append((float(np.count_nonzero(pos)), rate))
    return _weighted(parts)


Metric = Callable[[Columns, Index], float]
GLOBAL, STRATIFIED = "global", "stratified"


def metrics(stratify: bool) -> dict[str, Metric]:
    found: dict[str, Metric] = {GLOBAL: global_auroc}
    return {**found, STRATIFIED: stratified_auroc} if stratify else found


def key(reading: Reading, metric: str) -> str:
    return f"{reading}/{metric}"


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def above(bound: float | None, floor: float) -> bool:
    return bound is not None and bound > floor


def indices(sample: Sequence[int]) -> Index:
    return np.fromiter(sample, dtype=np.int64, count=len(sample))


@dataclass(frozen=True)
class _Flags:
    positive: NDArray[np.bool_]
    flag: NDArray[np.bool_]
    family: Index


def _flag_columns(flags: Sequence[Flag], families: Families | None) -> _Flags:
    found = columns([Valued(f.scene_id, None, f.positive) for f in flags], families)
    return _Flags(found.positive, np.array([f.flag for f in flags], dtype=bool), found.family)


def _statistics(
    c: Columns, flags: _Flags, rows: Index, flag_rows: Index, stratify: bool
) -> dict[str, float]:
    found = {
        key(reading, name): metric(c, c.rows(reading, rows))
        for reading in READINGS
        for name, metric in metrics(stratify).items()
    }
    if stratify and flags.flag.size:
        p, f, g = flags.positive[flag_rows], flags.flag[flag_rows], flags.family[flag_rows]
        found[J_KEY] = stratified_informedness(p, f, g)
    return found


def _auroc(c: Columns, reading: Reading, value: float, ci: Ci) -> Auroc:
    kept = c.rows(reading, np.arange(len(c.scene)))
    n_positive = int(np.count_nonzero(c.positive[kept]))
    return Auroc(
        value=finite(value),
        ci=ci,
        n_positive=n_positive,
        n_negative=int(kept.size) - n_positive,
        in_sample=False,
    )


def unit_scores(
    items: Sequence[Valued],
    flags: Sequence[Flag],
    families: Families | None,
    settings: Resampling,
) -> UnitScores:
    """Global AUROC_cv in both readings and, with families, the stratified block; every interval
    from one scene bootstrap, so they share the drawn scenes. ``flags`` are ``oof_flags``'s."""
    c, f, n = columns(items, families), _flag_columns(flags, families), len(items)
    stratify = families is not None
    blocks = [ClusterBlock(scene, i) for i, scene in enumerate(c.scene)]
    blocks += [ClusterBlock(flag.scene_id, n + k) for k, flag in enumerate(flags)]

    def statistic(sample: Sequence[int]) -> dict[str, float]:
        index = indices(sample)
        return _statistics(c, f, index[index < n], index[index >= n] - n, stratify)

    cis = scene_bootstrap(blocks, statistic, settings.bootstrap, settings.seed, settings.ci)
    point = _statistics(c, f, np.arange(n), np.arange(len(flags)), stratify)

    def scored_as(metric: str) -> dict[Reading, Auroc]:
        return {
            r: _auroc(c, r, point[key(r, metric)], cis.get(key(r, metric), Ci())) for r in READINGS
        }

    if not stratify:
        return UnitScores(scored_as(GLOBAL), None)
    every = np.arange(n)
    stratified = Stratified(
        auroc_cv=scored_as(STRATIFIED),
        family_floor={r: finite(family_floor(c, c.rows(r, every))) for r in READINGS},
        j_cv=weight(point[J_KEY], cis.get(J_KEY, Ci()), flags) if J_KEY in point else None,
    )
    return UnitScores(scored_as(GLOBAL), stratified)
