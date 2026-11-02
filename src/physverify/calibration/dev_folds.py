"""The scene folds of the dev score: one partition for every check, both levels and the clip
score; each fold's model is fitted without its scenes. Pure, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest

from physverify.calibration.artifact import Stratum
from physverify.calibration.fit import LevelData, fit_stratum
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.stats.folds import grouped_folds
from physverify.verification.sweep import CheckUnit


@dataclass(frozen=True)
class FoldPart:
    train: LevelData
    held: LevelData
    stratum: Stratum
    """Fitted on ``train`` only."""


def scene_folds(
    levels: Sequence[LevelData], settings: CalibrationSettings
) -> tuple[frozenset[str], ...]:
    """Base scenes partitioned as without any extension; extension scenes on their own, joined
    fold by fold."""
    extension = frozenset().union(*(level.extension_scenes for level in levels))
    scenes = {row.scene_id for level in levels for row in level.rows}
    base = grouped_folds(sorted(scenes - extension), settings.folds, settings.seed)
    added = grouped_folds(sorted(scenes & extension), settings.folds, settings.seed)
    return tuple(a | b for a, b in zip_longest(base, added, fillvalue=frozenset[str]()))


def split_data(data: LevelData, fold: frozenset[str]) -> tuple[LevelData, LevelData]:
    """(train, held out): rows, results and sigma pairs follow the scene of their row; a result
    of a clip without a row is dropped, as ``clip_scores`` drops it."""
    scene = {row.clip_id: row.scene_id for row in data.rows}

    def part(held: bool) -> LevelData:
        pairs = data.sigma_pairs
        return replace(
            data,
            rows=tuple(r for r in data.rows if (r.scene_id in fold) is held),
            results=tuple(
                r for r in data.results if r.clip_id in scene and (scene[r.clip_id] in fold) is held
            ),
            sigma_pairs=None
            if pairs is None
            else tuple(p for p in pairs if (p.scene_id in fold) is held),
        )

    return part(False), part(True)


def fold_parts(
    data: LevelData,
    checks: Sequence[CheckUnit],
    profile: str,
    config: CalibrationConfig,
    folds: Sequence[frozenset[str]],
) -> tuple[FoldPart, ...]:
    found: list[FoldPart] = []
    for fold in folds:
        train, held = split_data(data, fold)
        found.append(FoldPart(train, held, fit_stratum(train, checks, profile, config)))
    return tuple(found)
