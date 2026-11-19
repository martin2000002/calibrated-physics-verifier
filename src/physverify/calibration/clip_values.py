"""A clip's value from its check results (1 - S, or an injected value such as B1) and its label,
for the aggregate AUROC and the out-of-fold rows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from physverify.calibration.apply import score_clip
from physverify.calibration.artifact import Stratum
from physverify.calibration.artifact_io import Stratum as StratumView
from physverify.calibration.table import CalibrationRow, Role
from physverify.core.models import ArtifactModel
from physverify.schema.checks import CheckResult

if TYPE_CHECKING:
    from physverify.calibration.fit import LevelData


ClipValueOf = Callable[[Sequence[CheckResult]], float | None]
"""A clip's ordering value from its results (higher: more likely perturbed); None: no value."""


class ClipValue(ArtifactModel):
    """``value`` = 1 - S of a clip with a label (or the injected value); None when the clip has
    no value."""

    clip_id: str
    scene_id: str
    value: float | None
    perturbed: bool


def clip_values(
    model: Stratum, data: LevelData, value_of: ClipValueOf | None = None
) -> list[ClipValue]:
    """Every labelled clip of ``data`` outside the extension scenes, by clip id; the value is
    1 - S unless ``value_of`` is given."""
    value_of = value_of or _one_minus_score(model)
    by_clip: dict[str, list[CheckResult]] = {}
    for result in data.results:
        by_clip.setdefault(result.clip_id, []).append(result)
    found: list[ClipValue] = []
    for clip_id, (scene, perturbed) in sorted(clip_info(data.rows).items()):
        if perturbed is None or scene in data.extension_scenes:
            continue
        results = by_clip.get(clip_id)
        value = value_of(results) if results else None
        found.append(ClipValue(clip_id=clip_id, scene_id=scene, value=value, perturbed=perturbed))
    return found


def _one_minus_score(model: Stratum) -> ClipValueOf:
    view = StratumView(model=model, source="fit", artifact=None)

    def value(results: Sequence[CheckResult]) -> float | None:
        score = score_clip(results, view).score
        return None if score is None else 1.0 - score

    return value


def clip_scores(
    model: Stratum, data: LevelData, value_of: ClipValueOf | None = None
) -> list[tuple[str, float, bool]]:
    """(scene, value, perturbed?) of the clips of ``clip_values`` with a value."""
    return [
        (c.scene_id, c.value, c.perturbed)
        for c in clip_values(model, data, value_of)
        if c.value is not None
    ]


def clip_info(rows: Sequence[CalibrationRow]) -> dict[str, tuple[str, bool | None]]:
    """clip -> (scene, perturbed?); None for a clip whose perturbation targets no check."""
    info: dict[str, tuple[str, bool | None]] = {}
    for row in rows:
        scene, previous = info.get(row.clip_id, (row.scene_id, None))
        if row.role is Role.CLEAN:
            info[row.clip_id] = (scene, False)
        elif row.role is Role.POSITIVE or previous:
            info[row.clip_id] = (scene, True)
        else:
            info[row.clip_id] = (scene, previous)
    return info
