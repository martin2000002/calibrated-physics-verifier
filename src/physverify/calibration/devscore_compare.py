"""Paired comparison of two devscore runs from their out-of-fold values (ADR 0036 clause (i)).

Both runs must hold the same labelled clips with the same scenes and bootstrap settings; every
unit (check and clip score) present in both is compared. Pure, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from physverify.calibration.oof import DevScoreOof, LevelOof, units
from physverify.calibration.paired import PairedUnit, paired_deltas
from physverify.calibration.stratified import FamilyOf, Stratification, Valued, family_map
from physverify.calibration.stratified import stratification as stratification_of
from physverify.core.errors import UsageError
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.schema.checks import InputLevel

DEVSCORE_COMPARE_SCHEMA = "physverify.devscore_compare"
DEVSCORE_COMPARE_VERSION = "1.0.0"
DEVSCORE_COMPARE_FILE = "devscore_compare.json"


class DevScoreCompare(ArtifactModel):
    """``units[u]`` holds b - a for unit ``u`` (a check id or ``aggregate``)."""

    schema_: str = Field(default=DEVSCORE_COMPARE_SCHEMA, alias=SCHEMA_KEY)
    version: str = DEVSCORE_COMPARE_VERSION
    run_a: str
    run_b: str
    level: InputLevel
    bootstrap: int
    seed: int
    ci: float
    n_clips: int
    stratification: Stratification
    units: dict[str, PairedUnit]


def _level(oof: DevScoreOof, level: InputLevel) -> LevelOof:
    found = oof.level(level)
    if found is None:
        raise UsageError(f"devscore run {oof.run_id} has no {level.value} level")
    return found


def _scenes(values: Mapping[str, Valued]) -> dict[str, str]:
    return {clip: v.scene_id for clip, v in values.items()}


def _require_same_clips(unit: str, a: Mapping[str, Valued], b: Mapping[str, Valued]) -> None:
    if set(a) != set(b):
        raise UsageError(
            f"{unit}: the runs hold different labelled clips ({len(set(a) - set(b))} only in a, "
            f"{len(set(b) - set(a))} only in b): pair runs of one calib selection"
        )
    moved = sorted(clip for clip in a if a[clip].scene_id != b[clip].scene_id)
    if moved:
        raise UsageError(f"{unit}: {len(moved)} clips change scene between the runs ({moved[0]})")


def compare_oof(
    a: DevScoreOof, b: DevScoreOof, level: InputLevel, family_of: FamilyOf
) -> DevScoreCompare:
    """b - a per unit on the bootstrap settings both runs share."""
    if (a.bootstrap, a.seed, a.ci) != (b.bootstrap, b.seed, b.ci):
        raise UsageError("the runs were scored with different bootstrap settings")
    va, vb = units(_level(a, level)), units(_level(b, level))
    shared = [unit for unit in va if unit in vb]
    for unit in shared:
        _require_same_clips(unit, va[unit], vb[unit])
    scenes = {clip: scene for unit in shared for clip, scene in _scenes(va[unit]).items()}
    families = family_map(scenes.values(), family_of)
    return DevScoreCompare(
        run_a=a.run_id,
        run_b=b.run_id,
        level=level,
        bootstrap=a.bootstrap,
        seed=a.seed,
        ci=a.ci,
        n_clips=len(scenes),
        stratification=stratification_of(scenes.values(), family_of),
        units={unit: paired_deltas(va[unit], vb[unit], families, a) for unit in shared},
    )
