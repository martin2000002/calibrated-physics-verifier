"""Out-of-fold values of the dev score, per clip: what ``calibrate compare-devscore`` pairs.

Per check, the corrected magnitude of each held-out row with the bias of the model fitted without
its fold (None when the row is not scored) and that fold's flag; per clip, 1 - S of the fold's
clip score. Pure, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from physverify.calibration.artifact import CheckCalibration, Stratum
from physverify.calibration.fit import LABELLED_ROLES, ClipValue, LevelData, clip_values
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stratified import Valued
from physverify.calibration.table import CalibrationRow, Role, corrected
from physverify.calibration.thresholds import fails
from physverify.calibration.verdict import Rule
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.schema.checks import InputLevel
from physverify.verification.thresholding import ThresholdRule

DEVSCORE_OOF_SCHEMA = "physverify.devscore_oof"
DEVSCORE_OOF_VERSION = "1.0.0"
DEVSCORE_OOF_FILE = "devscore_oof.json"
AGGREGATE = "aggregate"
"""The clip score's unit name next to the check ids."""


class CheckValue(ArtifactModel):
    clip_id: str
    scene_id: str
    role: Role
    value: float | None
    """m - bias of the fold; None when the row is not scored."""
    flag: bool
    """Fails the fold's tau (a fold without tau flags nothing)."""


class LevelOof(ArtifactModel):
    level: InputLevel
    checks: dict[str, tuple[CheckValue, ...]]
    aggregate: tuple[ClipValue, ...]


class DevScoreOof(ArtifactModel):
    schema_: str = Field(default=DEVSCORE_OOF_SCHEMA, alias=SCHEMA_KEY)
    version: str = DEVSCORE_OOF_VERSION
    run_id: str
    bootstrap: int
    seed: int
    ci: float
    levels: tuple[LevelOof, ...]

    def level(self, level: InputLevel) -> LevelOof | None:
        return next((found for found in self.levels if found.level is level), None)


def flagger(check: CheckCalibration, policy: ThresholdRule) -> Callable[[CalibrationRow], bool]:
    rule, tau = Rule(policy, check.sigma.bias, check.sigma.factor), check.tau

    def flagged(row: CalibrationRow) -> bool:
        return tau is not None and fails(row, rule, tau)

    return flagged


def check_values(
    held: Sequence[CalibrationRow], check: CheckCalibration, policy: ThresholdRule
) -> list[CheckValue]:
    flagged = flagger(check, policy)

    def value(row: CalibrationRow) -> float | None:
        found = corrected(row, check.sigma.bias) if row.scored else math.nan
        return found if math.isfinite(found) else None

    return [
        CheckValue(
            clip_id=r.clip_id, scene_id=r.scene_id, role=r.role, value=value(r), flag=flagged(r)
        )
        for r in held
    ]


def level_oof(
    level: InputLevel,
    folds: Sequence[tuple[LevelData, Stratum]],
    check_ids: Sequence[str],
    policy: ThresholdRule,
) -> LevelOof:
    """``folds``: each held-out part with the model fitted without it."""
    checks = {
        check_id: tuple(
            value
            for held, model in folds
            for value in check_values(
                [r for r in held.rows if r.check_id == check_id], model.checks[check_id], policy
            )
        )
        for check_id in check_ids
    }
    aggregate = tuple(c for held, model in folds for c in clip_values(model, held))
    return LevelOof(level=level, checks=checks, aggregate=aggregate)


def check_valued(values: Sequence[CheckValue]) -> dict[str, Valued]:
    """The labelled rows, by clip id."""
    return {
        v.clip_id: Valued(v.scene_id, v.value, v.role is Role.POSITIVE)
        for v in values
        if v.role in LABELLED_ROLES
    }


def clip_valued(values: Sequence[ClipValue]) -> dict[str, Valued]:
    return {v.clip_id: Valued(v.scene_id, v.value, v.perturbed) for v in values}


def oof_report(
    run_id: str, levels: Sequence[LevelOof], settings: CalibrationSettings
) -> DevScoreOof:
    return DevScoreOof(
        run_id=run_id,
        bootstrap=settings.bootstrap,
        seed=settings.seed,
        ci=settings.ci,
        levels=tuple(levels),
    )


def units(level: LevelOof) -> Mapping[str, dict[str, Valued]]:
    """Every check and the clip score (``AGGREGATE``), as labelled values by clip id."""
    found = {check_id: check_valued(values) for check_id, values in level.checks.items()}
    return {**found, AGGREGATE: clip_valued(level.aggregate)}
