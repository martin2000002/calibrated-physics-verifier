"""H1 monotonicity: Kendall tau_b between level and the change against the clean clip (D8)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from physverify.calibration.schema import Monotonicity, SceneKendall
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stats import median_or_none
from physverify.calibration.table import (
    CLEAN_LEVEL,
    CalibrationRow,
    Role,
    corrected,
    require_one_clean_per_scene,
)
from physverify.stats.bootstrap import SceneBlock, scene_bootstrap
from physverify.stats.intervals import Ci
from physverify.stats.rank import kendall

TAU_KEY = "kendall_tau_b"


@dataclass(frozen=True)
class LevelDelta:
    """Delta m = m(scene, L) - m(scene, clean); the clean clip itself is (0, 0)."""

    scene_id: str
    level: int
    delta: float


def deltas(rows: Sequence[CalibrationRow], bias: float) -> tuple[LevelDelta, ...]:
    """Pairs of the scenes with a scored clean clip and at least one scored positive."""
    require_one_clean_per_scene(rows)
    clean = {r.scene_id: corrected(r, bias) for r in rows if r.role is Role.CLEAN and r.scored}
    found: list[LevelDelta] = []
    for scene_id in sorted(clean):
        positives = [r for r in rows if r.scene_id == scene_id and r.role is Role.POSITIVE]
        scored = [r for r in positives if r.scored]
        if not scored:
            continue
        found.append(LevelDelta(scene_id, CLEAN_LEVEL, 0.0))
        found += [
            LevelDelta(scene_id, r.pert_level, corrected(r, bias) - clean[scene_id]) for r in scored
        ]
    return tuple(found)


def _tau(pairs: Sequence[LevelDelta]) -> float:
    return kendall([(float(p.level), p.delta) for p in pairs])


def monotonicity(pairs: Sequence[LevelDelta], settings: CalibrationSettings) -> Monotonicity:
    """``h1_pass``: the lower bound of the scene-bootstrap interval of tau_b is above 0."""
    tau = _tau(pairs)
    blocks = [SceneBlock(p.scene_id, p) for p in pairs]
    ci = scene_bootstrap(
        blocks, lambda s: {TAU_KEY: _tau(s)}, settings.bootstrap, settings.seed, settings.ci
    ).get(TAU_KEY, Ci())
    medians = {
        str(level): median_or_none([p.delta for p in pairs if p.level == level])
        for level in sorted({p.level for p in pairs})
    }
    ordered = [m for m in medians.values() if m is not None]
    return Monotonicity(
        kendall_tau_b=tau,
        ci=ci,
        n_pairs=len(pairs),
        n_scenes=len({p.scene_id for p in pairs}),
        median_delta_by_level=medians,
        monotone_medians=all(a <= b for a, b in zip(ordered, ordered[1:], strict=False)),
        h1_pass=ci.low is not None and ci.low > 0.0,
    )


def scene_kendall(rows: Sequence[CalibrationRow], bias: float) -> tuple[SceneKendall, ...]:
    """Per scene with at least two perturbed levels (the ``levels`` subset)."""
    pairs = deltas(rows, bias)
    split = {r.scene_id: r.split.value for r in rows}
    found: list[SceneKendall] = []
    for scene_id in sorted({p.scene_id for p in pairs}):
        members = [p for p in pairs if p.scene_id == scene_id]
        n_levels = len({p.level for p in members}) - 1
        if n_levels < 2:
            continue
        check_id = rows[0].check_id
        found.append(
            SceneKendall(
                scene_id=scene_id,
                check_id=check_id,
                kendall_tau_b=_tau(members),
                n_levels=n_levels,
                split=split[scene_id],
            )
        )
    return tuple(found)
