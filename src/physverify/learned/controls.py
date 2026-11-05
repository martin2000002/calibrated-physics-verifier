"""Gates and shortcut controls of a learned variant, on clip scores of the internal validation
split (or of a control set built like it). A clip's score is the largest estimate over its
subjects; intervals resample whole scenes.

- C-dec: with tau the conformal (1 - alpha) quantile of the clean scores, the decoys' flag rate
  may exceed the clean flag rate by at most ``decoy_margin``.
- C-inv: positives that are invisible by construction against clean clips: the AUROC interval
  must contain 0.5.
- C-scr (static checks): AUROC of the clip features minus AUROC of single-frame features over the
  same clips; it fails when the interval's lower bound is above 0.
- C-hdri: an AUROC drop larger than ``hdri_drop`` on unseen backgrounds is flagged, not failed.
- G-L0 / G-L1: the AUROC lower bound of visible positives against decoys reaches ``oracle_floor``
  (ground-truth inputs) or exceeds 0.5 (perceived inputs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.calibration.stats import auroc, conformal_quantile
from physverify.learned.samples import Role
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.stats.intervals import Ci

AUROC_KEY = "auroc"
CHANCE = 0.5


@dataclass(frozen=True)
class ControlSettings:
    alpha: float = 0.05
    decoy_margin: float = 0.02
    hdri_drop: float = 0.10
    oracle_floor: float = 0.7
    ci: float = 0.95
    n_boot: int = 2000
    seed: int = 0


@dataclass(frozen=True)
class ClipScore:
    clip_id: str
    scene_id: str
    role: Role
    score: float


@dataclass(frozen=True)
class ControlResult:
    name: str
    passed: bool
    value: float
    ci: Ci | None
    detail: str


def clip_scores(
    keys: Sequence[tuple[str, str, Role]], estimates: Sequence[float]
) -> list[ClipScore]:
    """The largest estimate of each clip; ``keys`` are (clip_id, scene_id, role) per subject."""
    best: dict[str, ClipScore] = {}
    for (clip, scene, role), value in zip(keys, estimates, strict=True):
        if clip not in best or value > best[clip].score:
            best[clip] = ClipScore(clip, scene, role, float(value))
    return list(best.values())


def _scores(clips: Sequence[ClipScore], role: Role) -> list[float]:
    return [c.score for c in clips if c.role is role]


def auroc_ci(
    positives: Sequence[ClipScore], negatives: Sequence[ClipScore], settings: ControlSettings
) -> tuple[float, Ci]:
    blocks = [ClusterBlock(c.scene_id, (c.score, True)) for c in positives]
    blocks += [ClusterBlock(c.scene_id, (c.score, False)) for c in negatives]

    def statistic(items: Sequence[tuple[float, bool]]) -> dict[str, float]:
        pos = [s for s, positive in items if positive]
        neg = [s for s, positive in items if not positive]
        return {AUROC_KEY: auroc(pos, neg)}

    point = auroc([c.score for c in positives], [c.score for c in negatives])
    intervals = scene_bootstrap(blocks, statistic, settings.n_boot, settings.seed, settings.ci)
    return point, intervals.get(AUROC_KEY, Ci())


def _low(ci: Ci) -> float:
    return ci.low if ci.low is not None else float("nan")


def decoy_control(clips: Sequence[ClipScore], settings: ControlSettings) -> ControlResult:
    tau = conformal_quantile(_scores(clips, Role.CLEAN), settings.alpha).value
    if tau is None:
        return ControlResult("C-dec", False, float("nan"), None, "no clean tau")
    clean = float(np.mean(np.asarray(_scores(clips, Role.CLEAN)) > tau))
    decoys = np.asarray(_scores(clips, Role.DECOY))
    decoy = float(np.mean(decoys > tau)) if decoys.size else float("nan")
    excess = decoy - clean
    passed = bool(excess <= settings.decoy_margin)
    detail = f"tau={tau:.6g} clean={clean:.4f} decoy={decoy:.4f}"
    return ControlResult("C-dec", passed, excess, None, detail)


def invisible_control(
    invisible: Sequence[ClipScore], clean: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    point, ci = auroc_ci(invisible, clean, settings)
    contains = ci.low is not None and ci.high is not None and ci.low <= CHANCE <= ci.high
    return ControlResult("C-inv", contains, point, ci, f"n_invisible={len(invisible)}")


def single_frame_control(
    clip: Sequence[ClipScore], single: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    """Paired by clip id: the AUROC of positives against clean clips, clip minus single frame."""
    paired = {c.clip_id: c for c in single}
    both = [(c, paired[c.clip_id]) for c in clip if c.clip_id in paired]
    blocks = [ClusterBlock(a.scene_id, (a, b)) for a, b in both]

    def delta(items: Sequence[tuple[ClipScore, ClipScore]]) -> dict[str, float]:
        return {AUROC_KEY: _role_auroc([a for a, _ in items]) - _role_auroc([b for _, b in items])}

    point = delta(both)[AUROC_KEY]
    ci = scene_bootstrap(blocks, delta, settings.n_boot, settings.seed, settings.ci).get(
        AUROC_KEY, Ci()
    )
    passed = not (ci.low is not None and ci.low > 0.0)
    return ControlResult("C-scr", passed, point, ci, f"n_paired={len(both)}")


def _role_auroc(clips: Sequence[ClipScore]) -> float:
    return auroc(_scores(clips, Role.POSITIVE), _scores(clips, Role.CLEAN))


def hdri_control(reference: float, unseen: float, settings: ControlSettings) -> ControlResult:
    drop = reference - unseen
    flagged = bool(drop > settings.hdri_drop)
    return ControlResult("C-hdri", not flagged, drop, None, "texture dependence" if flagged else "")


def gate(
    name: str,
    visible: Sequence[ClipScore],
    decoys: Sequence[ClipScore],
    settings: ControlSettings,
    passes: Callable[[float], bool],
) -> ControlResult:
    point, ci = auroc_ci(visible, decoys, settings)
    return ControlResult(name, bool(passes(_low(ci))), point, ci, f"n_visible={len(visible)}")


def oracle_gate(
    visible: Sequence[ClipScore], decoys: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    return gate("G-L0", visible, decoys, settings, lambda low: low >= settings.oracle_floor)


def perceived_gate(
    visible: Sequence[ClipScore], decoys: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    return gate("G-L1", visible, decoys, settings, lambda low: low > CHANCE)
