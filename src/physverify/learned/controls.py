"""Gates and shortcut controls of a learned variant, on clip scores of the internal validation
split (or of a control set built like it). A clip's score is the largest estimate over its
subjects; intervals resample whole scenes. A value that cannot be computed (no clean tau, an
empty side, no interval) fails its gate or control.

- C-dec: with tau the conformal (1 - alpha) quantile of the clean scores, the decoys' flag rate
  may exceed the clean flag rate by at most ``decoy_margin``.
- C-inv: positives that are invisible by construction against clean clips: the AUROC interval
  must contain 0.5; with fewer than ``min_invisible`` of them it is not evaluable.
- C-scr (static checks): AUROC of the clip features minus AUROC of single-frame features over the
  same clips; it fails when the interval's lower bound is above 0.
- C-hdri: an AUROC drop larger than ``hdri_drop`` (or undefined) on unseen backgrounds is
  flagged; the caller does not stop on it.
- G-L0 / G-L1: the AUROC lower bound of visible positives against decoys reaches ``oracle_floor``
  (ground-truth inputs) or exceeds 0.5 (perceived inputs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
    min_invisible: int = 5
    ci: float = 0.95
    n_boot: int = 2000
    seed: int = 0


@dataclass(frozen=True)
class ClipScore:
    clip_id: str
    scene_id: str
    role: Role
    score: float


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"
    NOT_RUN = "not_run"


def status(passed: bool) -> Status:
    return Status.PASS if passed else Status.FAIL


@dataclass(frozen=True)
class ControlResult:
    name: str
    status: Status
    value: float
    ci: Ci | None
    detail: str

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS


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


def decoy_control(
    clips: Sequence[ClipScore], settings: ControlSettings, name: str = "C-dec"
) -> ControlResult:
    """``clips``: clean clips and the decoys compared with them; tau from the clean ones."""
    quantile = conformal_quantile(_scores(clips, Role.CLEAN), settings.alpha)
    decoys = np.asarray(_scores(clips, Role.DECOY))
    if quantile.value is None or not quantile.sufficient:
        return ControlResult(name, Status.FAIL, float("nan"), None, "no clean tau")
    if not decoys.size:
        return ControlResult(name, Status.FAIL, float("nan"), None, "no decoys")
    tau = quantile.value
    clean = float(np.mean(np.asarray(_scores(clips, Role.CLEAN)) > tau))
    decoy = float(np.mean(decoys > tau))
    excess = decoy - clean
    detail = f"tau={tau:.6g} clean={clean:.4f} decoy={decoy:.4f} n_decoy={decoys.size}"
    return ControlResult(name, status(excess <= settings.decoy_margin), excess, None, detail)


def invisible_control(
    invisible: Sequence[ClipScore], clean: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    detail = f"n_invisible={len(invisible)}"
    if len(invisible) < settings.min_invisible:
        return ControlResult("C-inv", Status.NOT_EVALUABLE, float("nan"), None, detail)
    point, ci = auroc_ci(invisible, clean, settings)
    contains = ci.low is not None and ci.high is not None and ci.low <= CHANCE <= ci.high
    return ControlResult("C-inv", status(contains), point, ci, detail)


def single_frame_control(
    clip: Sequence[ClipScore], single: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    """Paired by clip id: the AUROC of positives against clean clips, clip minus single frame."""
    paired = {c.clip_id: c for c in single}
    both = [(c, paired[c.clip_id]) for c in clip if c.clip_id in paired]
    blocks = [ClusterBlock(a.scene_id, (a, b)) for a, b in both]

    def delta(items: Sequence[tuple[ClipScore, ClipScore]]) -> dict[str, float]:
        return {AUROC_KEY: role_auroc([a for a, _ in items]) - role_auroc([b for _, b in items])}

    point = delta(both)[AUROC_KEY]
    ci = scene_bootstrap(blocks, delta, settings.n_boot, settings.seed, settings.ci).get(
        AUROC_KEY, Ci()
    )
    passed = ci.low is not None and ci.low <= 0.0
    return ControlResult("C-scr", status(passed), point, ci, f"n_paired={len(both)}")


def role_auroc(clips: Sequence[ClipScore]) -> float:
    return auroc(_scores(clips, Role.POSITIVE), _scores(clips, Role.CLEAN))


def hdri_control(reference: float, unseen: float, settings: ControlSettings) -> ControlResult:
    drop = reference - unseen
    flagged = not drop <= settings.hdri_drop
    detail = f"reference={reference:.4f} unseen={unseen:.4f}"
    if flagged:
        detail += " texture dependence"
    return ControlResult("C-hdri", status(not flagged), drop, None, detail)


def gate(
    name: str,
    visible: Sequence[ClipScore],
    decoys: Sequence[ClipScore],
    settings: ControlSettings,
    passes: Callable[[float], bool],
) -> ControlResult:
    point, ci = auroc_ci(visible, decoys, settings)
    detail = f"n_visible={len(visible)} n_decoy={len(decoys)}"
    return ControlResult(name, status(passes(_low(ci))), point, ci, detail)


def oracle_gate(
    visible: Sequence[ClipScore], decoys: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    return gate("G-L0", visible, decoys, settings, lambda low: low >= settings.oracle_floor)


def perceived_gate(
    visible: Sequence[ClipScore], decoys: Sequence[ClipScore], settings: ControlSettings
) -> ControlResult:
    return gate("G-L1", visible, decoys, settings, lambda low: low > CHANCE)
