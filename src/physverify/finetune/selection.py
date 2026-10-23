"""Checkpoint selection on ``dev``; pure, ``eval`` never enters.

Each checkpoint is compared with the base model on the same ``dev`` candidates (same seeds): the
paired difference of the scores re-computed on the checks decided in both members of each
(checkpoint, base) candidate pair, as the pairs are judged, with a normal
interval. A
checkpoint whose abstention rate rises or whose motion drops past the limits is
``suspected_hacking`` and is never chosen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from physverify.core.models import FrozenModel
from physverify.finetune.pairs import Rescore, on_common_checks
from physverify.finetune.records import CandidateScore
from physverify.finetune.settings import SelectSettings


class CheckpointResult(FrozenModel):
    step: int
    adapter: str
    model_tag: str
    n: int
    n_paired: int
    mean_score: float | None
    paired_diff: float | None
    ci: tuple[float, float] | None
    abstention: float
    base_abstention: float
    motion: float | None
    base_motion: float | None
    suspected_hacking: bool


class Selection(FrozenModel):
    """``beats_base`` records whether the chosen checkpoint beats the base on ``dev``; a
    selection written before the field existed reads as False."""

    run_id: str
    scorer: str
    thresholds: str
    selected: CheckpointResult | None
    checkpoints: list[CheckpointResult]
    beats_base: bool = False


def spaced(steps: Sequence[int], count: int) -> list[int]:
    """``count`` checkpoints evenly spread over ``steps``, always including the last one."""
    ordered = sorted(steps)
    if len(ordered) <= count:
        return ordered
    picks = np.linspace(len(ordered) - 1, 0, count)
    return sorted({ordered[int(round(i))] for i in picks})


def _rate(scores: Sequence[CandidateScore]) -> float:
    return sum(s.score is None for s in scores) / len(scores) if scores else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _motion(scores: Sequence[CandidateScore]) -> float | None:
    return _mean([s.motion for s in scores if s.motion is not None])


def _paired_diffs(
    scores: Sequence[CandidateScore], base: Sequence[CandidateScore], rescore: Rescore
) -> list[float]:
    base_by_id = {s.candidate_id: s for s in base}
    diffs = []
    for score in scores:
        other = base_by_id.get(score.candidate_id)
        if other is None or score.score is None or other.score is None:
            continue
        both = on_common_checks(score, other, rescore)
        if both is not None:
            diffs.append(both.score_a - both.score_b)
    return diffs


def evaluate_checkpoint(
    step: int,
    adapter: str,
    model_tag: str,
    scores: Sequence[CandidateScore],
    base: Sequence[CandidateScore],
    cfg: SelectSettings,
    rescore: Rescore,
) -> CheckpointResult:
    diffs = _paired_diffs(scores, base, rescore)
    mean = _mean(diffs)
    ci = None
    if mean is not None and len(diffs) > 1:
        half = cfg.ci_z * float(np.std(diffs, ddof=1)) / math.sqrt(len(diffs))
        ci = (mean - half, mean + half)
    abstention, base_abstention = _rate(scores), _rate(base)
    motion, base_motion = _motion(scores), _motion(base)
    quieter = (
        motion is not None
        and base_motion is not None
        and motion < (1.0 - cfg.max_motion_drop) * base_motion
    )
    return CheckpointResult(
        step=step,
        adapter=adapter,
        model_tag=model_tag,
        n=len(scores),
        n_paired=len(diffs),
        mean_score=_mean([s.score for s in scores if s.score is not None]),
        paired_diff=mean,
        ci=ci,
        abstention=abstention,
        base_abstention=base_abstention,
        motion=motion,
        base_motion=base_motion,
        suspected_hacking=abstention - base_abstention > cfg.max_abstention_rise or quieter,
    )


def choose(results: Sequence[CheckpointResult]) -> CheckpointResult | None:
    """Largest paired difference among the checkpoints not suspected of hacking."""
    usable = [r for r in results if not r.suspected_hacking and r.paired_diff is not None]
    return max(usable, key=lambda r: (r.paired_diff or 0.0, r.step), default=None)


def beats_base(chosen: CheckpointResult | None) -> bool:
    """The chosen checkpoint's paired difference with the base is positive and its CI excludes 0."""
    if chosen is None or chosen.paired_diff is None or chosen.paired_diff <= 0:
        return False
    return chosen.ci is not None and chosen.ci[0] > 0
