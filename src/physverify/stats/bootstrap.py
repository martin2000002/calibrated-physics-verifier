"""Percentile bootstrap over whole clusters (clips or scenes), whose blocks are correlated."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from physverify.stats.intervals import Ci, Interval


class Clustered(Protocol):
    @property
    def clip_id(self) -> str: ...


@dataclass(frozen=True)
class BootstrapResult:
    """``non_finite`` counts, per key, the kept replicas whose value was NaN or infinite (for
    example kappa on a one-class resample); they are left out of that key's interval."""

    intervals: Mapping[str, Interval]
    replicas: int
    dropped: int
    non_finite: Mapping[str, int] = field(default_factory=dict)


def _interval(values: Sequence[float], ci: float) -> Interval:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return Interval(low=math.nan, high=math.nan)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(finite, (alpha, 1.0 - alpha))
    return Interval(low=float(low), high=float(high))


def bootstrap[B: Clustered](
    blocks: Sequence[B],
    statistic: Callable[[Sequence[B]], Mapping[str, float]],
    n: int,
    seed: int,
    ci: float,
    clusters: Sequence[str] | None = None,
) -> BootstrapResult:
    """Percentile intervals from ``n`` resamples of whole clips (blocks of a clip are correlated).

    ``clusters`` are the clip ids to resample (default: the clips of ``blocks``); a replica that
    draws only clips without blocks is dropped and counted. Non-finite values of a key are left
    out of that key's interval and counted in ``non_finite``.
    """
    ids = tuple(sorted(set(clusters) if clusters is not None else {b.clip_id for b in blocks}))
    if n <= 0 or not ids:
        return BootstrapResult(intervals={}, replicas=0, dropped=0)
    by_clip = {clip_id: [b for b in blocks if b.clip_id == clip_id] for clip_id in ids}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    dropped = 0
    for _ in range(n):
        sample = [
            b for index in rng.integers(0, len(ids), size=len(ids)) for b in by_clip[ids[index]]
        ]
        if not sample:
            dropped += 1
            continue
        for key, value in statistic(sample).items():
            draws.setdefault(key, []).append(value)
    intervals = {key: _interval(values, ci) for key, values in sorted(draws.items())}
    non_finite = {
        key: int(np.sum(~np.isfinite(np.asarray(values, dtype=np.float64))))
        for key, values in sorted(draws.items())
    }
    return BootstrapResult(intervals, n - dropped, dropped, non_finite)


@dataclass(frozen=True)
class ClusterBlock[T]:
    """One ``item`` of a resampling cluster (a scene, a caption) named by ``clip_id``."""

    clip_id: str
    item: T


def scene_bootstrap[T](
    blocks: Sequence[ClusterBlock[T]],
    statistic: Callable[[Sequence[T]], Mapping[str, float]],
    n: int,
    seed: int,
    ci: float,
) -> dict[str, Ci]:
    """Percentile intervals from resampling whole scenes (a drawn twice scene counts twice);
    ``non_finite`` counts the replicates whose statistic was undefined."""
    result = bootstrap(blocks, lambda sample: statistic([b.item for b in sample]), n, seed, ci)
    return {
        key: Ci(
            low=_or_none(i.low),
            high=_or_none(i.high),
            non_finite=result.non_finite.get(key, 0),
        )
        for key, i in result.intervals.items()
    }


def _or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
