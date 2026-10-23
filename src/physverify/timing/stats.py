"""Order statistics of timing samples: per clip, then across clips over the per-clip medians.

Warm-up samples never enter a statistic. Quantiles are Hyndman and Fan type 7 (linear
interpolation), the same as ``numpy.quantile(method="linear")``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from physverify.timing.records import MeasureStats, Sample, TimingError

ACROSS_CLIPS: Final = "*"


@dataclass(frozen=True)
class Summary:
    n: int
    median: float
    p25: float
    p75: float
    iqr: float
    iqr_rel: float  # NaN when the median is 0
    mean: float
    sd: float  # ddof = 1, NaN when n = 1
    min: float
    max: float


def quantile(ordered: Sequence[float], p: float) -> float:
    """Type 7: h = (n - 1) p, Q = x[floor h] + (h - floor h) (x[floor h + 1] - x[floor h])."""
    if not ordered:
        raise TimingError("quantile of an empty sample")
    if not 0.0 <= p <= 1.0:
        raise TimingError(f"quantile level {p} is outside [0, 1]")
    h = (len(ordered) - 1) * p
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def summarize(values: Sequence[float]) -> Summary:
    if not values:
        raise TimingError("no samples to summarize")
    ordered = sorted(float(value) for value in values)
    median = quantile(ordered, 0.5)
    p25, p75 = quantile(ordered, 0.25), quantile(ordered, 0.75)
    return Summary(
        n=len(ordered),
        median=median,
        p25=p25,
        p75=p75,
        iqr=p75 - p25,
        iqr_rel=(p75 - p25) / median if median != 0 else math.nan,
        mean=math.fsum(ordered) / len(ordered),
        sd=statistics.stdev(ordered) if len(ordered) > 1 else math.nan,
        min=ordered[0],
        max=ordered[-1],
    )


def stat_row(
    target: str, clip_id: str, measure: str, unit: str, values: Sequence[float], max_iqr_rel: float
) -> MeasureStats:
    summary = summarize(values)
    return MeasureStats(
        target=target,
        clip_id=clip_id,
        measure=measure,
        unit=unit,
        unstable=summary.iqr_rel > max_iqr_rel,
        **asdict(summary),
    )


def per_clip(samples: Iterable[Sample], max_iqr_rel: float) -> list[MeasureStats]:
    """One row per (target, clip, measure), in the order the samples arrived."""
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for sample in samples:
        if not sample.warmup:
            key = (sample.target, sample.clip_id, sample.measure, sample.unit)
            groups.setdefault(key, []).append(sample.value)
    return [stat_row(*key, values, max_iqr_rel) for key, values in groups.items()]


def across_clips(rows: Sequence[MeasureStats], max_iqr_rel: float) -> list[MeasureStats]:
    """Rows with ``clip_id = "*"`` over the per-clip medians (``n`` = clips with samples)."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.clip_id != ACROSS_CLIPS:
            groups.setdefault((row.target, row.measure, row.unit), []).append(row.median)
    return [
        stat_row(target, ACROSS_CLIPS, measure, unit, medians, max_iqr_rel)
        for (target, measure, unit), medians in groups.items()
    ]


def stat_rows(samples: Iterable[Sample], max_iqr_rel: float) -> list[MeasureStats]:
    """Every row of ``stats.csv``: per clip first, then across clips."""
    clip_rows = per_clip(samples, max_iqr_rel)
    return [*clip_rows, *across_clips(clip_rows, max_iqr_rel)]


def across_medians(rows: Iterable[MeasureStats]) -> dict[str, float]:
    return {row.measure: row.median for row in rows if row.clip_id == ACROSS_CLIPS}
