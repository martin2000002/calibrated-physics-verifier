"""Measurement cases and sample-level metric outcomes (Level 1)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from physverify.datasets._base import GroundTruthStore
from physverify.diagnosis._base import StageMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MetricCase:
    """``pred`` from V_k, ``truth`` and ``context`` from V_{k+1} (the truth variant)."""

    clip: ClipManifest
    pred: StageOutput
    truth: StageOutput
    context: Mapping[Stage, StageOutput]
    gt: GroundTruthStore | None
    options: MetricSettings
    plan: ResolvedPlan | None


@dataclass(frozen=True)
class MetricOutcome:
    """Signed residuals or per-sample scores, and (numerator, denominator) of set rates."""

    samples: Mapping[str, FloatArray] = field(default_factory=lambda: MappingProxyType({}))
    ratios: Mapping[str, tuple[float, float]] = field(default_factory=lambda: MappingProxyType({}))


@runtime_checkable
class SampledMetric(StageMetric, Protocol):
    def samples(self, case: MetricCase) -> MetricOutcome: ...


def stage_output[O: StageOutput](
    outputs: Mapping[Stage, StageOutput], stage: Stage, kind: type[O]
) -> O | None:
    """The output of ``stage`` narrowed to ``kind``; None when the stage is absent."""
    output = outputs.get(stage)
    if output is None:
        return None
    if not isinstance(output, kind):
        raise TypeError(f"{stage.value} output is {type(output).__name__}, not {kind.__name__}")
    return output


def sample_stats(values: FloatArray) -> tuple[float, float, float, int]:
    """Mean, std (ddof 1), RMS and count of the finite values."""
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    std = float(np.std(finite, ddof=1)) if n > 1 else math.nan
    return float(np.mean(finite)), std, float(np.sqrt(np.mean(finite**2))), n


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def summarize(outcome: MetricOutcome) -> dict[str, float]:
    """``<q>.mean/.std/.rms/.n`` over finite samples; ``<r>``, ``<r>.num``, ``<r>.den``."""
    summary: dict[str, float] = {}
    for name, values in outcome.samples.items():
        mean, std, rms, n = sample_stats(np.asarray(values, dtype=np.float64).ravel())
        summary |= {f"{name}.mean": mean, f"{name}.std": std, f"{name}.rms": rms}
        summary[f"{name}.n"] = float(n)
    for name, (numerator, denominator) in outcome.ratios.items():
        summary |= {f"{name}.num": float(numerator), f"{name}.den": float(denominator)}
        summary[name] = _rate(numerator, denominator)
    return summary


def pool(outcomes: Iterable[MetricOutcome]) -> MetricOutcome:
    """Concatenate the samples and add up the ratio terms of several outcomes."""
    samples: dict[str, list[FloatArray]] = {}
    ratios: dict[str, tuple[float, float]] = {}
    for outcome in outcomes:
        for name, values in outcome.samples.items():
            samples.setdefault(name, []).append(np.asarray(values, dtype=np.float64).ravel())
        for name, (numerator, denominator) in outcome.ratios.items():
            total = ratios.get(name, (0.0, 0.0))
            ratios[name] = (total[0] + numerator, total[1] + denominator)
    joined = {name: np.concatenate(parts) for name, parts in samples.items()}
    return MetricOutcome(MappingProxyType(joined), MappingProxyType(ratios))


def bare_compute(
    metric: SampledMetric,
    pred: StageOutput,
    truth: StageOutput,
    clip: ClipManifest,
    options: MetricSettings,
) -> dict[str, float]:
    """``StageMetric.compute`` without context: keys that need context come out NaN."""
    case = MetricCase(
        clip=clip,
        pred=pred,
        truth=truth,
        context=MappingProxyType({}),
        gt=None,
        options=options,
        plan=None,
    )
    return summarize(metric.samples(case))
