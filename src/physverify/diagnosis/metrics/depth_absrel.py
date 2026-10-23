"""DEPTH: per-frame AbsRel and delta1 after aligning the prediction to the truth by its kind."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
)
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

DELTA1 = 1.25


def valid_pixels(pred: DepthMap, truth: DepthMap) -> BoolArray:
    d = pred.depth.astype(np.float64)
    z = truth.depth.astype(np.float64)
    valid: BoolArray = pred.valid & truth.valid & np.isfinite(d) & np.isfinite(z) & (z > 0)
    if pred.kind is not DepthKind.AFFINE_DISPARITY:
        valid &= d > 0
    return valid


def align(
    d: FloatArray, z: FloatArray, kind: DepthKind, eps_frac: float, metric_truth: bool
) -> FloatArray:
    """z_hat for matching 1-D pixel samples ``d`` (prediction) and ``z`` (truth depth).

    Affine disparity: least squares a d + b ~ 1/z, z_hat = 1/max(a d + b, eps) with
    eps = eps_frac * median(1/z). Relative depth (or metric depth against a relative truth):
    z_hat = median(z/d) d."""
    if kind is DepthKind.AFFINE_DISPARITY:
        inverse = 1.0 / z
        design = np.column_stack([d, np.ones_like(d)])
        (a, b), *_ = np.linalg.lstsq(design, inverse, rcond=None)
        eps = eps_frac * float(np.median(inverse))
        aligned: FloatArray = 1.0 / np.maximum(a * d + b, eps)
        return aligned
    if kind is DepthKind.METRIC_Z and metric_truth:
        return d
    scaled: FloatArray = float(np.median(z / d)) * d
    return scaled


def alignment_groups(n_frames: int, mode: str) -> Sequence[Sequence[int]]:
    return [list(range(n_frames))] if mode == "clip" else [[t] for t in range(n_frames)]


def aligned_depth(
    pred: DepthMap, truth: DepthMap, valid: BoolArray, case: MetricCase
) -> FloatArray:
    """z_hat on every valid pixel, NaN elsewhere."""
    if truth.kind is DepthKind.AFFINE_DISPARITY:
        raise ValueError("truth depth must be a depth, not a disparity")
    d, z = pred.depth.astype(np.float64), truth.depth.astype(np.float64)
    out = np.full(d.shape, np.nan)
    metric_truth = truth.kind is DepthKind.METRIC_Z
    for frames in alignment_groups(d.shape[0], case.options.depth_alignment):
        group = np.zeros_like(valid)
        group[frames] = valid[frames]
        if group.any():
            out[group] = align(
                d[group], z[group], pred.kind, case.options.depth_eps_frac, metric_truth
            )
    return out


@dataclass(frozen=True)
class AbsRelMetric:
    """``depth_absrel`` per frame with valid pixels; ``delta1`` pooled over pixels. No context."""

    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "absrel"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, DepthMap) or not isinstance(truth, DepthMap):
            raise TypeError("absrel metric compares two DepthMap")
        valid = valid_pixels(pred, truth)
        z_hat = aligned_depth(pred, truth, valid, case)
        z = truth.depth.astype(np.float64)
        absrel = [
            float(np.mean(np.abs(z_hat[t][valid[t]] - z[t][valid[t]]) / z[t][valid[t]]))
            for t in range(z.shape[0])
            if valid[t].any()
        ]
        ratio = np.maximum(z_hat[valid] / z[valid], z[valid] / z_hat[valid])
        return MetricOutcome(
            samples=MappingProxyType({"depth_absrel": np.array(absrel, dtype=np.float64)}),
            ratios=MappingProxyType(
                {"delta1": (float(np.count_nonzero(ratio < DELTA1)), float(ratio.size))}
            ),
        )


PLUGINS = (AbsRelMetric,)
