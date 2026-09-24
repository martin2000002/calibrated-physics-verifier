"""Scale from the longest window of an object falling with constant acceleration (ADR 0005, 4.7).

s = g0 / |a| assumes the clip obeys standard gravity, so R5b abstains on this scale (ADR 0001).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.physics.geometry import characteristic_lengths
from physverify.physics.kinematics import uniform_step, visible_runs
from physverify.physics.measures.r5_gravity import FallFit, fall_fit, fall_residual_ratio
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
FIT_PARAMETERS = 3
"""p0, v0, a per axis."""


class Params(ProviderParams):
    min_frames: int = Field(default=5, ge=FIT_PARAMETERS + 1)
    max_residual_ratio: float = Field(default=0.1, gt=0.0)
    min_drop_rel: float = Field(default=1.0, ge=0.0)
    max_windows: int = Field(default=5000, ge=1)


@dataclass(frozen=True)
class FallWindow:
    fit: FallFit
    times: FloatArray
    residual_ratio: float


class FreeFallScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "free_fall"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        best = self._best_window(lift)
        if best is None:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, "no_free_fall")
        accel = float(np.linalg.norm(best.fit.a))
        scale = STANDARD_GRAVITY / accel
        sigma = scale * acceleration_sigma(best.fit, best.times) / accel
        detail = f"window={best.fit.n}"
        return ScaleEstimate(scale, sigma, ScaleSource.FREE_FALL, lift.source_key, detail)

    def _best_window(self, lift: ObjectStateTable) -> FallWindow | None:
        """Longest accepted window over all objects; ties go to the smaller residual ratio."""
        best: FallWindow | None = None
        candidates = islice(self._candidates(lift), self.params.max_windows)
        for times, positions, length in candidates:
            if best is not None and times.size < best.times.size:
                break
            window = self._accept(times, positions, length)
            if window is not None and (best is None or window.residual_ratio < best.residual_ratio):
                best = window
        return best

    def _candidates(self, lift: ObjectStateTable) -> Iterator[tuple[FloatArray, FloatArray, float]]:
        """Contiguous visible windows [i, i + W) of every object, longest first."""
        runs: list[tuple[slice, int, float]] = []
        for n in range(lift.n_objects):
            valid = lift.visible[:, n] & np.all(np.isfinite(lift.position[:, n]), axis=1)
            lengths = characteristic_lengths(lift.size[:, n][valid])
            finite = lengths[np.isfinite(lengths)]
            length = float(np.median(finite)) if finite.size else 0.0
            runs.extend((run, n, length) for run in visible_runs(valid))
        longest = max((run.stop - run.start for run, _, _ in runs), default=0)
        for width in range(longest, self.params.min_frames - 1, -1):
            for run, n, length in runs:
                for start in range(run.start, run.stop - width + 1):
                    frames = slice(start, start + width)
                    yield lift.t[frames], lift.position[frames, n], length

    def _accept(self, times: FloatArray, positions: FloatArray, length: float) -> FallWindow | None:
        duration = times.size * uniform_step(times)
        fit = fall_fit(times, positions)
        drop = 0.5 * float(np.linalg.norm(fit.a)) * duration**2
        if not (length > 0 and drop >= self.params.min_drop_rel * length):
            return None
        ratio = fall_residual_ratio(fit, duration)
        if not ratio <= self.params.max_residual_ratio:
            return None
        return FallWindow(fit=fit, times=times, residual_ratio=ratio)


def acceleration_sigma(fit: FallFit, times: FloatArray) -> float:
    """sigma_a = sigma_res sqrt([(D^T D)^-1]_aa) for p = p0 + v0 tau + a tau^2 / 2.

    ``fit.rms`` sums the three axes, so the per-axis residual sigma is
    rms sqrt(n / (3 (n - 3))) (unbiased for three fitted parameters per axis)."""
    tau = times - times[0]
    design = np.column_stack([np.ones_like(tau), tau, 0.5 * tau**2])
    dof = fit.n - FIT_PARAMETERS
    sigma_res = fit.rms * math.sqrt(fit.n / (3.0 * dof))
    return sigma_res * math.sqrt(float(np.linalg.inv(design.T @ design)[2, 2]))


PLUGINS = (FreeFallScale,)
