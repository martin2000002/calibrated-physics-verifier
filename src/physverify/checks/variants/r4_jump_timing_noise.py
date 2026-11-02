"""R4_jump rigid2_dfloor_timing: rigid2_dfloor with the noise of each stencil grown by the timing
noise of the track times its speed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.checks.r4_track_jump import NoiseStencils, TrackNoise
from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck
from physverify.physics.timing_noise import TimingFit, stencil_sigma, stencil_speed, timing_sigma

FloatArray = NDArray[np.float64]


class TimingJumpCheck(DeclaredDepthJumpCheck):
    """R4_jump rigid2_dfloor_timing [jump_over_noise]: rigid2_dfloor with the noise of stencil k
    and axis a sqrt(sigma_a^2 + (sigma_delta v_{k,a})^2), sigma-hat per axis as in rigid2_dfloor
    and v the median |step| of the stencil (``stencil_speed``). sigma_delta, in rows, is fitted per
    track on the moving cells of the stencils sigma-hat is fitted on (``timing_sigma``); it is 0
    for exact or declared noise, so the oracle keeps rigid2_dfloor's statistic.

    A video whose frames are not evenly spaced in time (repeated or irregular frames under
    uniform timestamps) moves a body by delta v, which v2 reads as jumps on a moving track. The
    price is a declared blind spot: a jump during motion smaller than about sqrt(20) sigma_delta
    v tau / 2 is not seen. The details ``timing_sigma_frames`` (sigma_delta, an estimate),
    ``timing_moving_cells`` and ``timing_sigma_eff`` (the noise per axis at the peak stencil) are
    added to every scored object."""

    variant: ClassVar[str] = "rigid2_dfloor_timing"
    version: ClassVar[str] = "1"

    def _stencil_sigma(
        self, inputs: CheckInputs, index: int, stencils: NoiseStencils, noise: TrackNoise
    ) -> tuple[FloatArray, Mapping[str, object]]:
        if noise.source != "track":
            fit, sigma = TimingFit(0.0, 0), noise.sigma
        else:
            d3, usable = stencils
            speed = stencil_speed(inputs.states.position[:, index])
            fit = timing_sigma(d3, noise.sigma, speed, usable, self.params.min_noise_stencils)
            sigma = noise.sigma
            if fit.sigma_frames:
                sigma = stencil_sigma(noise.sigma, speed, fit.sigma_frames)
        return sigma, {"timing_sigma_frames": fit.sigma_frames, "timing_moving_cells": fit.cells}

    def _peak_details(self, row: int, stencil: FloatArray) -> Mapping[str, object]:
        """``timing_sigma_eff``: the noise per axis of the peak row's stencil, row - 1 of a
        per-stencil ``stencil`` (``noise_jumps`` writes stencil j at row j + 1)."""
        at_peak = stencil[row - 1] if stencil.ndim == 2 else stencil
        return {"timing_sigma_eff": [float(value) for value in at_peak]}


PLUGINS = (TimingJumpCheck,)
