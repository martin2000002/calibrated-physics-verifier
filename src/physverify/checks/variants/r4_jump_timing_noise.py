"""R4_jump rigid2_dfloor_timing: rigid2_dfloor with the noise of each stencil grown by the timing
noise of the track times its speed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.checks._peaks import first_max
from physverify.checks.r4_track_jump import TrackNoise
from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck
from physverify.physics.measures.track_jump import noise_jumps
from physverify.physics.timing_noise import TimingFit, stencil_sigma, stencil_speed, timing_sigma
from physverify.physics.tracks import noise_stencils

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
        self, inputs: CheckInputs, index: int, excluded: NDArray[np.bool_], noise: TrackNoise
    ) -> tuple[FloatArray, Mapping[str, object]]:
        positions = inputs.states.position[:, index]
        if noise.source != "track":
            fit, sigma = TimingFit(0.0, 0), noise.sigma
        else:
            d3, usable = noise_stencils(inputs.states, index, excluded)
            speed = stencil_speed(positions)
            fit = timing_sigma(d3, noise.sigma, speed, usable, self.params.min_noise_stencils)
            sigma = noise.sigma
            if fit.sigma_frames:
                sigma = stencil_sigma(noise.sigma, speed, fit.sigma_frames)
        return sigma, {
            "timing_sigma_frames": fit.sigma_frames,
            "timing_moving_cells": fit.cells,
            "timing_sigma_eff": _peak_sigma(positions, sigma, excluded),
        }


def _peak_sigma(
    positions: FloatArray, sigma: FloatArray, excluded: NDArray[np.bool_]
) -> list[float] | None:
    """The noise per axis of the stencil the statistic peaks at; None without a finite one."""
    per_row = noise_jumps(positions, sigma, excluded)
    if not np.isfinite(per_row).any():
        return None
    row = first_max(per_row)
    return [float(value) for value in (sigma[row - 1] if sigma.ndim == 2 else sigma)]


PLUGINS = (TimingJumpCheck,)
