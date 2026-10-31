"""Tests of velocities against the measured velocity error; not a plugin.

sigma_v of an object is the profile's relative velocity error times its reference speed, per
component. A row is at rest when its velocity is compatible with zero at chi2_3(alpha); a run of
such rows is kept when its velocity is also constant (end minus start). A velocity change is
significant along a normal when it beats the one-sided normal quantile at alpha / m (Bonferroni
over the m candidates of the track).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from physverify.physics.kinematics import reference_speed, visible_runs
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import ErrorEntry
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
COMPONENTS = 3


@dataclass(frozen=True)
class VelocityError:
    """Relative velocity error e_v (NaN when unknown); sigma_v = e_v * v_ref per component."""

    relative: float

    @classmethod
    def of(cls, entry: ErrorEntry | None) -> VelocityError:
        return cls(entry_rms(entry))

    @property
    def known(self) -> bool:
        return math.isfinite(self.relative)

    def sigma(self, velocity: FloatArray, length: float, duration: float) -> float:
        return self.relative * reference_speed(velocity, length, duration)

    def sigmas(self, states: ObjectStateTable, lengths: FloatArray) -> FloatArray:
        """sigma_v of each object from its visible velocities over the clip's duration."""
        duration = float(states.t[-1] - states.t[0]) if states.n_frames else math.nan
        return np.array(
            [
                self.sigma(states.velocity[states.visible[:, n], n], float(lengths[n]), duration)
                for n in range(states.n_objects)
            ],
            dtype=np.float64,
        )


def rest_quantile(alpha: float) -> float:
    return float(chi2.isf(alpha, COMPONENTS))


def change_quantile(alpha: float, tests: int) -> float:
    return float(norm.isf(alpha / max(tests, 1)))


def rest_rows(velocity: FloatArray, visible: BoolArray, sigma: float, alpha: float) -> BoolArray:
    """Visible rows with ||v||^2 <= chi2_3(alpha) sigma^2 (none when sigma is unknown)."""
    squared = np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=-1)
    with np.errstate(invalid="ignore"):
        rest = squared <= rest_quantile(alpha) * sigma**2
    return np.asarray(rest & visible, dtype=np.bool_)


def long_runs(rows: BoolArray, min_rows: int) -> tuple[slice, ...]:
    return tuple(run for run in visible_runs(rows) if run.stop - run.start >= min_rows)


def constant(velocity: FloatArray, run: slice, sigma: float, alpha: float) -> bool:
    """||v_end - v_start||^2 <= chi2_3(alpha) 2 sigma^2: no velocity change across the run."""
    change = velocity[run.stop - 1] - velocity[run.start]
    return bool(float(change @ change) <= rest_quantile(alpha) * 2.0 * sigma**2)


def rest_runs(
    velocity: FloatArray, visible: BoolArray, sigma: float, alpha: float, min_rows: int
) -> tuple[slice, ...]:
    """Runs of at least ``min_rows`` rest rows whose velocity is constant; a run that is not
    constant is dropped whole, never split."""
    rows = rest_rows(velocity, visible, sigma, alpha)
    return tuple(r for r in long_runs(rows, min_rows) if constant(velocity, r, sigma, alpha))


def rest_mask(
    states: ObjectStateTable, sigmas: FloatArray, alpha: float, min_rows: int
) -> BoolArray:
    """(T, N) rows inside each object's rest runs."""
    mask = np.zeros(states.visible.shape, dtype=np.bool_)
    for n in range(states.n_objects):
        velocity = states.velocity[:, n]
        for run in rest_runs(velocity, states.visible[:, n], float(sigmas[n]), alpha, min_rows):
            mask[run, n] = True
    return mask


@dataclass(frozen=True)
class NormalChange:
    """Velocity change du along a normal with its sigma and the number of tests m."""

    delta: float
    sigma: float
    tests: int

    @classmethod
    def along(
        cls, delta_v: FloatArray, normal: FloatArray, sigma_v: float, tests: int
    ) -> NormalChange:
        """du = delta_v . normal with sigma = sqrt(2) sigma_v: delta_v is the difference of two
        velocity estimates, one on each side of the kink."""
        delta = float(np.asarray(delta_v, dtype=np.float64) @ normal)
        return cls(delta, math.sqrt(2.0) * sigma_v, tests)

    @property
    def ratio(self) -> float:
        return self.delta / self.sigma if self.sigma > 0.0 else math.nan

    def bound(self, alpha: float) -> float:
        return change_quantile(alpha, self.tests) * self.sigma

    def significant(self, alpha: float) -> bool:
        return bool(self.delta > self.bound(alpha))

    @property
    def confidence(self) -> float:
        return float(norm.cdf(self.ratio))
