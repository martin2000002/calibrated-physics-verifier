"""R5a line of sight: whitened's candidates, each window fitted by generalized least squares on
the covariance a monocular lift implies, and scored by the tilt of its acceleration away from the
perceived gravity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from physverify.checks.r5a_fall_shape import Candidate, FallShapeCheck
from physverify.checks.variants._line_of_sight import known_gravity_sigma, window_covariance
from physverify.checks.variants._whitened import WhitenedFallShapeCheck
from physverify.physics.flights import Flight
from physverify.physics.measures.vector_fall import VectorFallFit, gls_fall_fit, tilt_statistic
from physverify.schema.checks import AbstainReason
from physverify.schema.states import ObjectStateTable


class LineOfSightFallShapeCheck(WhitenedFallShapeCheck):
    """R5a los_tilt [accel_tilt_chi2_4]: per candidate of v1, each of the two windows gets the
    GLS parabola p0 + v0 tau + a tau^2 / 2 under Sigma_t = r_d^2 p p^T + diag(rest)
    (``line_of_sight_covariance`` of whitened's floored sigma), and the magnitude is the sum
    over both windows of ``tilt_statistic``: the part of a orthogonal to the perceived gravity
    against the fit's covariance plus a gravity tilt of gravity_dir_sigma per axis. Each term is
    chi-square with two degrees of freedom when a is parallel to gravity; the sum is about
    chi-square(4) (the shared gravity error makes the terms dependent, so it is approximate).
    A depth error moves a point along its ray and is weighted there only, so a clip whose depth
    breathes no longer reads as a change of gravity, but a fall seen along the line of sight
    still tilts if its depth scale drifts. A change of |a| along gravity is seen only through
    the misfit of the window that holds it. Without the gravity direction or its sigma no window
    scores and the object abstains with UNKNOWN_UNCERTAINTY; the sigma rule and the
    ``accel_rel_diff`` estimate are whitened's."""

    variant: ClassVar[str] = "los_tilt"
    version: ClassVar[str] = "1"
    magnitude_unit: ClassVar[str] = "accel_tilt_chi2_4"

    def _statistic(
        self, states: ObjectStateTable, windows: Sequence[Flight], offsets: Sequence[Any]
    ) -> float:
        if not known_gravity_sigma(states):
            return math.nan
        fits = [self._vector_fit(states, w, o) for w, o in zip(windows, offsets, strict=True)]
        return _tilt(states, fits)

    def _no_candidate(self, states: ObjectStateTable, flights: Sequence[Flight]) -> AbstainReason:
        if not known_gravity_sigma(states):
            return AbstainReason.UNKNOWN_UNCERTAINTY
        return super()._no_candidate(states, flights)

    def _statistic_details(self, states: ObjectStateTable, best: Candidate) -> dict[str, Any]:
        rel_diff = FallShapeCheck._statistic(self, states, best.windows, (0.0, 0.0))
        estimate = {self.perturbation_estimate: rel_diff}
        fits = [self._vector_fit(states, w, 0.0) for w in best.windows]
        if any(f is None for f in fits) or not known_gravity_sigma(states):
            return estimate
        return estimate | {
            "gls_accelerations": [f.a for f in fits if f is not None],
            "gls_accel_cov": [f.a_cov for f in fits if f is not None],
            "tilt_per_window": [_tilt(states, [f]) for f in fits],
        }

    def _vector_fit(
        self, states: ObjectStateTable, window: Flight, offset: Any
    ) -> VectorFallFit | None:
        sigma = self._window_sigma(states, window)
        if sigma is None:
            return None
        cov = window_covariance(states, window, sigma, self.params.exact_sigma_lengths)
        positions = states.position[window.rows, window.index] + offset
        return gls_fall_fit(states.t[window.rows], positions, cov)


def _tilt(states: ObjectStateTable, fits: Sequence[VectorFallFit | None]) -> float:
    """Sum of the windows' tilt statistics about the unit perceived gravity; NaN when a window
    has no fit."""
    if any(f is None for f in fits):
        return math.nan
    g = np.asarray(states.gravity_dir, dtype=np.float64)
    g = g / np.linalg.norm(g)
    sigma = float(states.gravity_dir_sigma)
    return float(sum(tilt_statistic(f, g, sigma) for f in fits if f is not None))


PLUGINS = (LineOfSightFallShapeCheck,)
