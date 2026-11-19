"""R3a image_apex: restitution from the apex heights of the projected track, with no depth of the
body."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._collision_windows import FloatArray, IntArray, side_rows
from physverify.checks._params import as_params
from physverify.checks.r3a_bounce_restitution import Bounce, RestitutionCheck, RestitutionParams
from physverify.physics.tracks import body_length, flight_mask, poly_fit
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, ObjectStateTable


class ImageApexRestitutionCheck(RestitutionCheck):
    """R3a image_apex [e_minus_1]: e = sqrt(H+ / H-) of the apex heights of the flight runs on
    each side of the impact, read along the projected normal d = J n / |J n| of the body's
    projected centre u = pi(K p).

    u lies on the pixel's ray, so it does not depend on the depth; d is (K[:2] - u K[2]) n at
    the incoming fit's pixel at t_c, where z drops out. Heights are s = u . d, measured from s_c,
    the mean of the two sides' unweighted parabolas of u at t_c (the fit rows); each apex is the
    highest row of its run, refined by the vertex of the parabola through it and its two
    neighbours when interior and concave, the shift clipped to one row. Windows, t_c and the
    minimum normal speed gate are those of the default implementation. A ground-truth LIFT (by
    provenance) is scored by the default implementation.

    sigma is the delta-method propagation of ``position_sigma`` (independent rows, diagonal per
    axis) through the projection: sigma_s,j^2 = d^T J_j diag(sigma_j^2) J_j^T d with
    J_j = (K[:2] - u_j K[2]) / z_j, and dE/ds_j from the apex weights and the fit gains, so it is
    invariant to the depth scale of each row. It treats t_c and d as fixed, lets the depth and
    the isotropic centroid error of the lift leak into the image plane (an overestimate), does not
    model rows correlated by smoothing, and is NaN when a used row's sigma is."""

    variant: ClassVar[str] = "image_apex"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = RestitutionParams

    def __init__(self, params: BaseModel) -> None:
        self.params = as_params(RestitutionParams, params)

    def _score(
        self,
        inputs: CheckInputs,
        subject: tuple[str, str],
        index: int,
        bounce: Bounce,
        t_c: float,
        span: tuple[int, int],
        details: dict[str, Any],
    ) -> CheckResult:
        camera_k, ill = _camera_k(inputs), AbstainReason.ILL_CONDITIONED
        if inputs.provenance.get(Stage.LIFT.value) == GROUND_TRUTH:
            if camera_k is not None:
                details["tangent_bias"] = _tangent_bias(camera_k, bounce, t_c)
            return super()._score(inputs, subject, index, bounce, t_c, span, details)
        if camera_k is None:
            return abstain(type(self), inputs, subject, AbstainReason.MISSING_INPUT, span, details)
        v_minus, _ = bounce.velocities(t_c)
        floor = self.params.min_normal_speed * body_length(inputs.states, index)
        if not abs(float(v_minus @ bounce.normal)) >= floor:
            return abstain(type(self), inputs, subject, ill, span, details)
        runs = _run_rows(inputs, index, bounce, self.params)
        apex = apex_restitution(camera_k, inputs.states, index, bounce, runs, t_c)
        details |= apex.details()
        if not math.isfinite(apex.e):
            return abstain(type(self), inputs, subject, ill, span, details)
        return unscored(type(self), inputs, subject, apex.e - 1.0, apex.sigma, span, details)


@dataclass(frozen=True, eq=False)
class ApexRestitution:
    """e and its sigma; ``weights`` is dE/ds_j and ``height_sigma`` sigma_s,j on the table rows
    (zero weight off the used rows)."""

    e: float
    sigma: float
    s_c: float
    heights: tuple[float, float]
    interior: tuple[bool, bool]
    direction: FloatArray
    weights: FloatArray
    height_sigma: FloatArray

    def details(self) -> dict[str, Any]:
        return {
            "e": self.e,
            "s_c": self.s_c,
            "height_before": self.heights[0],
            "height_after": self.heights[1],
            "apex_interior_before": self.interior[0],
            "apex_interior_after": self.interior[1],
        }


@dataclass(frozen=True, eq=False)
class Apex:
    """Time and height of a run's apex, its weights dA/ds on the run's rows and whether the
    highest row is interior."""

    t: float
    height: float
    weights: FloatArray
    interior: bool


def apex_restitution(
    camera_k: FloatArray,
    states: ObjectStateTable,
    index: int,
    bounce: Bounce,
    runs: tuple[IntArray, IntArray],
    t_c: float,
) -> ApexRestitution:
    sides = (bounce.before, bounce.after)
    ones = np.ones(1)
    fits = [poly_fit(side.t, _project(camera_k, side.y), ones, 2) for side in sides]
    d = _image_normal(camera_k, fits[0].value_at(t_c)[0], bounce.normal)
    s_c = float(np.mean([fit.value_at(t_c)[0] @ d for fit in fits]))
    positions = states.position[:, index]
    n_rows = positions.shape[0]
    apexes = [run_apex(states.t[rows], _project(camera_k, positions[rows]) @ d) for rows in runs]
    drop, climb = apexes[0].height - s_c, apexes[1].height - s_c
    e = math.sqrt(climb / drop) if drop > 0.0 and climb > 0.0 else math.nan
    weights, used = _gradient(states.t, bounce, runs, apexes, t_c, (e, drop, climb))
    height_sigma = np.zeros(n_rows)
    height_sigma[used] = _row_sigma_along(
        camera_k, positions[used], states.position_sigma[used, index], d
    )
    sigma = float(np.sqrt(np.sum(weights[used] ** 2 * height_sigma[used] ** 2)))
    return ApexRestitution(
        e=e,
        sigma=sigma,
        s_c=s_c,
        heights=(drop, climb),
        interior=(apexes[0].interior, apexes[1].interior),
        direction=d,
        weights=weights,
        height_sigma=height_sigma,
    )


def _gradient(
    t: FloatArray,
    bounce: Bounce,
    runs: tuple[IntArray, IntArray],
    apexes: list[Apex],
    t_c: float,
    restitution: tuple[float, float, float],
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """dE/ds_j on the table rows and the rows E depends on. s_c = sum c_j s_j (half the fit gain
    of each side at t_c), H+- = A+- - s_c and E = sqrt(H+ / H-), so
    dE/ds_j = E / 2 [(a+_j - c_j) / H+ - (a-_j - c_j) / H-]; NaN where E is."""
    e, drop, climb = restitution
    centre = np.zeros(t.size)
    for side in (bounce.before, bounce.after):
        np.add.at(centre, np.searchsorted(t, side.t), 0.5 * _fit_gain_at(side.t, t_c))
    tops = [np.zeros(t.size), np.zeros(t.size)]
    for top, rows, apex in zip(tops, runs, apexes, strict=True):
        np.add.at(top, rows, apex.weights)
    used = (centre != 0.0) | (tops[0] != 0.0) | (tops[1] != 0.0)
    if not math.isfinite(e):
        return np.full(t.size, math.nan), used
    return 0.5 * e * ((tops[1] - centre) / climb - (tops[0] - centre) / drop), used


def _camera_k(inputs: CheckInputs) -> FloatArray | None:
    """K of the camera the states were lifted with, None without one or off the camera frame."""
    if inputs.camera is None or inputs.states.frame is not CoordFrame.CAMERA_OPENCV:
        return None
    return np.asarray(inputs.camera.K, dtype=np.float64)


def _project(camera_k: FloatArray, positions: FloatArray) -> FloatArray:
    """Pixels (n, 2) of camera-frame points (n, 3)."""
    homogeneous = positions @ camera_k.T
    return np.asarray(homogeneous[:, :2] / homogeneous[:, 2:3], dtype=np.float64)


def _image_normal(camera_k: FloatArray, pixel: FloatArray, normal: FloatArray) -> FloatArray:
    """Unit image direction of J n at ``pixel``: J is (K[:2] - u K[2]) / z, so z drops out."""
    direction = (camera_k[:2] - np.outer(pixel, camera_k[2])) @ normal
    return np.asarray(direction / np.linalg.norm(direction), dtype=np.float64)


def _run_rows(
    inputs: CheckInputs, index: int, bounce: Bounce, params: RestitutionParams
) -> tuple[IntArray, IntArray]:
    """All flight rows of the runs that hold the fit rows on each side of the impact."""
    states, contacts = inputs.states, inputs.contacts
    assert contacts is not None
    margin = params.rows(inputs).margin
    fi = states.frame_index
    flight = flight_mask(str(states.object_ids[index]), contacts, fi, margin)
    finite = np.all(np.isfinite(states.position[:, index]), axis=-1)
    k = int(np.searchsorted(states.t, bounce.before.t[-1]))
    m = int(np.searchsorted(states.t, bounce.after.t[0]))
    before = side_rows(flight, finite, range(k, -1, -1), fi.shape[0])
    after = side_rows(flight, finite, range(m, fi.shape[0]), fi.shape[0])
    return before, after


def run_apex(t: FloatArray, s: FloatArray) -> Apex:
    """The highest row; an interior concave maximum is refined by the vertex of the parabola
    through it and its two neighbours. By the envelope theorem dA/ds is the Lagrange basis of
    the three rows at the vertex shift (clipped or not)."""
    i = int(np.argmax(s))
    weights = np.zeros(s.size)
    interior = 0 < i < s.size - 1
    if interior:
        a, b, c = np.polyfit(t[i - 1 : i + 2] - t[i], s[i - 1 : i + 2], 2)
        if a < 0.0:
            dt = min(max(-b / (2.0 * a), t[i - 1] - t[i]), t[i + 1] - t[i])
            weights[i - 1 : i + 2] = _lagrange_at(t[i - 1 : i + 2] - t[i], dt)
            return Apex(float(t[i] + dt), float(c + b * dt + a * dt * dt), weights, interior)
    weights[i] = 1.0
    return Apex(float(t[i]), float(s[i]), weights, interior)


def _lagrange_at(nodes: FloatArray, x: float) -> FloatArray:
    """Lagrange basis polynomials of three nodes evaluated at ``x``."""
    return np.array(
        [
            math.prod((x - nodes[m]) / (nodes[j] - nodes[m]) for m in range(3) if m != j)
            for j in range(3)
        ]
    )


def _fit_gain_at(t: FloatArray, t_c: float) -> FloatArray:
    """g with fit(t_c) = g . y for the unweighted degree-2 fit on times ``t``."""
    return poly_fit(t, np.eye(t.size), np.ones(1), 2).value_at(t_c)[0]


def _row_sigma_along(
    camera_k: FloatArray, positions: FloatArray, sigmas: FloatArray, d: FloatArray
) -> FloatArray:
    """sigma of u . d per row: |J^T d| weighted by the per-axis sigma, J = (K[:2] - u K[2]) / z."""
    pixels = _project(camera_k, positions)
    rows = camera_k[:2][None] - pixels[:, :, None] * camera_k[2][None, None]
    along = np.einsum("i,nij->nj", d, rows) / positions[:, 2:3]
    return np.asarray(np.sqrt(np.sum(along**2 * sigmas**2, axis=1)), dtype=np.float64)


def _tangent_bias(camera_k: FloatArray, bounce: Bounce, t_c: float) -> float:
    """eps = (J v_t) . d / (|v_n| |J n|) of the incoming velocity at the contact point (a
    ground-truth lift only: it needs the true depth)."""
    v_minus, _ = bounce.velocities(t_c)
    n = bounce.normal
    p_c = bounce.p_k + v_minus * (t_c - float(bounce.before.t[-1]))
    pixel = _project(camera_k, p_c[None])[0]
    jacobian = (camera_k[:2] - np.outer(pixel, camera_k[2])) / float(p_c[2])
    v_n = float(v_minus @ n)
    jn = jacobian @ n
    d = jn / np.linalg.norm(jn)
    return float((jacobian @ (v_minus - v_n * n)) @ d / (abs(v_n) * np.linalg.norm(jn)))


PLUGINS = (ImageApexRestitutionCheck,)
