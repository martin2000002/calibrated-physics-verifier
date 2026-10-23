"""R3a restitution of ground impacts (ADR 0006 3.4)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._collision_windows import (
    CollisionParams,
    FloatArray,
    SideData,
    Window,
    as_params,
    contact_span,
    fit_window,
    no_contacts,
    side_data,
)
from physverify.checks._results import abstain, unscored
from physverify.checks._uncertainty import check_rng, mc_sigma
from physverify.physics.geometry import Plane
from physverify.physics.measures.collision_energy import restitution
from physverify.physics.tracks import FitSource, MotionFit, body_length, native_span, support_plane
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable

# Slack (relative to the search interval) for a root that lands on its ends by rounding.
ROOT_TOL = 1e-9


class RestitutionParams(CollisionParams):
    primitives: tuple[str, ...] = ("sphere",)
    min_normal_speed: float = Field(default=0.5, ge=0.0)


def _real_roots(a2: float, a1: float, a0: float) -> list[float]:
    """Real roots of a2 x^2 + a1 x + a0 (stable form, degenerate cases included)."""
    if not all(math.isfinite(c) for c in (a2, a1, a0)):
        return []
    if a2 == 0.0:
        return [] if a1 == 0.0 else [-a0 / a1]
    disc = a1 * a1 - 4.0 * a2 * a0
    if disc < 0.0:
        return []
    q = -0.5 * (a1 + math.copysign(math.sqrt(disc), a1))
    return [0.0] if q == 0.0 else [q / a2, a0 / q]


def _first_root(a2: float, a1: float, a0: float, upper: float) -> float | None:
    tol = ROOT_TOL * upper
    inside = [r for r in _real_roots(a2, a1, a0) if -tol <= r <= upper + tol]
    if not inside:
        return None
    return min(max(min(inside), 0.0), upper)


class RestitutionCheck:
    """R3a [e_minus_1]: restitution e - 1 of each impact of a sphere with the environment."""

    check_id: ClassVar[str] = "R3a"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    magnitude_unit: ClassVar[str] = "e_minus_1"
    Params: ClassVar[type[BaseModel]] = RestitutionParams

    def __init__(self, params: BaseModel) -> None:
        self.params: RestitutionParams = as_params(RestitutionParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        contacts = inputs.contacts
        if contacts is None:
            return no_contacts(type(self), inputs)
        return [
            self._impact(inputs, contacts, r)
            for r in contacts.records
            if r.kind is ContactKind.IMPACT and r.object_b == ENV_ID
        ]

    def _impact(
        self, inputs: CheckInputs, contacts: ContactTable, record: ContactRecord
    ) -> CheckResult:
        states = inputs.states
        index = states.object_index(record.object_a)
        subject = (record.object_a, ENV_ID)
        primitive = str(states.primitive[index])
        if primitive not in self.params.primitives:
            scope = {"primitive": primitive}
            return abstain(type(self), inputs, subject, AbstainReason.OUT_OF_SCOPE, None, scope)
        window = fit_window(states, contacts, record, index, self.params)
        if not window.enough(self.params.min_side_frames):
            span = contact_span(states, window)
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, subject, reason, span, window.counts())
        bounce = _bounce(states, contacts, record, index, window, self.params.fit_source)
        span = native_span(states.frame_index, int(window.before[0]), int(window.after[-1]))
        t_c, method = bounce.impact_time(*bounce.fits())
        details: dict[str, Any] = {"t_c": t_c, "t_c_method": method}
        if t_c is None:
            return abstain(
                type(self), inputs, subject, AbstainReason.ILL_CONDITIONED, span, details
            )
        return self._score(inputs, subject, index, bounce, t_c, span, details)

    def _score(
        self,
        inputs: CheckInputs,
        subject: tuple[str, str],
        index: int,
        bounce: _Bounce,
        t_c: float,
        span: tuple[int, int],
        details: dict[str, Any],
    ) -> CheckResult:
        v_minus, v_plus = bounce.velocities(t_c)
        vn_minus, vn_plus = float(v_minus @ bounce.normal), float(v_plus @ bounce.normal)
        details |= {"v_minus_n": vn_minus, "v_plus_n": vn_plus}
        floor = self.params.min_normal_speed * body_length(inputs.states, index)
        if not abs(vn_minus) >= floor:
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details)
        e = restitution(v_minus, v_plus, bounce.normal)
        details["e"] = e
        rng = check_rng(self.params.seed, inputs.clip.clip_id, self.check_id)
        sigma = bounce.sigma(t_c, self.params.mc_draws, rng)
        return unscored(type(self), inputs, subject, e - 1.0, sigma, span, details)


@dataclass(frozen=True, eq=False)
class _Bounce:
    """One ground impact: the fit rows of each side, the rows k and m that bound the contact
    window, the ground plane (None without one) and the sphere radius."""

    source: FitSource
    before: SideData
    after: SideData
    p_k: FloatArray
    p_m: FloatArray
    normal: FloatArray
    plane: Plane | None
    radius: float

    def fits(self) -> tuple[MotionFit, MotionFit]:
        return self.before.fit(self.source), self.after.fit(self.source)

    def velocities(self, t_c: float) -> tuple[FloatArray, FloatArray]:
        before, after = self.fits()
        return before.velocity_at(t_c)[0], after.velocity_at(t_c)[0]

    def impact_time(self, before: MotionFit, after: MotionFit) -> tuple[float | None, str | None]:
        return self._root(self._arcs(before, after, (0.0, 0.0)))

    def _arcs(self, before: MotionFit, after: MotionFit, ends: tuple[Any, Any]) -> _Arcs:
        t_k, t_m = float(self.before.t[-1]), float(self.after.t[0])
        return _Arcs(
            p_k=self.p_k + ends[0],
            v_k=before.velocity_at(t_k)[0],
            a_minus=before.acceleration,
            p_m=self.p_m + ends[1],
            v_m=after.velocity_at(t_m)[0],
            a_plus=after.acceleration,
        )

    def _root(self, arcs: _Arcs) -> tuple[float | None, str | None]:
        """First time in [t_k, t_m] the incoming arc touches the ground plane, else where the
        incoming and outgoing arcs cross along the normal."""
        t_k, t_m = float(self.before.t[-1]), float(self.after.t[0])
        delta = t_m - t_k
        if self.plane is not None:
            n = self.plane.normal
            gap = float(n @ arcs.p_k) - self.plane.offset - self.radius
            root = _first_root(0.5 * float(n @ arcs.a_minus), float(n @ arcs.v_k), gap, delta)
            if root is not None:
                return t_k + root, "plane"
        n = self.normal
        root = _first_root(
            0.5 * float(n @ (arcs.a_minus - arcs.a_plus)),
            float(n @ (arcs.v_k - arcs.v_m + arcs.a_plus * delta)),
            float(n @ (arcs.p_k - arcs.p_m + arcs.v_m * delta - 0.5 * arcs.a_plus * delta**2)),
            delta,
        )
        return (None, None) if root is None else (t_k + root, "arcs")

    def sigma(self, t_c: float, draws: int, rng: np.random.Generator) -> float:
        """Monte Carlo over the noise of the fitted rows, with t_c re-estimated in each draw.
        Fitting positions, the noise of p_k and p_m is that of the rows they belong to; the
        legacy velocity fit keeps them exact."""
        sigmas = {"before": self.before.sigma, "after": self.after.sigma}
        if not all(bool(np.all(np.isfinite(s))) for s in sigmas.values()):
            return math.nan

        def model(noise: Mapping[str, FloatArray]) -> FloatArray:
            before = self.before.fit_draws(self.source, noise["before"])
            after = self.after.fit_draws(self.source, noise["after"])
            position = self.source == "position"
            ends = (noise["before"][:, -1], noise["after"][:, 0]) if position else (0.0, 0.0)
            arcs = self._arcs(before, after, ends)
            return np.array([self._restitution(arcs.draw(d), t_c) for d in range(draws)])

        return mc_sigma(model, sigmas, draws, rng)

    def _restitution(self, arcs: _Arcs, t_c: float) -> float:
        t_draw = self._root(arcs)[0]
        t = t_c if t_draw is None else t_draw
        t_k, t_m = float(self.before.t[-1]), float(self.after.t[0])
        v_minus = arcs.v_k + arcs.a_minus * (t - t_k)
        return restitution(v_minus, arcs.v_m + arcs.a_plus * (t - t_m), self.normal)


@dataclass(frozen=True, eq=False)
class _Arcs:
    """Constant-acceleration arcs around an impact: the incoming one at t_k and the outgoing
    one at t_m, optionally batched along a leading draw axis."""

    p_k: FloatArray
    v_k: FloatArray
    a_minus: FloatArray
    p_m: FloatArray
    v_m: FloatArray
    a_plus: FloatArray

    def draw(self, d: int) -> _Arcs:
        shape = self.v_k.shape
        return _Arcs(*(np.broadcast_to(getattr(self, f.name), shape)[d] for f in fields(self)))


def _bounce(
    states: ObjectStateTable,
    contacts: ContactTable,
    record: ContactRecord,
    index: int,
    window: Window,
    source: FitSource,
) -> _Bounce:
    k, m = int(window.before[-1]), int(window.after[0])
    return _Bounce(
        source=source,
        before=side_data(states, index, window.before, source),
        after=side_data(states, index, window.after, source),
        p_k=states.position[k, index],
        p_m=states.position[m, index],
        normal=np.asarray(record.normal, dtype=np.float64),
        plane=support_plane(record.object_a, replace(contacts, records=(record,))),
        radius=body_length(states, index) / 2.0,
    )


PLUGINS = (RestitutionCheck,)
