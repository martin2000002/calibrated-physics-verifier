"""R3a restitution of ground impacts and R3b kinetic energy gain of object collisions (ADR 0006
3.4). Velocities on each side of an impact are line fits over the nearest flight rows,
evaluated at the impact time."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._common import CommonParams, abstain, check_rng, mc_sigma, unscored
from physverify.physics.geometry import matrix_from_quaternion
from physverify.physics.measures.r3_collision import (
    kinetic_energy_gain,
    momentum_mass_ratio,
    restitution,
)
from physverify.physics.tracks import (
    LineFit,
    body_length,
    contact_rows,
    flight_mask,
    line_fit,
    native_span,
    records_of,
    row_step,
    support_plane,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
# Slack (relative to the search interval) for a root that lands on its ends by rounding.
ROOT_TOL = 1e-9


class CollisionParams(CommonParams):
    impact_margin_frames: int = Field(default=0, ge=0)
    fit_frames: int = Field(default=4, ge=2)
    min_side_frames: int = Field(default=2, ge=2)


class RestitutionParams(CollisionParams):
    primitives: tuple[str, ...] = ("sphere",)
    min_normal_speed: float = Field(default=0.5, ge=0.0)


class EnergyGainParams(CollisionParams):
    cos_min: float = Field(default=0.9, ge=-1.0, le=1.0)
    dv_min_rel: float = Field(default=0.05, ge=0.0)
    equal_mass_phrases: tuple[str, ...] = ("identical", "same mass", "equal mass")


@dataclass(frozen=True, eq=False)
class _Window:
    """Rows of one body around an impact: in contact, and the fit rows before and after."""

    contact: IntArray
    before: IntArray
    after: IntArray

    def enough(self, min_rows: int) -> bool:
        return min(self.before.size, self.after.size) >= min_rows

    def counts(self) -> dict[str, int]:
        return {"rows_before": int(self.before.size), "rows_after": int(self.after.size)}


@dataclass(frozen=True, eq=False)
class _Sides:
    """Velocity fits of one body before and after an impact."""

    window: _Window
    before: LineFit
    after: LineFit


def _window(
    states: ObjectStateTable, contacts: ContactTable, record: ContactRecord, index: int, p: Any
) -> _Window:
    """Nearest ``fit_frames`` rows with a finite velocity in the flight runs around the record."""
    fi = states.frame_index
    margin = p.impact_margin_frames
    contact = np.flatnonzero(contact_rows(record, fi, margin))
    if contact.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return _Window(contact, empty, empty.copy())
    flight = flight_mask(str(states.object_ids[index]), contacts, fi, margin)
    finite = np.all(np.isfinite(states.velocity[:, index]), axis=-1)
    before = _side(flight, finite, range(int(contact[0]) - 1, -1, -1), p.fit_frames)
    after = _side(flight, finite, range(int(contact[-1]) + 1, fi.shape[0]), p.fit_frames)
    return _Window(contact, before, after)


def _side(
    flight: NDArray[np.bool_], finite: NDArray[np.bool_], walk: range, count: int
) -> IntArray:
    rows: list[int] = []
    for row in walk:
        if not flight[row] or len(rows) == count:
            break
        if finite[row]:
            rows.append(row)
    return np.array(sorted(rows), dtype=np.int64)


def _fit(states: ObjectStateTable, index: int, rows: IntArray) -> LineFit:
    return line_fit(
        states.t[rows], states.velocity[rows, index], states.velocity_sigma[rows, index]
    )


def _sides(states: ObjectStateTable, index: int, window: _Window) -> _Sides:
    return _Sides(window, _fit(states, index, window.before), _fit(states, index, window.after))


def _contact_span(states: ObjectStateTable, window: _Window) -> tuple[int, int] | None:
    if window.contact.size == 0:
        return None
    return native_span(states.frame_index, int(window.contact[0]), int(window.contact[-1]))


def _no_contacts(check: type[Any], inputs: CheckInputs) -> list[CheckResult]:
    subject = tuple(sorted(str(i) for i in inputs.states.object_ids))
    details = {"missing": "contacts"}
    return [abstain(check, inputs, subject, AbstainReason.MISSING_INPUT, None, details)]


def _params(cls: type[BaseModel], params: BaseModel) -> Any:
    return (
        params
        if isinstance(params, cls)
        else cls.model_validate(params.model_dump(exclude_unset=True))
    )


def _normal_sigma(sigma: FloatArray, normal: FloatArray) -> float:
    """Sigma of v . n from independent per-component sigmas."""
    return float(np.sqrt(np.sum((normal * sigma) ** 2)))


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
        self.params: RestitutionParams = _params(RestitutionParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        contacts = inputs.contacts
        if contacts is None:
            return _no_contacts(type(self), inputs)
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
        window = _window(states, contacts, record, index, self.params)
        if not window.enough(self.params.min_side_frames):
            span = _contact_span(states, window)
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, subject, reason, span, window.counts())
        sides = _sides(states, index, window)
        span = native_span(states.frame_index, int(window.before[0]), int(window.after[-1]))
        t_c, method = _impact_time(states, contacts, record, index, sides)
        details: dict[str, Any] = {"t_c": t_c, "t_c_method": method}
        if t_c is None:
            return abstain(
                type(self), inputs, subject, AbstainReason.ILL_CONDITIONED, span, details
            )
        return self._score(inputs, subject, record, index, sides, t_c, span, details)

    def _score(
        self,
        inputs: CheckInputs,
        subject: tuple[str, str],
        record: ContactRecord,
        index: int,
        sides: _Sides,
        t_c: float,
        span: tuple[int, int],
        details: dict[str, Any],
    ) -> CheckResult:
        n = np.asarray(record.normal, dtype=np.float64)
        v_minus, s_minus = sides.before.at(t_c)
        v_plus, s_plus = sides.after.at(t_c)
        vn_minus, vn_plus = float(v_minus @ n), float(v_plus @ n)
        details |= {"v_minus_n": vn_minus, "v_plus_n": vn_plus}
        floor = self.params.min_normal_speed * body_length(inputs.states, index)
        if not abs(vn_minus) >= floor:
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details)
        e = restitution(v_minus, v_plus, n)
        details["e"] = e
        # sigma_e^2 = (sigma_v+n / v-n)^2 + (e sigma_v-n / v-n)^2; the sigma of t_c is second order.
        sigma = math.hypot(_normal_sigma(s_plus, n), e * _normal_sigma(s_minus, n)) / abs(vn_minus)
        return unscored(type(self), inputs, subject, e - 1.0, sigma, span, details)


def _impact_time(
    states: ObjectStateTable,
    contacts: ContactTable,
    record: ContactRecord,
    index: int,
    sides: _Sides,
) -> tuple[float | None, str | None]:
    """First time in [t_k, t_m] the incoming arc touches the ground plane, else where the
    incoming and outgoing arcs cross along the normal (k = last row before, m = first after)."""
    k, m = int(sides.window.before[-1]), int(sides.window.after[0])
    t_k, t_m = float(states.t[k]), float(states.t[m])
    p_k, p_m = states.position[k, index], states.position[m, index]
    v_k, v_m = sides.before.at(t_k)[0], sides.after.at(t_m)[0]
    a_minus, a_plus = sides.before.slope, sides.after.slope
    delta = t_m - t_k
    plane = support_plane(record.object_a, replace(contacts, records=(record,)))
    if plane is not None:
        n = plane.normal
        radius = body_length(states, index) / 2.0
        gap = float(n @ p_k) - plane.offset - radius
        root = _first_root(0.5 * float(n @ a_minus), float(n @ v_k), gap, delta)
        if root is not None:
            return t_k + root, "plane"
    n = np.asarray(record.normal, dtype=np.float64)
    root = _first_root(
        0.5 * float(n @ (a_minus - a_plus)),
        float(n @ (v_k - v_m + a_plus * delta)),
        float(n @ (p_k - p_m + v_m * delta - 0.5 * a_plus * delta**2)),
        delta,
    )
    return (None, None) if root is None else (t_k + root, "arcs")


class EnergyGainCheck:
    """R3b [dE_over_E]: relative gain of translational kinetic energy in a two-body collision."""

    check_id: ClassVar[str] = "R3b"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    magnitude_unit: ClassVar[str] = "dE_over_E"
    Params: ClassVar[type[BaseModel]] = EnergyGainParams

    def __init__(self, params: BaseModel) -> None:
        self.params: EnergyGainParams = _params(EnergyGainParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        contacts = inputs.contacts
        if contacts is None:
            return _no_contacts(type(self), inputs)
        return [
            self._collision(inputs, contacts, r)
            for r in contacts.records
            if r.kind is ContactKind.IMPACT and r.object_b != ENV_ID
        ]

    def _collision(
        self, inputs: CheckInputs, contacts: ContactTable, record: ContactRecord
    ) -> CheckResult:
        states = inputs.states
        subject = (record.object_a, record.object_b)
        external = _external_contacts(states, contacts, record)
        if external:
            scope = {"external_contacts": external}
            return abstain(type(self), inputs, subject, AbstainReason.OUT_OF_SCOPE, None, scope)
        indices = (states.object_index(record.object_a), states.object_index(record.object_b))
        windows = [_window(states, contacts, record, i, self.params) for i in indices]
        if not all(w.enough(self.params.min_side_frames) for w in windows):
            span = _contact_span(states, windows[0])
            details = {name: w.counts() for name, w in zip(subject, windows, strict=True)}
            return abstain(type(self), inputs, subject, AbstainReason.TOO_FEW_FRAMES, span, details)
        sides = [_sides(states, i, w) for i, w in zip(indices, windows, strict=True)]
        first = min(int(w.before[0]) for w in windows)
        last = max(int(w.after[-1]) for w in windows)
        span = native_span(states.frame_index, first, last)
        return self._score(inputs, subject, record, indices, sides, span)

    def _score(
        self,
        inputs: CheckInputs,
        subject: tuple[str, str],
        record: ContactRecord,
        indices: tuple[int, int],
        sides: list[_Sides],
        span: tuple[int, int],
    ) -> CheckResult:
        spf = inputs.clip.seconds_per_frame
        t_c = record.t_start * spf
        fits = [(s.before.at(t_c), s.after.at(t_c)) for s in sides]
        (vm_i, sm_i), (vp_i, sp_i) = fits[0]
        (vm_j, sm_j), (vp_j, sp_j) = fits[1]
        ratio, cos, source = self._mass_ratio(inputs, vp_i - vm_i, vp_j - vm_j, vm_i, vm_j)
        q = row_step(inputs.states.frame_index) - 1
        details: dict[str, Any] = {
            "mass_ratio": ratio,
            "mass_ratio_source": source,
            "cos_antiparallel": cos,
            "t_c": t_c,
            "t_c_quantization_s": q * spf,
        }
        if ratio is None:
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details)
        velocities = {"vi_minus": vm_i, "vj_minus": vm_j, "vi_plus": vp_i, "vj_plus": vp_j}
        sigmas = {"vi_minus": sm_i, "vj_minus": sm_j, "vi_plus": sp_i, "vj_plus": sp_j}
        magnitude = kinetic_energy_gain(ratio, vm_i, vm_j, vp_i, vp_j)
        incoming = ratio * float(vm_i @ vm_i) + float(vm_j @ vm_j)
        spin = _spin_change(inputs.states, indices, sides, ratio)
        details["rot_energy_fraction"] = None if spin is None else spin / incoming
        fixed_ratio = ratio if source == "prompt" else None
        sigma = self._sigma(inputs, velocities, sigmas, fixed_ratio)
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)

    def _mass_ratio(
        self,
        inputs: CheckInputs,
        dv_i: FloatArray,
        dv_j: FloatArray,
        vm_i: FloatArray,
        vm_j: FloatArray,
    ) -> tuple[float | None, float | None, str]:
        """m_i / m_j from momentum when the impulses are clear and opposite, else the prompt."""
        p = self.params
        scale = max(float(np.linalg.norm(v)) for v in (vm_i, vm_j, vm_i - vm_j))
        smallest = min(float(np.linalg.norm(dv_i)), float(np.linalg.norm(dv_j)))
        cos = None
        if smallest > p.dv_min_rel * scale and smallest > 0.0:
            ratio, cos = momentum_mass_ratio(dv_i, dv_j)
            if cos >= p.cos_min:
                return ratio, cos, "momentum"
        prompt = (inputs.clip.prompt or "").lower()
        if any(phrase in prompt for phrase in p.equal_mass_phrases):
            return 1.0, cos, "prompt"
        return None, cos, "none"

    def _sigma(
        self,
        inputs: CheckInputs,
        velocities: Mapping[str, FloatArray],
        sigmas: Mapping[str, FloatArray],
        fixed_ratio: float | None,
    ) -> float:
        """Monte Carlo over the fitted velocities; the momentum mass ratio is redrawn too."""
        if not all(bool(np.all(np.isfinite(s))) for s in sigmas.values()):
            return math.nan

        def model(noise: Mapping[str, FloatArray]) -> FloatArray:
            draws = noise["vi_minus"].shape[0]
            drawn = ({k: velocities[k] + noise[k][d] for k in velocities} for d in range(draws))
            return np.array([_gain(v) for v in drawn])

        def _gain(v: Mapping[str, FloatArray]) -> float:
            dv_i, dv_j = v["vi_plus"] - v["vi_minus"], v["vj_plus"] - v["vj_minus"]
            ratio = fixed_ratio if fixed_ratio is not None else momentum_mass_ratio(dv_i, dv_j)[0]
            return kinetic_energy_gain(
                ratio, v["vi_minus"], v["vj_minus"], v["vi_plus"], v["vj_plus"]
            )

        rng = check_rng(self.params.seed, inputs.clip.clip_id, self.check_id)
        return mc_sigma(model, dict(sigmas), self.params.mc_draws, rng)


def _external_contacts(
    states: ObjectStateTable, contacts: ContactTable, record: ContactRecord
) -> list[str]:
    """ENV records of either body whose widened native window overlaps the collision."""
    q = row_step(states.frame_index) - 1
    hits = []
    for name in (record.object_a, record.object_b):
        for env in records_of(name, contacts):
            if env.object_b != ENV_ID:
                continue
            lo = math.floor(env.t_start) - 1 - q
            hi = math.ceil(env.t_end) + 1 + q
            if lo <= record.t_end and record.t_start <= hi:
                hits.append(f"{name}|{env.kind.value}|{env.t_start:g}-{env.t_end:g}")
    return hits


def _spin_change(
    states: ObjectStateTable, indices: tuple[int, int], sides: list[_Sides], ratio: float
) -> float | None:
    """Change of rotational energy (masses in m_j, without the common 1/2 of R3b's E).

    omega comes from the nearest row on each side of the impact.
    """
    change = 0.0
    for index, mass, side in zip(indices, (ratio, 1.0), sides, strict=True):
        before = _spin_energy(states, index, int(side.window.before[-1]))
        after = _spin_energy(states, index, int(side.window.after[0]))
        if before is None or after is None:
            return None
        change += mass * (after - before)
    return change


def _spin_energy(states: ObjectStateTable, index: int, row: int) -> float | None:
    """omega^T I omega per unit mass, with I the primitive's inertia in its body frame."""
    omega = states.angular_velocity[row, index]
    size = states.size[:, index]
    known = np.all(np.isfinite(size), axis=-1)
    if not (bool(np.all(np.isfinite(omega))) and bool(known.any())):
        return None
    kind = PrimitiveKind(str(states.primitive[index]))
    inertia = _unit_inertia(kind, np.median(size[known], axis=0))
    if kind is not PrimitiveKind.SPHERE:
        quaternion = states.quaternion[row, index]
        if not bool(np.all(np.isfinite(quaternion))):
            return None
        omega = matrix_from_quaternion(quaternion).T @ omega
    return float(omega**2 @ inertia)


def _unit_inertia(kind: PrimitiveKind, extents: FloatArray) -> FloatArray:
    """Principal moments per unit mass of a solid sphere, cylinder (axis z) or box."""
    ex, ey, ez = (float(e) for e in extents)
    if kind is PrimitiveKind.SPHERE:
        return np.full(3, 0.4 * (ex / 2.0) ** 2)
    if kind is PrimitiveKind.CYLINDER:
        r2 = (ex / 2.0) ** 2
        side = r2 / 4.0 + ez**2 / 12.0
        return np.array([side, side, r2 / 2.0])
    return np.array([ey**2 + ez**2, ex**2 + ez**2, ex**2 + ey**2]) / 12.0


PLUGINS = (RestitutionCheck, EnergyGainCheck)
