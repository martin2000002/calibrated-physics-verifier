"""R3b kinetic energy gain of object collisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._collision_windows import (
    CollisionParams,
    FloatArray,
    Window,
    contact_span,
    fit_window,
    no_contacts,
    side_data,
)
from physverify.checks._params import as_params
from physverify.checks._results import abstain, unscored
from physverify.checks._uncertainty import check_rng, mc_sigma
from physverify.physics.geometry import matrix_from_quaternion
from physverify.physics.measures.collision_energy import (
    incoming_energy,
    kinetic_energy_gain,
    momentum_mass_ratio,
)
from physverify.physics.tracks import FitSource, MotionFit, native_span, records_of, row_step
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable, PrimitiveKind


class EnergyGainParams(CollisionParams):
    cos_min: float = Field(default=0.9, ge=-1.0, le=1.0)
    dv_min_rel: float = Field(default=0.05, ge=0.0)
    equal_mass_phrases: tuple[str, ...] = ("identical", "same mass", "equal mass")


@dataclass(frozen=True, eq=False)
class _Sides:
    """Motion fits of one body before and after an impact."""

    window: Window
    before: MotionFit
    after: MotionFit


def _sides(states: ObjectStateTable, index: int, window: Window, source: FitSource) -> _Sides:
    before = side_data(states, index, window.before, source).fit(source)
    return _Sides(window, before, side_data(states, index, window.after, source).fit(source))


class EnergyGainCheck:
    """R3b [dE_over_E]: relative gain of translational kinetic energy in a two-body collision."""

    check_id: ClassVar[str] = "R3b"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "dE_over_E"
    Params: ClassVar[type[BaseModel]] = EnergyGainParams

    def __init__(self, params: BaseModel) -> None:
        self.params = as_params(EnergyGainParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        contacts = inputs.contacts
        if contacts is None:
            return no_contacts(type(self), inputs)
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
        windows = [fit_window(states, contacts, record, i, self.params) for i in indices]
        if not all(w.enough(self.params.min_side_frames) for w in windows):
            span = contact_span(states, windows[0])
            details = {name: w.counts() for name, w in zip(subject, windows, strict=True)}
            return abstain(type(self), inputs, subject, AbstainReason.TOO_FEW_FRAMES, span, details)
        source = self.params.fit_source
        sides = [_sides(states, i, w, source) for i, w in zip(indices, windows, strict=True)]
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
        fits = [(s.before.velocity_at(t_c), s.after.velocity_at(t_c)) for s in sides]
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
        incoming = None if ratio is None else incoming_energy(ratio, vm_i, vm_j)
        if ratio is None or not incoming:
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details)
        velocities = {"vi_minus": vm_i, "vj_minus": vm_j, "vi_plus": vp_i, "vj_plus": vp_j}
        sigmas = {"vi_minus": sm_i, "vj_minus": sm_j, "vi_plus": sp_i, "vj_plus": sp_j}
        magnitude = kinetic_energy_gain(ratio, vm_i, vm_j, vp_i, vp_j)
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


PLUGINS = (EnergyGainCheck,)
