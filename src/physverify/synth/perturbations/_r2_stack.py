"""Shared by the R2 edits: the top body of a stack shifted along a horizontal face normal of its
support, checked for clearance and for a support face too tilted to hold it."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from physverify.physics.contact_manifold import contact_manifold
from physverify.physics.geometry import Solid, characteristic_length
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.frame_windows import (
    on_top_pairs,
    solid_at,
    supports_nothing,
    touching,
)
from physverify.synth.magnitudes import overhang_on
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    half_width,
    horizontal_face_normals,
    object_friction,
    pick,
    provisional_record,
    require_clear_shift,
    shift_records,
    translated,
)
from physverify.synth.placement import UP
from physverify.synth.settings import PerturbationCommon


class StackParams(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
    min_clearance_m: float = 0.01
    # Largest centre-of-pressure shift of a tilted face, as a share of the requested overhang.
    tilt_share: float = 0.5


class StackShift(PluginBase[StackParams]):
    """Picks (top, support, normal) and shifts the top by ``_shift``; subclasses set the target."""

    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str]
    target_checks: ClassVar[tuple[str, ...]]
    magnitude_unit: ClassVar[str] = "overhang_over_size"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.INJECTED_PARAMETER
    families: ClassVar[tuple[str, ...]] = ("f4_stack",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        candidates = top_bodies(contacts)
        if not candidates:
            raise PerturbationNotApplicableError(
                f"{self.name}: no body rests on exactly one movable body alone"
            )
        top_id, support_id = pick(rng, candidates)
        top, support = states.object_index(top_id), states.object_index(support_id)
        mid = states.n_frames // 2
        direction = pick(rng, horizontal_face_normals(states, support, mid))
        offset = self._shift(states, top, support, direction, magnitude) * direction
        moved = translated(states, top, 0, offset)
        self._check(scene, moved, top, support, magnitude)
        record = provisional_record(
            self,
            magnitude,
            (top_id,),
            (0, states.n_frames - 1),
            {"support": support_id, "offset_m": offset.tolist()},
        )
        return moved, shift_records(contacts, top_id, offset), record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        top = states.object_index(record.target_objects[0])
        support = states.object_index(str(record.params["support"]))
        mid = states.n_frames // 2
        return overhang_on(
            solid_at(states, top, mid), [solid_at(states, support, mid)], self.params.contact_tol_m
        )

    def _shift(
        self, states: ObjectStateTable, top: int, support: int, direction: Vec, magnitude: float
    ) -> float:
        raise NotImplementedError

    def _check(
        self, scene: SceneSpec, moved: ObjectStateTable, top: int, support: int, magnitude: float
    ) -> None:
        require_clear_shift(
            scene,
            moved,
            top,
            support,
            self.params.min_clearance_m,
            with_floor=True,
            too_close=f"{self.name}: the shifted body comes too close to another body",
            leaves=f"{self.name}: the shifted body leaves the image",
        )
        mid = moved.n_frames // 2
        ids = moved.object_ids.astype(str)
        friction = object_friction(scene, str(ids[top])) * object_friction(scene, str(ids[support]))
        shift = tilt_shift(
            solid_at(moved, top, mid),
            solid_at(moved, support, mid),
            friction,
            self.params.contact_tol_m,
        )
        if shift >= self.params.tilt_share * magnitude * characteristic_length(
            moved.size[mid, top]
        ):
            raise PerturbationNotApplicableError(f"{self.name}: the support face is too tilted")


def shift_span(
    states: ObjectStateTable, top: int, support: int, direction: Vec, tol: float
) -> tuple[float, float]:
    """Shifts along ``direction`` that bring the top's centre over the support's centre and just
    short of its far edge (both clipped by ``tol``)."""
    mid = states.n_frames // 2
    lower, upper = solid_at(states, support, mid), solid_at(states, top, mid)
    already = float((states.position[mid, top] - states.position[mid, support]) @ direction)
    reach = half_width(lower, direction) + half_width(upper, direction) - already - 2.0 * tol
    return -already, reach


def overhang_after(
    states: ObjectStateTable, top: int, support: int, direction: Vec, tol: float
) -> Callable[[float], float]:
    mid = states.n_frames // 2
    lower = solid_at(states, support, mid)

    def overhang(shift: float) -> float:
        return overhang_on(solid_at(states, top, mid, shift * direction), [lower], tol)

    return overhang


def top_bodies(contacts: ContactTable) -> list[tuple[str, str]]:
    """(T, B): T rests on exactly one movable body B, nothing rests on T and T touches only B."""
    pairs = on_top_pairs(contacts)
    uppers = sorted({up for up, _ in pairs})
    found: list[tuple[str, str]] = []
    for top in uppers:
        lowers = {low for up, low in pairs if up == top}
        if len(lowers) != 1 or ENV_ID in lowers or not supports_nothing(top, contacts):
            continue
        if touching(top, contacts) == lowers:
            found.append((top, next(iter(lowers))))
    return found


def tilt_shift(upper: Solid, lower: Solid, friction: float, tol: float) -> float:
    """mu d_P sin(theta): bound on the pressure-centre shift of a support face tilted by theta."""
    manifold = contact_manifold(upper, lower, tol)
    if manifold is None:
        return math.inf
    normal = np.mean(manifold.normals, axis=0)
    cos_theta = abs(float(normal @ UP)) / float(np.linalg.norm(normal))
    points = manifold.points
    diameter = max(float(np.linalg.norm(p - q)) for p in points for q in points)
    return friction * diameter * math.sqrt(max(0.0, 1.0 - cos_theta**2))
