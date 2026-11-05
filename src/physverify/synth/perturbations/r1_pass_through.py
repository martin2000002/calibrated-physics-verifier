"""R1_dyn: a rolling sphere passes through the body it would have hit (ADR 0047 section 1).

From the last row before the first object-object contact the mover keeps rolling at constant
velocity as if its partner were absent, every other body keeps its pose, and the whole mover
path is offset horizontally, perpendicular to the relative velocity, so that the peak
penetration ratio of the pair equals the level target.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from physverify.physics.geometry import matrix_from_quaternion, quaternion_from_matrix
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.ballistics import rolling_angular_velocity
from physverify.synth.frame_windows import rest_mask, solid_at
from physverify.synth.magnitudes import bounding_radius, pair_penetration, penetration_between
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._penetration import MIN_CLEARANCE_M, PEN_LEVELS, SOLVE_TOL
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    before,
    edit_object,
    keep_records,
    provisional_record,
    radius_of,
    require_clearance,
    require_in_image,
    rotation_about,
    shift_records,
    translated,
)
from physverify.synth.placement import UP, solve_offset
from physverify.synth.settings import PerturbationCommon

NAME = "r1_pass_through"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = PEN_LEVELS
    min_clearance_m: float = MIN_CLEARANCE_M
    image_margin_frames: int = 2


class PassThrough(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R1_dyn",)
    magnitude_unit: ClassVar[str] = "pen_over_size"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f5_contact",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        mover, partner, start = first_rolling_hit(states, contacts)
        free = absent_partner(states, mover, start)
        direction = impact_direction(free, mover, partner, rng)
        reach = bounding_radius(solid_at(free, mover, 0)) + bounding_radius(
            solid_at(free, partner, 0)
        )
        shift = solve_offset(
            lambda s: pair_penetration(translated(free, mover, 0, s * direction), mover, partner),
            magnitude,
            0.0,
            reach,
            tol=SOLVE_TOL,
        )
        offset = shift * direction
        moved = translated(free, mover, 0, offset)
        self._require_valid(scene, moved, mover, partner)
        ids = states.object_ids.astype(str)
        last_free = math.nextafter(float(states.frame_index[start]), math.inf)
        kept = keep_records(contacts, before(ids.tolist(), last_free))
        record = provisional_record(
            self,
            magnitude,
            (str(ids[partner]), str(ids[mover])),
            (0, states.n_frames - 1),
            {"moved": str(ids[mover]), "free_from_frame": start, "offset_m": offset.tolist()},
        )
        return moved, shift_records(kept, str(ids[mover]), offset), record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        a, b = (states.object_index(o) for o in record.target_objects)
        return pair_penetration(states, a, b)

    def _require_valid(
        self, scene: SceneSpec, moved: ObjectStateTable, mover: int, partner: int
    ) -> None:
        require_clearance(
            moved,
            mover,
            range(moved.n_frames),
            self.params.min_clearance_m,
            f"{NAME}: the shifted mover comes too close to a third body",
            skip=(partner,),
        )
        overlap = overlap_frames(moved, mover, partner)
        margin = self.params.image_margin_frames
        window = slice(max(0, overlap[0] - margin), min(moved.n_frames, overlap[-1] + margin + 1))
        require_in_image(
            scene,
            moved.position[window][:, [mover, partner]],
            f"{NAME}: a pair centre leaves the image around the overlap",
        )


def first_rolling_hit(states: ObjectStateTable, contacts: ContactTable) -> tuple[int, int, int]:
    """(mover, partner, k0): the first object-object contact of a moving body, if that body is a
    sphere whose partner and every other body rest until then; k0 is the last row before the
    contact starts. Contacts between bodies at rest (a resting cluster) are not hits."""
    ids = states.object_ids.astype(str).tolist()
    hits = sorted(
        (r for r in contacts.records if ENV_ID not in (r.object_a, r.object_b)),
        key=lambda r: r.t_start,
    )
    for record in hits:
        start = int(np.searchsorted(states.frame_index, record.t_start, side="left")) - 1
        pair = (ids.index(record.object_a), ids.index(record.object_b))
        if start < 0 or all(bool(rest_mask(states, j)[start]) for j in pair):
            continue
        for mover, partner in (pair, pair[::-1]):
            if _rolls_into(states, mover, start):
                return mover, partner, start
        break
    raise PerturbationNotApplicableError(f"{NAME}: no rolling sphere hits a resting body")


def _rolls_into(states: ObjectStateTable, mover: int, start: int) -> bool:
    if str(states.primitive[mover]) != PrimitiveKind.SPHERE.value:
        return False
    if bool(rest_mask(states, mover)[start]):
        return False
    return all(
        bool(np.all(rest_mask(states, j)[: start + 1]))
        for j in range(states.n_objects)
        if j != mover
    )


def absent_partner(states: ObjectStateTable, mover: int, start: int) -> ObjectStateTable:
    """From row ``start`` on: the mover rolls at its horizontal velocity there, the rest stand."""
    frames = slice(start, states.n_frames)
    tau = states.t[frames] - states.t[start]
    velocity = states.velocity[start, mover] - float(states.velocity[start, mover] @ UP) * UP
    omega = rolling_angular_velocity(velocity, radius_of(states, mover))
    rate = float(np.linalg.norm(omega))
    base = matrix_from_quaternion(states.quaternion[start, mover])
    turns = [rotation_about(omega, rate * float(s)) @ base if rate > 0.0 else base for s in tau]
    out = edit_object(
        states,
        mover,
        frames,
        position=states.position[start, mover] + tau[:, None] * velocity,
        quaternion=np.stack([quaternion_from_matrix(r) for r in turns]),
        velocity=velocity,
        acceleration=0.0,
        angular_velocity=omega,
    )
    for j in range(states.n_objects):
        if j != mover:
            out = edit_object(
                out,
                j,
                frames,
                position=states.position[start, j],
                quaternion=states.quaternion[start, j],
                velocity=0.0,
                acceleration=0.0,
                angular_velocity=0.0,
            )
    return out


def impact_direction(
    states: ObjectStateTable, mover: int, partner: int, rng: np.random.Generator
) -> Vec:
    """Horizontal unit vector perpendicular to the relative velocity at the closest approach;
    its sign is the plugin's single draw from ``rng``."""
    gaps = np.linalg.norm(states.position[:, mover] - states.position[:, partner], axis=1)
    closest = int(np.argmin(gaps))
    relative = states.velocity[closest, mover] - states.velocity[closest, partner]
    side = np.cross(UP, relative)
    norm = float(np.linalg.norm(side))
    if not math.isfinite(norm) or norm < 1e-9:
        raise PerturbationNotApplicableError(f"{NAME}: no horizontal relative motion")
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    out: Vec = sign * side / norm
    return out


def overlap_frames(states: ObjectStateTable, a: int, b: int) -> list[int]:
    frames = [
        t
        for t in range(states.n_frames)
        if penetration_between(solid_at(states, a, t), solid_at(states, b, t)) > 0.0
    ]
    if not frames:
        raise PerturbationNotApplicableError(f"{NAME}: the pair never overlaps")
    return frames


PLUGINS = (PassThrough,)
