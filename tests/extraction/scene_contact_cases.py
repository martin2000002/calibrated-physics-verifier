"""Scenes and a profiled context for ``contacts/kinematic_scene``.

Boxes stand on a world floor (z up) and are seen by the check-scenario camera, so the provider
only learns "up" from ``gravity_dir``. The error entries are the measured ones of the profile,
except the gap error, a round test value; masks are discs at the projected centres.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import contact_scenes as cs
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from support import entry

from physverify.extraction.providers._rest_existence import RestExistence, VelocityRest
from physverify.extraction.providers._scene_plane import GapError, HeightError
from physverify.extraction.providers._scene_records import SceneOutcome
from physverify.extraction.providers._velocity_tests import VelocityError
from physverify.extraction.providers.contacts_kinematic_scene import KinematicSceneContacts
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.clip import FrameSampling, SamplingSpec, resolve_sampling
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.error_profile import CENTROID_PX_REL, SUPPORT_GAP_REL, ErrorEntry
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

sc = cs.sc
FloatArray = NDArray[np.float64]
VEL_REL = entry(Stage.KINEMATICS, "savgol", "vel_rel", 0.3157, bias=-0.0022)
CENTROID_REL = entry(Stage.LIFT, "backproject", "centroid_rel", 0.0552, bias=0.0002)
SIZE_REL = entry(Stage.LIFT, "backproject", "size_rel", 0.0782, bias=0.0056)
GAP_REL = entry(Stage.LIFT, "backproject", SUPPORT_GAP_REL, 0.1)
CENTROID_PX = entry(Stage.TRACK, "edgetam", CENTROID_PX_REL, 0.3526)
VELOCITY = VelocityError.of(VEL_REL)
HEIGHTS = HeightError.of(CENTROID_REL, SIZE_REL)
GAPS = GapError.of(GAP_REL)
SIDE = cs.BOX_SIDE
LENGTH = float(characteristic_lengths(sc.cube(SIDE)))
SPACING = 2.0 * SIDE


def sampling(case: sc.Case) -> FrameSampling:
    return resolve_sampling(case.clip, SamplingSpec())


def outcome(
    case: sc.Case,
    states: ObjectStateTable | None = None,
    velocity: VelocityError = VELOCITY,
    heights: HeightError = HEIGHTS,
    gaps: GapError = GAPS,
    rest: RestExistence | None = None,
    **params: Any,
) -> SceneOutcome:
    """The provider's outcome with the 3D velocity rest unless ``rest`` is given."""
    provider = KinematicSceneContacts(KinematicSceneContacts.Params(**params))
    found = provider.outcome(
        case.states if states is None else states,
        cs.scene_camera(case),
        sampling(case).seconds_per_frame,
        velocity,
        heights,
        gaps,
        VelocityRest(velocity) if rest is None else rest,
    )
    assert found is not None
    return found


def measured_entries() -> dict[tuple[Stage, str], ErrorEntry]:
    measured = (VEL_REL, CENTROID_REL, SIZE_REL, GAP_REL, CENTROID_PX)
    return {(e.stage, e.quantity): e for e in measured}


def scene_table(
    case: sc.Case,
    entries: Mapping[tuple[Stage, str], ErrorEntry] | None = None,
    states: ObjectStateTable | None = None,
    track: MaskTrack | None = None,
    **params: Any,
) -> ContactTable:
    """The provider run on the case with disc masks of its (or ``states``') objects."""
    states = case.states if states is None else states
    camera = cs.scene_camera(case)
    ctx = cs.StubContext(
        upstream={
            Stage.TRACK: cs.disk_track(states, sampling(case)) if track is None else track,
            Stage.KINEMATICS: states,
            Stage.CAMERA: camera,
        },
        sampling=sampling(case),
        entries=measured_entries() if entries is None else entries,
    )
    provider = KinematicSceneContacts(KinematicSceneContacts.Params(**params))
    table = provider.run(ctx)  # type: ignore[arg-type]
    assert isinstance(table, ContactTable)
    return table


def box(name: str, x: float, bottom: float) -> sc.SceneObject:
    centre = (x, 0.0, bottom + SIDE / 2.0)
    return sc.still(name, PrimitiveKind.BOX, sc.cube(SIDE), centre)


def boxes_at(bottoms: Sequence[float], pitch_deg: float = 25.0) -> sc.Case:
    """Still boxes ``box<i>`` side by side, apart, with their bottoms at the given heights."""
    objects = [box(f"box{i}", SPACING * i, b) for i, b in enumerate(bottoms)]
    return sc.build("boxes", objects, [], pitch_deg)


def stacked_boxes(pitch_deg: float = 25.0) -> sc.Case:
    """box0 on the floor and box1 resting on it."""
    return sc.build("stack", [box("box0", 0.0, 0.0), box("box1", 0.0, SIDE)], [], pitch_deg)


def world_plane(case: sc.Case, height: float = 0.0) -> tuple[FloatArray, float]:
    """Normal and offset, in the case's camera frame, of the world plane z = height."""
    up = case.view.vector(sc.UP)
    return up, float(up @ case.view.point(np.array([0.0, 0.0, height])))


def env_records(records: Sequence[ContactRecord], object_id: str | None = None) -> list[Any]:
    return [
        r
        for r in records
        if r.object_b == ENV_ID and (object_id is None or r.object_a == object_id)
    ]


def pair_records(records: Sequence[ContactRecord]) -> list[ContactRecord]:
    return [r for r in records if r.object_b != ENV_ID]


def gate(length: float = LENGTH, gaps: GapError = GAPS, alpha: float = 0.05) -> float:
    """K sigma_gap with K = z(alpha / 2): the farthest a body may sit from its plane."""
    return float(norm.isf(alpha / 2.0)) * gaps.sigma(length)


def midway_offsets(record: ContactRecord) -> FloatArray:
    """up . p - d / 2 of every point: the plane offset ``support_plane`` reads."""
    assert record.point_distances is not None
    offsets: FloatArray = record.points @ record.normal - record.point_distances / 2.0
    return offsets


def without_gravity(states: ObjectStateTable) -> ObjectStateTable:
    return replace(states, gravity_dir=np.full(3, math.nan), gravity_dir_sigma=math.nan)
