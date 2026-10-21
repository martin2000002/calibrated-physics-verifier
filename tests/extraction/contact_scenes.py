"""World scenes for the contact providers, built from the check scenarios (tests/checks).

A box resting on the floor gives the proximity provider its ground plane (it needs an object at
rest), so most scenes here add one next to the moving bodies.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CHECKS = str(Path(__file__).resolve().parents[1] / "checks")
if CHECKS not in sys.path:
    sys.path.insert(0, CHECKS)

import check_scenarios as sc  # noqa: E402

from physverify.extraction.providers._common import centred_intrinsics  # noqa: E402
from physverify.schema.clip import FrameSampling, SamplingSpec, resolve_sampling  # noqa: E402
from physverify.schema.constants import STANDARD_GRAVITY  # noqa: E402
from physverify.schema.contacts import ContactTable  # noqa: E402
from physverify.schema.outputs import StageOutput  # noqa: E402
from physverify.schema.perception import CameraModel, IntrinsicsSource  # noqa: E402
from physverify.schema.stages import Stage  # noqa: E402
from physverify.schema.states import ObjectStateTable, PrimitiveKind  # noqa: E402

FloatArray = NDArray[np.float64]
BOX_SIDE = 0.2


def floor_box(x: float = 0.8, frames: int = sc.FRAMES, object_id: str = "box") -> sc.SceneObject:
    centre = (x, 0.0, BOX_SIDE / 2)
    return sc.still(object_id, PrimitiveKind.BOX, sc.cube(BOX_SIDE), centre, frames)


def bouncing(
    e: float = 0.8,
    impact_frame: float = sc.EVENT_FRAME,
    speed: float = 5.0,
    kind: PrimitiveKind = PrimitiveKind.SPHERE,
    frames: int = sc.FRAMES,
    object_id: str = "ball",
) -> sc.SceneObject:
    """The ball of ``check_scenarios.bounce``: it lands at ``impact_frame`` with ``speed`` m/s
    downwards and leaves with e times that, 0.5 m/s along x throughout."""
    g = STANDARD_GRAVITY * sc.DOWN
    v_in, v_out = np.array([0.5, 0.0, -speed]), np.array([0.5, 0.0, e * speed])
    anchor = np.array([0.0, 0.0, sc.BALL_RADIUS])
    t_c = impact_frame * sc.DT
    p, v, a = sc.piecewise_ballistic(sc.times(frames), t_c, anchor, (v_in, g), (v_out, g))
    return sc.SceneObject(object_id, kind, sc.cube(2 * sc.BALL_RADIUS), p, v, a, label=kind.value)


def rocking_box(
    tilt_deg: float = 5.0, period: int = 2, frames: int = sc.FRAMES, x: float = 0.0
) -> sc.SceneObject:
    """A cube on the floor that alternates every ``period`` frames between flat (four corners
    touching) and tilted about x by ``tilt_deg`` on one edge (two corners): the patch changes
    its point count while the normal stays the floor's."""
    half = BOX_SIDE / 2
    tilt = math.radians(tilt_deg)
    rotations, centres = [], []
    for frame in range(frames):
        tilted = (frame // period) % 2 == 1
        angle = tilt if tilted else 0.0
        c, s = math.cos(angle), math.sin(angle)
        rotations.append(np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]))
        centres.append((x, 0.0, half * c + half * s))
    still = sc.still("box", PrimitiveKind.BOX, sc.cube(BOX_SIDE), (x, 0.0, half), frames)
    return replace(still, position=np.array(centres), rotation=np.stack(rotations))


def rocking_case(tilt_deg: float = 5.0, period: int = 2, pitch_deg: float = 25.0) -> sc.Case:
    """``rocking_box`` expressed in the camera, with a per-frame orientation."""
    box = rocking_box(tilt_deg, period)
    first = replace(box, rotation=box.rotation[0])
    case = sc.build("rocking_box", [first], [], pitch_deg)
    quaternions = np.stack(
        [sc.quaternion_from_matrix(case.view.rotation @ r) for r in box.rotation]
    )
    position = case.view.point(box.position)
    states = replace(case.states, position=position[:, None, :], quaternion=quaternions[:, None, :])
    return replace(case, states=states)


def bounce_on_floor(e: float = 0.8, pitch_deg: float = 25.0, **ball: Any) -> sc.Case:
    return sc.build("bounce_on_floor", [bouncing(e, **ball), floor_box()], [], pitch_deg)


def collision_over_floor(
    mass_ratio: float = 1.0,
    gain: float = 0.0,
    pitch_deg: float = 25.0,
    impact_frame: float = sc.EVENT_FRAME,
) -> sc.Case:
    """``check_scenarios.collision`` (balls in the air, no gravity on them) beside a floor box."""
    case = sc.collision(
        mass_ratio=mass_ratio, gain=gain, impact_frame=impact_frame, pitch_deg=pitch_deg
    )
    box = floor_box(x=1.5)
    extra = sc.build("box_only", [box], [], pitch_deg).states
    return replace(case, states=joined(case.states, extra))


def joined(a: ObjectStateTable, b: ObjectStateTable) -> ObjectStateTable:
    """Objects of ``a`` then ``b`` (same rows, frame and gravity)."""
    fields = {
        name: np.concatenate([getattr(a, name), getattr(b, name)], axis=1) for name in PER_OBJECT_2D
    }
    fields.update(
        {name: np.concatenate([getattr(a, name), getattr(b, name)]) for name in PER_OBJECT_1D}
    )
    return replace(a, **fields)


PER_OBJECT_2D = (
    "position",
    "position_sigma",
    "velocity",
    "velocity_sigma",
    "acceleration",
    "acceleration_sigma",
    "angular_velocity",
    "quaternion",
    "orientation_sigma",
    "size",
    "size_sigma",
    "occlusion",
    "confidence",
    "visible",
    "in_frame",
)
PER_OBJECT_1D = ("object_ids", "labels", "primitive")


def jittered(states: ObjectStateTable, sigma: float, seed: int = 0) -> ObjectStateTable:
    """Positions with white noise of ``sigma`` and that sigma declared."""
    noise = np.random.default_rng(seed).normal(0.0, sigma, states.position.shape)
    return replace(
        states,
        position=states.position + noise,
        position_sigma=np.full(states.position_sigma.shape, sigma),
    )


FOCAL_PX = 500.0
"""Focal length of the test camera, in processing pixels."""


@dataclass(frozen=True)
class StubContext:
    """The part of a StageContext the CONTACTS providers read."""

    upstream: Mapping[Stage, StageOutput]
    sampling: FrameSampling

    def require(self, stage: Stage, kind: type[Any]) -> Any:
        output = self.upstream[stage]
        assert isinstance(output, kind)
        return output


def scene_camera(
    case: sc.Case, focal: float = FOCAL_PX, pose: FloatArray | None = None
) -> CameraModel:
    sampling = resolve_sampling(case.clip, SamplingSpec())
    return CameraModel(
        sampling=sampling,
        K=centred_intrinsics(focal, sampling),
        source=IntrinsicsSource.ASSUMED_FOV,
        fov_deg=60.0,
        fov_sigma_deg=0.0,
        world_from_camera=pose,
    )


def kinematic_table(
    case: sc.Case,
    states: ObjectStateTable | None = None,
    camera: CameraModel | None = None,
    seconds_per_frame: float | None = None,
    **params: Any,
) -> ContactTable:
    """``contacts/kinematic`` on the case's states (or ``states``) with the test camera. Imported
    here so that the golden modules can import this file with the code before it3 (audit L2)."""
    from physverify.extraction.providers.contacts_kinematic import KinematicContacts

    sampling = resolve_sampling(case.clip, SamplingSpec())
    if seconds_per_frame is not None:
        sampling = replace(sampling, seconds_per_frame=seconds_per_frame)
    ctx = StubContext(
        upstream={
            Stage.KINEMATICS: case.states if states is None else states,
            Stage.CAMERA: scene_camera(case) if camera is None else camera,
        },
        sampling=sampling,
    )
    provider = KinematicContacts(KinematicContacts.Params(**params))
    table = provider.run(ctx)  # type: ignore[arg-type]
    assert isinstance(table, ContactTable)
    return table
