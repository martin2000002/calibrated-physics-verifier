"""Contacts whose existence is kinematic, on a scene plane.

A RESTING contact needs a rest run of the chosen rest test (``_rest_existence.py``): by default
the image-plane one, a mask centroid compatible with standing still and constant under the
profile's TRACK centroid error (unknown: no rest at all), or the 3D velocity one under the
profile's velocity error. Under it lies another resting object (pair) or the scene plane
(environment). An optional gap gate keeps only environment rows where the body lies within
z(alpha / 2) sigma_gap of the plane, sigma_gap from the LIFT ``support_gap_rel`` entry; it is off
by default. An IMPACT needs a 3D velocity kink that explains its window best and a significant
change along the chosen surface normal. The scene plane has normal up from gravity and the height
of the lowest other object at rest or bouncing, never the judged body's own. Environment records
lie on the body's own bottom, so checks measure the body, not the plane it was tested against.
Pairs use the proximity geometry of ``contacts/proximity`` without a ground plane. See
``_velocity_tests.py``, ``_image_rest.py``, ``_scene_plane.py`` and ``_scene_records.py``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._impact_attribution import KinematicScene, camera_positions
from physverify.extraction.providers._impact_existence import IMPACT_TESTS
from physverify.extraction.providers._impacts import EDGE_SIDE_S
from physverify.extraction.providers._pair_rest import PAIR_RESTS
from physverify.extraction.providers._proximity import ContactScene, ProximityParams
from physverify.extraction.providers._rest_existence import (
    REST_TESTS,
    RestExistence,
    RestName,
)
from physverify.extraction.providers._scene_plane import GapError, HeightError
from physverify.extraction.providers._scene_records import (
    ImpactExistence,
    PairRest,
    SceneInputs,
    SceneOutcome,
    scene_contacts,
)
from physverify.extraction.providers._velocity_tests import VelocityError
from physverify.physics.row_time import reference_seconds
from physverify.schema.contacts import ContactTable
from physverify.schema.error_profile import CENTROID_REL, SIZE_REL, SUPPORT_GAP_REL, VEL_REL
from physverify.schema.perception import CameraModel
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


class Params(ProviderParams):
    tol_rel: float = Field(default=0.05, gt=0.0)
    min_rest_s: float = Field(default=reference_seconds(3), gt=0.0)
    normal_split_deg: float = Field(default=20.0, gt=0.0)
    side_s: float = Field(default=reference_seconds(4), ge=EDGE_SIDE_S)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    rest: RestName = "image"
    env_gap_gate: bool = False


def pair_params(params: Params) -> ProximityParams:
    """Proximity geometry for object pairs only; their kind comes from the rest test."""
    return ProximityParams(
        tol_rel=params.tol_rel,
        min_rest_s=params.min_rest_s,
        normal_split_deg=params.normal_split_deg,
        ground="none",
    )


class KinematicSceneContacts:
    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = "kinematic_scene"
    version: ClassVar[str] = "3"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.CAMERA, Stage.KINEMATICS)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.LIFT, Stage.KINEMATICS)
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ContactTable:
        states = ctx.require(Stage.KINEMATICS, ObjectStateTable)
        camera = ctx.require(Stage.CAMERA, CameraModel)
        velocity = VelocityError.of(ctx.error_entry(Stage.KINEMATICS, VEL_REL))
        heights = HeightError.of(
            ctx.error_entry(Stage.LIFT, CENTROID_REL), ctx.error_entry(Stage.LIFT, SIZE_REL)
        )
        gaps = GapError.of(ctx.error_entry(Stage.LIFT, SUPPORT_GAP_REL))
        seconds_per_frame = ctx.sampling.seconds_per_frame
        tests = (
            REST_TESTS[self.params.rest](ctx, states),
            IMPACT_TESTS["velocity"](ctx, states),
            PAIR_RESTS["proximity"](ctx, states),
        )
        outcome = self.outcome(states, camera, seconds_per_frame, velocity, heights, gaps, *tests)
        return ContactTable(
            clip_id=states.clip_id,
            frame=states.frame,
            unit=states.unit,
            source_key=states.source_key,
            records=() if outcome is None else outcome.records,
        )

    def outcome(
        self,
        states: ObjectStateTable,
        camera: CameraModel,
        seconds_per_frame: float,
        velocity: VelocityError,
        heights: HeightError,
        gaps: GapError,
        rest: RestExistence,
        impacts: ImpactExistence,
        pair_rest: PairRest,
    ) -> SceneOutcome | None:
        """The records and the tests behind them; None when sigma_v is unknown."""
        if not velocity.known:
            return None
        p = self.params
        pairs = ContactScene(states, pair_params(p), seconds_per_frame, split_on_point_count=False)
        intervals = tuple(pairs.intervals())
        scene = KinematicScene(
            states=states,
            proximity=pairs,
            records=tuple(pairs.record(pair, interval) for pair, interval in intervals),
            camera_positions=camera_positions(states, camera),
            seconds_per_frame=seconds_per_frame,
            tol_rel=p.tol_rel,
        )
        inputs = SceneInputs(
            scene=scene,
            intervals=intervals,
            rest=rest.mask(states, p.alpha, pairs.min_rest_rows),
            rest_known=rest.known,
            sigma_v=velocity.sigmas(states, pairs.lengths),
            height_error=heights,
            gap_error=gaps,
            alpha=p.alpha,
            focal=float(camera.K[0, 0]),
            side_s=p.side_s,
            env_gap_gate=p.env_gap_gate,
            impacts=impacts,
            pair_rest=pair_rest,
        )
        return scene_contacts(inputs)


PLUGINS = (KinematicSceneContacts,)
