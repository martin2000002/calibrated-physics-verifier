"""AC4: the eight conditional ground-truth providers on the analytic fixture."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from support import make_context, run_provider
from synthetic_clip import SyntheticClip

from physverify.extraction._base import StageProvider
from physverify.extraction.providers.gt_camera import GroundTruthCamera
from physverify.extraction.providers.gt_contacts import GroundTruthContacts
from physverify.extraction.providers.gt_depth import GroundTruthDepth
from physverify.extraction.providers.gt_detect import GroundTruthDetect
from physverify.extraction.providers.gt_kinematics import GroundTruthKinematics
from physverify.extraction.providers.gt_lift import GroundTruthLift
from physverify.extraction.providers.gt_scale import GroundTruthScale
from physverify.extraction.providers.gt_track import GroundTruthTrack
from physverify.schema.contacts import ContactTable
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel, DepthKind, DepthMap, Detections, MaskTrack
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

SCALE = 7.3
LIFT_KEY = "lift-key"
REQUIRES: dict[type[StageProvider], tuple[Stage, ...]] = {
    GroundTruthDetect: (),
    GroundTruthTrack: (Stage.DETECT,),
    GroundTruthDepth: (),
    GroundTruthCamera: (),
    GroundTruthLift: (Stage.TRACK,),
    GroundTruthScale: (Stage.TRACK, Stage.LIFT),
    GroundTruthKinematics: (Stage.TRACK, Stage.LIFT),
    GroundTruthContacts: (Stage.TRACK, Stage.LIFT),
}
SIGMA_FIELDS = ("position_sigma", "size_sigma", "velocity_sigma", "acceleration_sigma")


@dataclass(frozen=True)
class Chain:
    outputs: dict[Stage, StageOutput]
    metric_lift: ObjectStateTable

    def __getitem__(self, stage: Stage) -> StageOutput:
        return self.outputs[stage]


@pytest.fixture(scope="module")
def chain(fixture_clip: SyntheticClip) -> Chain:
    """Truth DETECT..LIFT, then SCALE/KINEMATICS/CONTACTS over a LIFT equal to truth / 7.3."""
    outputs: dict[Stage, StageOutput] = {}
    perception: tuple[type[StageProvider], ...] = (
        GroundTruthDetect,
        GroundTruthTrack,
        GroundTruthDepth,
        GroundTruthCamera,
        GroundTruthLift,
    )
    for provider in perception:
        ctx = make_context(fixture_clip, upstream=outputs, gt=fixture_clip.ground_truth)
        outputs[provider.stage] = run_provider(provider, ctx)
    metric = outputs[Stage.LIFT]
    assert isinstance(metric, ObjectStateTable)
    outputs[Stage.LIFT] = replace(
        metric,
        unit=LengthUnit.RELATIVE,
        source_key=LIFT_KEY,
        position=metric.position / SCALE,
        size=metric.size / SCALE,
    )
    for provider in (GroundTruthScale, GroundTruthKinematics, GroundTruthContacts):
        ctx = make_context(fixture_clip, upstream=outputs, gt=fixture_clip.ground_truth)
        outputs[provider.stage] = run_provider(provider, ctx)
    return Chain(outputs=outputs, metric_lift=metric)


def test_requires_match_the_design() -> None:
    for provider, requires in REQUIRES.items():
        assert provider.requires == requires, provider.__name__
        assert provider.is_ground_truth and not provider.heavy
        assert provider.error_profile_reads == ()


def test_outputs_carry_the_context_sampling(chain: Chain, fixture_clip: SyntheticClip) -> None:
    sampling = make_context(fixture_clip).sampling
    for stage in (Stage.DETECT, Stage.TRACK, Stage.DEPTH, Stage.CAMERA):
        output = chain[stage]
        assert isinstance(output, (Detections, MaskTrack, DepthMap, CameraModel))
        assert output.sampling == sampling, stage


def test_truth_sigma_is_zero_except_the_scale(chain: Chain) -> None:
    camera = chain[Stage.CAMERA]
    assert isinstance(camera, CameraModel)
    assert camera.fov_sigma_deg == 0.0
    kinematics = chain[Stage.KINEMATICS]
    assert isinstance(kinematics, ObjectStateTable)
    for table in (chain.metric_lift, kinematics):
        visible = table.visible
        for name in SIGMA_FIELDS:
            values = getattr(table, name)[visible]
            assert bool((values == 0.0).all()), name
    scale = chain[Stage.SCALE]
    assert isinstance(scale, ScaleEstimate)
    assert np.isfinite(scale.sigma)
    assert scale.meters_per_unit == pytest.approx(SCALE, abs=1e-9)
    assert scale.lift_key == LIFT_KEY


def test_one_object_seen_in_one_frame_leaves_the_truth_scale_unknown(
    chain: Chain, fixture_clip: SyntheticClip
) -> None:
    lift = chain[Stage.LIFT]
    assert isinstance(lift, ObjectStateTable)
    t, i = np.argwhere(lift.visible)[0]
    visible = np.zeros_like(lift.visible)
    visible[t, i] = True
    upstream = {**chain.outputs, Stage.LIFT: replace(lift, visible=visible)}
    ctx = make_context(fixture_clip, upstream=upstream, gt=fixture_clip.ground_truth)
    scale = run_provider(GroundTruthScale, ctx)
    assert isinstance(scale, ScaleEstimate)
    assert scale.source is ScaleSource.NONE and scale.detail == "single_ratio"
    assert scale.meters_per_unit is None and np.isnan(scale.sigma)
    assert scale.lift_key == LIFT_KEY


def test_kinematics_truth_uses_the_conditional_fit(chain: Chain) -> None:
    lift, kinematics = chain[Stage.LIFT], chain[Stage.KINEMATICS]
    assert isinstance(lift, ObjectStateTable) and isinstance(kinematics, ObjectStateTable)
    assert kinematics.unit is LengthUnit.RELATIVE
    assert kinematics.frame is CoordFrame.CAMERA_OPENCV
    assert kinematics.source_key == LIFT_KEY
    visible = kinematics.visible
    np.testing.assert_allclose(
        kinematics.position[visible], lift.position[visible], rtol=0, atol=1e-9
    )


def test_contacts_truth_uses_the_conditional_fit(chain: Chain, fixture_clip: SyntheticClip) -> None:
    contacts = chain[Stage.CONTACTS]
    assert isinstance(contacts, ContactTable)
    truth = fixture_clip.ground_truth.contacts(CoordFrame.CAMERA_OPENCV)
    assert contacts.unit is LengthUnit.RELATIVE and contacts.source_key == LIFT_KEY
    assert len(contacts.records) == len(truth.records) == 1
    np.testing.assert_allclose(
        contacts.records[0].points, truth.records[0].points / SCALE, rtol=0, atol=1e-9
    )


def test_relative_truth_depth_has_unit_object_median(chain: Chain) -> None:
    depth, track = chain[Stage.DEPTH], chain[Stage.TRACK]
    assert isinstance(depth, DepthMap) and isinstance(track, MaskTrack)
    assert depth.kind is DepthKind.RELATIVE_DEPTH
    objects = track.masks.any(axis=1) & depth.valid
    assert float(np.median(depth.depth[objects])) == pytest.approx(1.0, abs=1e-6)
