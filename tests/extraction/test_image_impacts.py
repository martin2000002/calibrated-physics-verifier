"""Impacts in the image plane for ``contacts/kinematic_scene``: kinks of the disc masks' centroid
tracks, attributed to the environment or to another object in the image and placed in 3D."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scene_contact_cases as cases

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.providers._image_impacts import ImageImpacts
from physverify.extraction.providers._image_motion import (
    ImageProjection,
    PixelError,
    object_images,
)
from physverify.extraction.providers._scene_plane import SupportPlane
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord
from physverify.schema.error_profile import CENTROID_PX_REL, ErrorEntry
from physverify.schema.stages import Stage

cs = cases.cs
sc = cs.sc
TOL = 1e-9
EXACT_ENTRY = ErrorEntry.exact(Stage.TRACK, GROUND_TRUTH, CENTROID_PX_REL)


def image_impacts(
    case: cs.sc.Case, error: PixelError | None = None, last_row: int | None = None
) -> ImageImpacts:
    """Image impacts of the case's disc masks, on an image tall enough for the balls in the
    air, with no mask after ``last_row`` when given."""
    states = case.states
    track = cs.disk_track(states, cases.sampling(case), size=(400, 640))
    if last_row is not None:
        masks = track.masks.copy()
        masks[last_row + 1 :] = False
        track = replace(track, masks=masks)
    return ImageImpacts(
        object_images(track, states),
        ImageProjection.of(states, cs.scene_camera(case)),
        PixelError.of(cases.CENTROID_PX) if error is None else error,
    )


def impacts_of(
    case: cs.sc.Case, error: PixelError | None = None, last_row: int | None = None
) -> list[ContactRecord]:
    found = cases.outcome(case, impacts=image_impacts(case, error, last_row))
    return [r for r in found.records if r.kind is ContactKind.IMPACT]


def test_a_bounce_on_the_floor_is_one_environment_impact_at_its_time() -> None:
    case = cs.bounce_on_floor()
    impacts = impacts_of(case)
    assert [(r.object_a, r.object_b) for r in impacts] == [("ball", ENV_ID)]
    record = impacts[0]
    assert abs(record.t_start - sc.EVENT_FRAME) <= 1.0
    assert record.points.shape == (1, 3)
    assert record.point_distances == pytest.approx(np.zeros(1), abs=TOL)
    np.testing.assert_allclose(record.normal, cases.world_plane(case)[0], atol=TOL)
    assert 0.5 < record.confidence <= 1.0


def test_a_fall_without_its_bounce_has_no_impact() -> None:
    assert impacts_of(cs.bounce_on_floor(), last_row=int(sc.EVENT_FRAME) - 1) == []


def test_a_collision_in_the_air_is_one_pair_impact() -> None:
    """Under an exact track: the 2 m/s exchange is about 280 px/s, below the noise of the
    measured centroid error at this distance."""
    case = cs.collision_over_floor()
    impacts = impacts_of(case, PixelError.of(EXACT_ENTRY))
    pairs = cases.pair_records(impacts)
    assert len(pairs) == 1 and not cases.env_records(impacts)
    record = pairs[0]
    assert abs(record.t_start - sc.EVENT_FRAME) <= 1.0
    assert str(record.object_a) < str(record.object_b)


def test_an_unknown_centroid_error_finds_no_impact() -> None:
    assert impacts_of(cs.bounce_on_floor(), PixelError.of(None)) == []


def plane_values(planes: tuple[SupportPlane | None, ...]) -> list[tuple[float, ...] | None]:
    return [None if p is None else (*p.plane.normal, p.height, p.sigma, p.source) for p in planes]


def test_image_impacts_never_move_the_scene_planes() -> None:
    case = cs.bounce_on_floor()
    image = cases.outcome(case, impacts=image_impacts(case))
    velocity = cases.outcome(case)
    assert image.planes[0] is not None
    assert plane_values(image.planes) == plane_values(velocity.planes)
