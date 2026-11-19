"""Rest of object pairs stacked in the image for ``contacts/kinematic_scene``: a body whose mask
lies on another's within the pixel noise rests on it when the bodies are within the gap noise in
3D and the upper one is higher beyond the height noise."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scene_contact_cases as cases

from physverify.extraction.providers._image_motion import (
    ImageProjection,
    PixelError,
    object_images,
)
from physverify.extraction.providers._pair_rest import ImagePairs, Key, PairRow, image_gap
from physverify.extraction.providers._scene_records import SceneInputs, SceneOutcome
from physverify.schema.contacts import ContactKind
from physverify.schema.states import PrimitiveKind

cs = cases.cs
sc = cs.sc
GAP = 0.1 * cases.SIDE


@dataclass(frozen=True)
class SpyPairs(ImagePairs):
    """Keeps the image-stacked rows it found."""

    seen: list[dict[Key, dict[int, PairRow]]] = field(default_factory=list)

    def stack_rows(self, inputs: SceneInputs, up: np.ndarray) -> dict[Key, dict[int, PairRow]]:
        found = super().stack_rows(inputs, up)
        self.seen.append(found)
        return found


def image_pairs(case: cs.sc.Case, error: PixelError | None = None) -> SpyPairs:
    track = cs.disk_track(case.states, cases.sampling(case))
    return SpyPairs(
        object_images(track, case.states),
        ImageProjection.of(case.states, cs.scene_camera(case)),
        PixelError.of(cases.CENTROID_PX) if error is None else error,
    )


def two_boxes(bottom: float, behind: float = 0.0) -> cs.sc.Case:
    """box0 on the floor and box1 ``behind`` it (away from the camera) with its bottom at
    ``bottom``."""
    side = cases.SIDE
    box1 = sc.still("box1", PrimitiveKind.BOX, sc.cube(side), (0.0, behind, bottom + side / 2.0))
    return sc.build("pair", [cases.box("box0", 0.0, 0.0), box1], [], 25.0)


def resting_pairs(found: SceneOutcome) -> list[tuple[str, str]]:
    records = cases.pair_records(found.records)
    assert all(r.kind is ContactKind.RESTING for r in records)
    return [(r.object_a, r.object_b) for r in records]


def test_a_box_just_above_another_rests_on_it_only_in_the_image() -> None:
    case = two_boxes(cases.SIDE + GAP)
    assert GAP > 0.05 * cases.LENGTH
    assert resting_pairs(cases.outcome(case)) == []
    found = cases.outcome(case, pair_rest=image_pairs(case))
    assert resting_pairs(found) == [("box0", "box1")]
    assert not cases.env_records(found.records, "box1")


def test_a_box_just_behind_another_lies_on_it_in_the_image_but_is_not_stacked() -> None:
    case = two_boxes(0.0, behind=cases.SIDE + GAP)
    spy = image_pairs(case)
    found = cases.outcome(case, pair_rest=spy)
    assert any(spy.seen[0].values())
    assert resting_pairs(found) == []


def test_an_unknown_centroid_error_keeps_only_the_proximity_rows() -> None:
    case = two_boxes(cases.SIDE + GAP)
    assert (
        resting_pairs(cases.outcome(case, pair_rest=image_pairs(case, PixelError.of(None)))) == []
    )
    touching = two_boxes(cases.SIDE)
    unknown = image_pairs(touching, PixelError.of(None))
    assert resting_pairs(cases.outcome(touching, pair_rest=unknown)) == [("box0", "box1")]


def test_the_image_gap_runs_from_the_upper_bottom_to_the_lower_top() -> None:
    upper, lower = np.zeros((10, 6), dtype=np.bool_), np.zeros((10, 6), dtype=np.bool_)
    upper[1:4, 1:4] = True
    lower[6:9, 2:6] = True
    assert image_gap(upper, lower) == 3.0
    assert image_gap(upper, np.zeros_like(lower)) is None
