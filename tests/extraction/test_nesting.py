"""lift/backproject_unnested: nested tracks left out, real-video labels mapped to primitives."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from support import make_context, run_provider
from synthetic_clip import SyntheticClip
from test_lift_backproject import truth_inputs

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.lift_params import (
    KUBRIC_PRIMITIVES,
    REAL_PRIMITIVES,
    LiftParams,
    UnnestedLiftParams,
)
from physverify.extraction.providers._nesting import containment, outer_tracks, without_nested
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.extraction.providers.lift_backproject_unnested import UnnestedBackprojectLift
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.sizing.priors import label_candidates, lookup_label

FRAMES, SIDE = 4, 20


def squares(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    """(T, N, SIDE, SIDE) masks, one still square (row0, row1, col0, col1) per track."""
    masks = np.zeros((FRAMES, len(boxes), SIDE, SIDE), dtype=np.bool_)
    for n, (r0, r1, c0, c1) in enumerate(boxes):
        masks[:, n, r0:r1, c0:c1] = True
    return masks


def test_a_mask_inside_another_is_dropped() -> None:
    masks = squares((0, 10, 0, 10), (2, 6, 2, 6), (12, 18, 12, 18))
    inside = containment(masks)
    assert inside[1, 0] == pytest.approx(1.0) and inside[0, 1] == pytest.approx(0.16)
    assert outer_tracks(masks, 0.5) == [0, 2]


def test_of_two_duplicates_the_larger_stays() -> None:
    masks = squares((4, 10, 4, 10), (3, 10, 3, 10))
    assert outer_tracks(masks, 0.5) == [1]


@pytest.mark.parametrize(("max_overlap", "kept"), [(0.5, [0, 1]), (0.3, [0])])
def test_the_overlap_is_configurable(max_overlap: float, kept: list[int]) -> None:
    masks = squares((0, 10, 0, 10), (0, 10, 6, 16))
    assert containment(masks)[1, 0] == pytest.approx(0.4)
    assert outer_tracks(masks, max_overlap) == kept


def test_frames_where_a_track_is_absent_do_not_count() -> None:
    masks = squares((0, 10, 0, 10), (2, 6, 2, 6))
    masks[:2, 0] = False
    masks[:2, 1, 12:18, 12:18] = True
    assert containment(masks)[1, 0] == pytest.approx(1.0)


def with_part(track: MaskTrack) -> MaskTrack:
    """``track`` plus a second track on every other column of object 0's mask."""
    part = track.masks[:, :1].copy()
    part[..., ::2] = False
    return replace(
        track,
        object_ids=np.append(track.object_ids, "part_0"),
        labels=np.append(track.labels, "ball"),
        masks=np.concatenate([track.masks, part], axis=1),
        scores=np.concatenate([track.scores, track.scores[:, :1]], axis=1),
    )


def lifted(clip: SyntheticClip, provider: type[BackprojectLift], part: bool) -> ObjectStateTable:
    upstream = truth_inputs(clip)
    track = upstream[Stage.TRACK]
    assert isinstance(track, MaskTrack)
    if part:
        upstream[Stage.TRACK] = with_part(track)
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: provider.name}
    output = run_provider(provider, make_context(clip, upstream=upstream, plan=names))
    assert isinstance(output, ObjectStateTable)
    return output


def test_the_unnested_lift_leaves_the_part_out(fixture_clip: SyntheticClip) -> None:
    plain = lifted(fixture_clip, BackprojectLift, part=True)
    unnested = lifted(fixture_clip, UnnestedBackprojectLift, part=True)
    assert plain.object_ids.tolist()[-1] == "part_0"
    assert unnested.object_ids.tolist() == plain.object_ids.tolist()[:-1]
    reference = lifted(fixture_clip, BackprojectLift, part=False)
    np.testing.assert_array_equal(unnested.position, reference.position)
    np.testing.assert_array_equal(unnested.primitive, reference.primitive)


def test_a_track_without_nesting_is_returned_as_is(fixture_clip: SyntheticClip) -> None:
    track = truth_inputs(fixture_clip)[Stage.TRACK]
    assert isinstance(track, MaskTrack)
    assert without_nested(track, 0.5) is track


def test_the_kubric_map_is_the_default_of_backproject() -> None:
    assert LiftParams().primitive_by_label == dict(KUBRIC_PRIMITIVES)
    assert UnnestedLiftParams().primitive_by_label == dict(REAL_PRIMITIVES)
    assert all(REAL_PRIMITIVES[label] is kind for label, kind in KUBRIC_PRIMITIVES.items())


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("block", PrimitiveKind.BOX),
        ("domino", PrimitiveKind.BOX),
        ("marble", PrimitiveKind.SPHERE),
        ("white bottle", PrimitiveKind.CYLINDER),
        ("glass globe", PrimitiveKind.SPHERE),
        ("glass", None),
        ("orange plastic egg", None),
        ("rubber duck", None),
        ("bag of oranges", None),
    ],
)
def test_real_query_phrases_find_their_primitive(label: str, kind: PrimitiveKind | None) -> None:
    assert lookup_label(label, REAL_PRIMITIVES) is kind
    assert lookup_label(label, KUBRIC_PRIMITIVES) is None


@pytest.mark.parametrize(
    ("label", "kind"),
    [("large orange ball", PrimitiveKind.SPHERE), ("ball box", PrimitiveKind.BOX)],
)
def test_the_last_known_word_is_the_head_noun(label: str, kind: PrimitiveKind) -> None:
    assert lookup_label(label, KUBRIC_PRIMITIVES) is kind


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("ball on book", PrimitiveKind.SPHERE),
        ("cup on book", PrimitiveKind.CYLINDER),
        ("red cylinder on glass", PrimitiveKind.CYLINDER),
        ("book under a ball", PrimitiveKind.BOX),
        ("tower of jenga blocks", PrimitiveKind.BOX),
    ],
)
def test_the_head_noun_comes_before_the_first_preposition(
    label: str, kind: PrimitiveKind
) -> None:
    assert lookup_label(label, REAL_PRIMITIVES) is kind


def test_the_candidates_stop_at_the_first_preposition() -> None:
    assert label_candidates("Ball on Book") == ("ball on book", "ball")
    assert label_candidates("on book") == ("on book",)


def test_without_its_own_profile_the_unnested_lift_leaves_sigma_unknown(
    fixture_clip: SyntheticClip,
) -> None:
    names = {
        **dict.fromkeys(STAGE_ORDER, GROUND_TRUTH),
        Stage.TRACK: "edgetam",
        Stage.DEPTH: "vda_small",
        Stage.LIFT: UnnestedBackprojectLift.name,
    }
    ctx = make_context(fixture_clip, upstream=truth_inputs(fixture_clip), plan=names)
    table = run_provider(UnnestedBackprojectLift, ctx)
    assert isinstance(table, ObjectStateTable)
    visible = table.visible
    assert bool(visible.any())
    assert bool(np.isnan(table.position_sigma[visible]).all())
    assert bool(np.isnan(table.size_sigma[visible]).all())
