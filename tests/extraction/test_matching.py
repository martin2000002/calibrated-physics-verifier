"""AC3: identity matching, truth re-expressed in perceived ids, and the conditional scale fit."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import sampling_of
from synthetic_clip import SyntheticClip

from physverify.extraction.matching import (
    align_ground_truth,
    conditional_scale_fit,
    match_boxes,
    match_masks,
    remap_contacts,
    remap_states,
    take_frames,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.matching import IdMap, MatchMethod
from physverify.schema.perception import Detections, MaskTrack
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

SCALE = 7.3
RENAMED = {"ball": "obj_1", "box": "obj_0"}


def permuted_masks(truth: MaskTrack) -> MaskTrack:
    order = np.arange(truth.object_ids.size)[::-1]
    return replace(
        truth,
        object_ids=np.array([RENAMED[str(i)] for i in truth.object_ids[order]]),
        labels=truth.labels[order],
        masks=truth.masks[:, order],
        scores=truth.scores[:, order],
    )


def id_map(pairs: dict[str, str], unmatched_pred: tuple[str, ...] = ()) -> IdMap:
    return IdMap(
        pairs=tuple((pred, gt, 1.0) for pred, gt in pairs.items()),
        unmatched_pred=unmatched_pred,
        unmatched_gt=(),
        method=MatchMethod.MASK_IOU,
        frames_used=(0,),
        min_score=0.3,
    )


def camera_truth(clip: SyntheticClip) -> ObjectStateTable:
    return clip.ground_truth.states(CoordFrame.CAMERA_OPENCV)


def test_permuted_ids_are_recovered(fixture_clip: SyntheticClip) -> None:
    truth = fixture_clip.ground_truth.masks(sampling_of(fixture_clip.manifest))
    ids = match_masks(permuted_masks(truth), truth)
    assert ids.pred_to_gt() == {"obj_1": "ball", "obj_0": "box"}
    assert ids.unmatched_pred == () and ids.unmatched_gt == ()
    boxes = fixture_clip.ground_truth.detections(sampling_of(fixture_clip.manifest))
    order = np.arange(boxes.object_ids.size)[::-1]
    renamed = replace(
        boxes,
        boxes_xyxy=boxes.boxes_xyxy[order],
        scores=boxes.scores[order],
        labels=boxes.labels[order],
        object_ids=np.array([RENAMED[str(i)] for i in boxes.object_ids[order]]),
    )
    assert match_boxes(renamed, boxes).pred_to_gt() == {"obj_1": "ball", "obj_0": "box"}


def test_pairs_below_min_iou_stay_unmatched(fixture_clip: SyntheticClip) -> None:
    truth: Detections = fixture_clip.ground_truth.detections(sampling_of(fixture_clip.manifest))
    far = replace(truth, boxes_xyxy=truth.boxes_xyxy + 500.0)
    ids = match_boxes(far, truth)
    assert ids.pairs == ()
    assert set(ids.unmatched_pred) == set(ids.unmatched_gt) == {"ball", "box"}


def test_remap_states_leaves_unmatched_ids_nan_and_invisible(fixture_clip: SyntheticClip) -> None:
    truth = camera_truth(fixture_clip)
    ids = id_map({"p_box": "box"}, unmatched_pred=("ghost",))
    remapped = remap_states(truth, ids, ["ghost", "p_box"], ["thing", "box"])
    assert remapped.object_ids.tolist() == ["ghost", "p_box"]
    assert bool(np.isnan(remapped.position[:, 0]).all())
    assert not bool(remapped.visible[:, 0].any())
    assert remapped.primitive[0] == "unknown"
    np.testing.assert_array_equal(
        remapped.position[:, 1], truth.position[:, truth.object_index("box")]
    )


def test_remap_contacts_keeps_env_and_drops_unmatched(fixture_clip: SyntheticClip) -> None:
    table = fixture_clip.ground_truth.contacts(CoordFrame.CAMERA_OPENCV)
    record = table.records[0]
    assert (record.object_a, record.object_b) == ("box", ENV_ID)
    kept = remap_contacts(
        table, id_map({"p_box": "box"}), source_key="k", unit=LengthUnit.RELATIVE, length_scale=2.0
    )
    assert [(r.object_a, r.object_b) for r in kept.records] == [("p_box", ENV_ID)]
    np.testing.assert_allclose(kept.records[0].points, record.points / 2.0)
    assert kept.unit is LengthUnit.RELATIVE and kept.source_key == "k"
    dropped = remap_contacts(
        table, id_map({}, ("x",)), source_key="k", unit=LengthUnit.METER, length_scale=1.0
    )
    assert dropped.records == ()


def test_remap_contacts_reorders_pairs_and_flips_the_normal(fixture_clip: SyntheticClip) -> None:
    table = fixture_clip.ground_truth.contacts(CoordFrame.CAMERA_OPENCV)
    pair = replace(table.records[0], object_a="ball", object_b="box")
    swapped = remap_contacts(
        replace(table, records=(pair,)),
        id_map({"z_ball": "ball", "a_box": "box"}),
        source_key="k",
        unit=LengthUnit.METER,
        length_scale=1.0,
    ).records[0]
    assert (swapped.object_a, swapped.object_b) == ("a_box", "z_ball")
    np.testing.assert_array_equal(swapped.normal, -pair.normal)


def test_remap_contacts_drops_and_clips_records_outside_the_window(
    fixture_clip: SyntheticClip,
) -> None:
    table = fixture_clip.ground_truth.contacts(CoordFrame.CAMERA_OPENCV)
    base = table.records[0]
    inside = replace(base, t_start=2.0, t_end=9.0)
    outside = replace(base, t_start=12.0, t_end=15.0)
    clipped = remap_contacts(
        replace(table, records=(inside, outside)),
        id_map({"p_box": "box"}),
        source_key="k",
        unit=LengthUnit.METER,
        length_scale=1.0,
        window=(0.0, 6.0),
    )
    assert [(r.t_start, r.t_end) for r in clipped.records] == [(2.0, 6.0)]


def test_conditional_scale_fit_recovers_the_scale(fixture_clip: SyntheticClip) -> None:
    truth = fixture_clip.ground_truth.masks(sampling_of(fixture_clip.manifest))
    aligned = align_ground_truth(fixture_clip.ground_truth, permuted_masks(truth), truth.sampling)
    lift = replace(
        aligned.states,
        unit=LengthUnit.RELATIVE,
        position=aligned.states.position / SCALE,
    )
    fit = conditional_scale_fit(lift, aligned.states)
    assert fit is not None
    assert fit.meters_per_unit == pytest.approx(SCALE, abs=1e-9)
    assert fit.sigma < 1e-9
    assert fit.n_objects == 2


def test_a_single_ratio_leaves_the_scale_sigma_unknown(fixture_clip: SyntheticClip) -> None:
    truth = fixture_clip.ground_truth.masks(sampling_of(fixture_clip.manifest))
    aligned = align_ground_truth(fixture_clip.ground_truth, permuted_masks(truth), truth.sampling)
    visible = np.zeros_like(aligned.states.visible)
    visible[0, 0] = True
    lift = replace(
        aligned.states,
        unit=LengthUnit.RELATIVE,
        position=aligned.states.position / SCALE,
        visible=visible,
    )
    fit = conditional_scale_fit(lift, aligned.states)
    assert fit is not None
    assert (fit.n_objects, fit.n_samples) == (1, 1)
    assert fit.meters_per_unit == pytest.approx(SCALE, abs=1e-9)
    assert math.isnan(fit.sigma)


def test_take_frames_cuts_every_per_frame_field_and_nothing_else(
    fixture_clip: SyntheticClip,
) -> None:
    states = fixture_clip.ground_truth.states(CoordFrame.CAMERA_OPENCV)
    rows = np.array([0, 2], dtype=np.int64)
    cut = take_frames(states, states.frame_index[rows])
    n = states.n_objects
    assert (cut.t.shape, cut.frame_index.shape) == ((2,), (2,))
    assert (cut.position.shape, cut.quaternion.shape, cut.visible.shape) == (
        (2, n, 3),
        (2, n, 4),
        (2, n),
    )
    np.testing.assert_array_equal(cut.position, states.position[rows])
    np.testing.assert_array_equal(cut.object_ids, states.object_ids)
    np.testing.assert_array_equal(cut.gravity_dir, states.gravity_dir)


def test_identities_follow_the_whole_track_not_its_first_frames(
    fixture_clip: SyntheticClip,
) -> None:
    truth = fixture_clip.ground_truth.masks(sampling_of(fixture_clip.manifest))
    frames = truth.masks.shape[0]
    assert frames >= 12, "the swap must cover a minority of the frames"
    masks = truth.masks.copy()
    masks[:5] = truth.masks[:5, ::-1]
    swapped_early = replace(truth, object_ids=np.array(["p0", "p1"]), masks=masks)
    whole = match_masks(swapped_early, truth)
    first_five = match_masks(swapped_early, truth, frames=5)
    gt_ids = truth.object_ids.tolist()
    assert whole.pred_to_gt() == {"p0": gt_ids[0], "p1": gt_ids[1]}
    assert first_five.pred_to_gt() == {"p0": gt_ids[1], "p1": gt_ids[0]}
    assert len(whole.frames_used) == frames


def test_an_object_hidden_in_the_first_frames_is_still_matched(
    fixture_clip: SyntheticClip,
) -> None:
    truth = fixture_clip.ground_truth.masks(sampling_of(fixture_clip.manifest))
    late = truth.masks.copy()
    late[:5, 1] = False
    entering = replace(truth, masks=late)
    gt_ids = truth.object_ids.tolist()
    assert match_masks(truth, entering).pred_to_gt() == {i: i for i in gt_ids}
    assert match_masks(truth, entering, frames=5).unmatched_gt == (gt_ids[1],)
