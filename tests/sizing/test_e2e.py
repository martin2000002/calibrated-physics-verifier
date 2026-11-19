"""The end-to-end truth scale, the box match and the summary (no network)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.sizing import end_to_end
from physverify.sizing.objectron import FrameObject, FrameRecord, VideoRef

REF = VideoRef("cup/batch-1/7")


def record(frame_id: int, depth: float, x0: float = 0.2) -> FrameRecord:
    points = np.array([[x0, 0.3], [x0 + 0.2, 0.6]] * 4 + [[x0 + 0.1, 0.45]])
    obj = FrameObject(0, points, np.full(9, depth))
    return FrameRecord(frame_id, (obj,), (), 1440, 1920)


def test_the_truth_takes_every_step_th_native_frame() -> None:
    records = [record(i, 0.5 + i / 100) for i in range(10)]
    truth = end_to_end.clip_truth(REF, records, step=2, n_frames=4)
    assert truth.clip_id == "cup_batch-1_7" and truth.label == "cup"
    assert [f.index for f in truth.frames] == [0, 1, 2, 3]
    assert [f.depth_m for f in truth.frames] == pytest.approx([0.5, 0.52, 0.54, 0.56])
    assert truth.frames[0].box == pytest.approx((0.2, 0.3, 0.4, 0.6))
    assert end_to_end.ClipTruth.from_json(truth.to_json()) == truth


def test_box_iou_and_mask_box() -> None:
    assert end_to_end.box_iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)
    assert end_to_end.box_iou((0, 0, 0.1, 0.1), (0.5, 0.5, 1, 1)) == 0.0
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 4:8] = True
    assert end_to_end.mask_box(mask) == pytest.approx((0.2, 0.2, 0.4, 0.5))
    assert end_to_end.mask_box(np.zeros((4, 4), dtype=bool)) is None


def masks_at(boxes: list[tuple[int, int, int, int]], t: int) -> np.ndarray:
    out = np.zeros((t, len(boxes), 10, 10), dtype=bool)
    for n, (x0, y0, x1, y1) in enumerate(boxes):
        out[:, n, y0:y1, x0:x1] = True
    return out


def test_the_match_is_the_object_with_the_best_median_iou() -> None:
    truth = end_to_end.clip_truth(REF, [record(i, 1.0) for i in range(4)], step=1, n_frames=4)
    masks = masks_at([(6, 6, 9, 9), (2, 3, 4, 6)], 4)
    n, iou = end_to_end.match_object(truth, masks, np.arange(4)) or (None, 0.0)
    assert n == 1 and iou == pytest.approx(1.0)


def test_the_true_scale_is_the_median_depth_ratio() -> None:
    truth = end_to_end.clip_truth(REF, [record(i, 2.0) for i in range(5)], step=1, n_frames=5)
    z = np.array([1.0, 0.5, math.nan, 1.0, -1.0])
    assert end_to_end.true_scale(truth, z, np.arange(5)) == pytest.approx(2.0)
    assert math.isnan(end_to_end.true_scale(truth, np.full(5, math.nan), np.arange(5)))


def test_the_summary_measures_relative_error_and_coverage_per_method() -> None:
    rows = [
        end_to_end.ScaleRow("a", "m", 1.0, 1.1, 0.2),
        end_to_end.ScaleRow("b", "m", 1.0, 0.5, 0.1),
        end_to_end.ScaleRow("a", "p", 1.0, 1.0, 0.1),
    ]
    out = end_to_end.summarize(rows, n_boot=20, seed=0, ci=0.9)
    assert out["m"]["n"] == 2 and out["p"]["n"] == 1
    assert out["m"]["median_abs_rel"]["value"] == pytest.approx((0.1 + 0.5) / 2)
    assert out["m"]["cover_1s"]["value"] == pytest.approx(0.5)
    assert end_to_end.step_of(15.0) == 2 and end_to_end.step_of(60.0) == 1
