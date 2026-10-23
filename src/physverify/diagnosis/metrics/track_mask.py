"""TRACK: region (J) and boundary (F) accuracy as in DAVIS, and the centroid error in radii."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from physverify.diagnosis.masks import mask_iou
from physverify.diagnosis.measurement_cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
)
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import DEFAULT_MATCH_FRAMES, DEFAULT_MIN_IOU, match_masks
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage


def region_similarity(pred: BoolArray, truth: BoolArray) -> float:
    """DAVIS J: the mask IoU, 1 when both masks are empty."""
    iou = float(mask_iou(pred, truth))
    return 1.0 if math.isnan(iou) else iou


def boundary(mask: BoolArray) -> BoolArray:
    boundary_pixels: BoolArray = mask & ~binary_erosion(mask)
    return boundary_pixels


def boundary_tolerance_px(width: int, height: int, fraction: float) -> float:
    return float(max(1, round(fraction * math.hypot(width, height))))


def within(points: BoolArray, reference: BoolArray, tolerance: float) -> int:
    """Pixels of ``points`` within ``tolerance`` of some pixel of the non-empty ``reference``."""
    distance = distance_transform_edt(~reference)
    return int(np.count_nonzero(points & (distance <= tolerance)))


def boundary_f(pred: BoolArray, truth: BoolArray, tolerance: float) -> float:
    """DAVIS F: both boundaries empty -> 1; exactly one empty -> 0."""
    b_pred, b_truth = boundary(pred), boundary(truth)
    n_pred, n_truth = int(b_pred.sum()), int(b_truth.sum())
    if n_pred == 0 and n_truth == 0:
        return 1.0
    if n_pred == 0 or n_truth == 0:
        return 0.0
    precision = within(b_pred, b_truth, tolerance) / n_pred
    recall = within(b_truth, b_pred, tolerance) / n_truth
    total = precision + recall
    return 0.0 if total == 0 else 2.0 * precision * recall / total


def centroid_px(mask: BoolArray) -> FloatArray:
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean(), rows.mean()], dtype=np.float64)


def centroid_rel(pred: BoolArray, truth: BoolArray) -> FloatArray:
    """(c_pred - c_gt)/r_eq,gt per image axis (x, y), with r_eq,gt = sqrt(area_gt/pi)."""
    r_eq = math.sqrt(float(truth.sum()) / math.pi)
    return (centroid_px(pred) - centroid_px(truth)) / r_eq


@dataclass
class _Collected:
    j: list[float]
    f: list[float]
    centroid: list[FloatArray]


def _collect_object(pred: BoolArray, truth: BoolArray, tolerance: float, into: _Collected) -> None:
    """``pred`` and ``truth`` are (T, h, w); only frames where the truth is visible count."""
    for p_t, g_t in zip(pred, truth, strict=True):
        if not g_t.any():
            continue
        into.j.append(region_similarity(p_t, g_t))
        into.f.append(boundary_f(p_t, g_t, tolerance))
        if p_t.any():
            into.centroid.append(centroid_rel(p_t, g_t))


@dataclass(frozen=True)
class MaskMetric:
    """Unmatched ground-truth objects score J = F = 0 on their visible frames. Needs no context."""

    stage: ClassVar[Stage] = Stage.TRACK
    name: ClassVar[str] = "mask"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, MaskTrack) or not isinstance(truth, MaskTrack):
            raise TypeError("mask metric compares two MaskTrack")
        if (
            pred.masks.shape[0] != truth.masks.shape[0]
            or pred.masks.shape[2:] != truth.masks.shape[2:]
        ):
            raise ValueError("pred and truth masks must share frames and size")
        height, width = truth.masks.shape[2:]
        tolerance = boundary_tolerance_px(width, height, case.options.boundary_tol_frac)
        to_pred = match_masks(pred, truth, DEFAULT_MATCH_FRAMES, DEFAULT_MIN_IOU).gt_to_pred()
        pred_ids = pred.object_ids.tolist()
        collected = _Collected([], [], [])
        present = matched = 0
        for j, gt_id in enumerate(truth.object_ids.tolist()):
            g = truth.masks[:, j]
            if not g.any():
                continue
            present += 1
            p = np.zeros_like(g)
            if gt_id in to_pred:
                matched += 1
                p = pred.masks[:, pred_ids.index(to_pred[gt_id])]
            _collect_object(p, g, tolerance, collected)
        j_arr, f_arr = (
            np.array(collected.j, dtype=np.float64),
            np.array(collected.f, dtype=np.float64),
        )
        centroid = np.concatenate(collected.centroid) if collected.centroid else np.zeros(0)
        return MetricOutcome(
            samples=MappingProxyType(
                {
                    "mask_iou": j_arr,
                    "boundary_f": f_arr,
                    "j_and_f": (j_arr + f_arr) / 2.0,
                    "centroid_px_rel": centroid,
                }
            ),
            ratios=MappingProxyType({"object_recall": (float(matched), float(present))}),
        )


PLUGINS = (MaskMetric,)
