"""State-table pairing and lengths shared by the LIFT, KINEMATICS and CONTACTS metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.diagnosis.cases import BoolArray, FloatArray
from physverify.diagnosis.normalizers import characteristic_lengths
from physverify.schema.states import ObjectStateTable

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class StatePairs:
    """Rows of the common frames and columns of the common object ids of two state tables."""

    pred_rows: IntArray
    truth_rows: IntArray
    pred_cols: IntArray
    truth_cols: IntArray

    def pred[A: np.generic](self, values: NDArray[A]) -> NDArray[A]:
        return values[self.pred_rows][:, self.pred_cols]

    def truth[A: np.generic](self, values: NDArray[A]) -> NDArray[A]:
        return values[self.truth_rows][:, self.truth_cols]

    def visible(self, pred: ObjectStateTable, truth: ObjectStateTable) -> BoolArray:
        both: BoolArray = self.pred(pred.visible) & self.truth(truth.visible)
        return both


def paired_states(pred: ObjectStateTable, truth: ObjectStateTable) -> StatePairs:
    _, pred_rows, truth_rows = np.intersect1d(
        pred.frame_index, truth.frame_index, return_indices=True
    )
    pred_ids = set(pred.object_ids.tolist())
    common = [
        (pred.object_index(object_id), j)
        for j, object_id in enumerate(truth.object_ids.tolist())
        if object_id in pred_ids
    ]
    pred_cols = np.array([i for i, _ in common], dtype=np.int64)
    truth_cols = np.array([j for _, j in common], dtype=np.int64)
    return StatePairs(
        pred_rows.astype(np.int64), truth_rows.astype(np.int64), pred_cols, truth_cols
    )


def finite_rows(values: FloatArray) -> BoolArray:
    rows: BoolArray = np.isfinite(values).all(axis=-1)
    return rows


def object_length(sizes: FloatArray) -> float:
    """Median over frames of the characteristic length of one object's (T, 3) sizes."""
    lengths = characteristic_lengths(sizes)
    usable = lengths[np.isfinite(lengths) & (lengths > 0)]
    return float(np.median(usable)) if usable.size else math.nan


def sigma_scores(residual: FloatArray, sigma: FloatArray, ok: BoolArray) -> FloatArray:
    """residual/sigma per axis, only where the declared sigma is finite and > 0."""
    usable = ok[..., None] & np.isfinite(sigma) & (sigma > 0)
    scores: FloatArray = residual[usable] / sigma[usable]
    return scores
