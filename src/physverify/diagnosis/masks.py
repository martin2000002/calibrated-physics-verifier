"""Mask overlap shared by the TRACK metric and the real-video motion scores."""

from __future__ import annotations

import numpy as np

from physverify.diagnosis.cases import BoolArray, FloatArray


def mask_iou(a: BoolArray, b: BoolArray) -> FloatArray:
    """IoU over the last two axes; NaN where both masks are empty."""
    inter = np.sum(a & b, axis=(-2, -1), dtype=np.float64)
    union = np.sum(a | b, axis=(-2, -1), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou: FloatArray = inter / union
    return iou
