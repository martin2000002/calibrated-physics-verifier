"""One annotated object as a training or test sample, and its filters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import characteristic_lengths
from physverify.sizing.objectron import ObjectBox, VideoRef

PLACEHOLDER_EXTENTS = (1.0, 1.0, 1.0)
"""Annotators' unedited default box: not a measurement."""


@dataclass(frozen=True)
class SizeSample:
    label: str
    video_id: str
    extents: tuple[float, float, float]
    split: str


def samples_of(
    ref: VideoRef, boxes: Iterable[ObjectBox], split: str, min_extent_m: float
) -> list[SizeSample]:
    """The usable boxes of one video, labelled by the video's class directory."""
    return [
        SizeSample(ref.label, ref.video_id, box.extents, split)
        for box in boxes
        if box.extents != PLACEHOLDER_EXTENTS and min(box.extents) >= min_extent_m
    ]


def log_lengths(samples: Sequence[SizeSample]) -> NDArray[np.float64]:
    """log L with L = (e_x e_y e_z)^(1/3), in meters."""
    extents = np.array([s.extents for s in samples], dtype=np.float64).reshape(-1, 3)
    return np.log(characteristic_lengths(extents))
