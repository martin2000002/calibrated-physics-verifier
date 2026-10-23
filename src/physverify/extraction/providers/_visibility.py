"""In-frame state and occlusion of a tracked object inside an interior gap of its masks.

A gap is a maximal run of rows without a usable mask that has a visible row before and after it.
The object stays in frame through the gap unless a mask enclosing it touches the image border.
Inside the gap its box is interpolated linearly between the two enclosing masks, and the
occlusion estimate is the share of that box covered by the other tracked masks of the row.
Declared approximations: a box instead of the silhouette, linear motion inside the gap, no depth
ordering, and an untracked occluder is not seen.

The estimate is decisive only at its ends: a covered share >= the occlusion threshold explains
the gap, a share of exactly 0 does not, and anything in between is unknown (NaN), so the check
abstains instead of counting a disappearance perception cannot tell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Box:
    """Pixel box [x0, x1) x [y0, y1) in column/row coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, eq=False)
class GapVisibility:
    in_frame: BoolArray  # (T, N)
    occlusion: FloatArray  # (T, N)


def interior_gaps(visible: BoolArray) -> list[tuple[int, int]]:
    """Runs [start, stop) of non-visible rows with a visible row on both sides."""
    shown = np.asarray(visible, dtype=bool)
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for row, seen in enumerate(shown.tolist()):
        if not seen and start is None:
            start = row
        elif seen and start is not None:
            if start > 0:
                gaps.append((start, row))
            start = None
    return gaps


def touches_border(mask: BoolArray) -> bool:
    """Some pixel of the mask lies in the first or last image row or column."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def mask_box(mask: BoolArray) -> Box | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return Box(float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))


def interpolated_boxes(before: Box, after: Box, n: int) -> list[Box]:
    """Boxes of the n gap rows: centre and size linear in time between the enclosing rows."""
    boxes: list[Box] = []
    for i in range(n):
        a = (i + 1) / (n + 1)
        cx = _lerp((before.x0 + before.x1) / 2, (after.x0 + after.x1) / 2, a)
        cy = _lerp((before.y0 + before.y1) / 2, (after.y0 + after.y1) / 2, a)
        w = _lerp(before.x1 - before.x0, after.x1 - after.x0, a)
        h = _lerp(before.y1 - before.y0, after.y1 - after.y0, a)
        boxes.append(Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def _lerp(start: float, stop: float, a: float) -> float:
    return start + a * (stop - start)


def covered_fraction(box: Box, others: BoolArray) -> float:
    """Share of the box pixels covered by ``others`` (H, W); NaN for a box without pixels."""
    height, width = others.shape
    c0, c1 = max(0, round(box.x0)), min(width, round(box.x1))
    r0, r1 = max(0, round(box.y0)), min(height, round(box.y1))
    area: int = max(0, c1 - c0) * max(0, r1 - r0)
    if area == 0:
        return math.nan
    return float(others[r0:r1, c0:c1].sum()) / area


def decisive_occlusion(fraction: float, occluded_at: float) -> float:
    """The estimate kept only at its decisive ends: >= ``occluded_at`` or exactly 0."""
    if fraction >= occluded_at or fraction == 0.0:
        return fraction
    return math.nan


def gap_visibility(masks: BoolArray, visible: BoolArray, occluded_at: float) -> GapVisibility:
    """``in_frame`` and ``occlusion`` for masks (T, N, H, W) and the visible table column (T, N).

    Visible cells keep in_frame True and occlusion NaN; non-visible cells outside an interior gap
    (clip start or end, never seen) are out of frame, since nothing shows the object stayed."""
    visible = np.asarray(visible, dtype=bool)
    in_frame = visible.copy()
    occlusion = np.full(visible.shape, math.nan)
    for n in range(visible.shape[1]):
        for lo, hi in interior_gaps(visible[:, n]):
            _fill_gap(masks, n, (lo, hi), occluded_at, in_frame, occlusion)
    return GapVisibility(in_frame=in_frame, occlusion=occlusion)


def _fill_gap(
    masks: BoolArray,
    n: int,
    gap: tuple[int, int],
    occluded_at: float,
    in_frame: BoolArray,
    occlusion: FloatArray,
) -> None:
    lo, hi = gap
    before, after = masks[lo - 1, n], masks[hi, n]
    box_before, box_after = mask_box(before), mask_box(after)
    if box_before is None or box_after is None or touches_border(before) or touches_border(after):
        return
    boxes = interpolated_boxes(box_before, box_after, hi - lo)
    for row, box in zip(range(lo, hi), boxes, strict=True):
        covered = np.delete(masks[row], n, axis=0).any(axis=0)
        in_frame[row, n] = True
        occlusion[row, n] = decisive_occlusion(covered_fraction(box, covered), occluded_at)
