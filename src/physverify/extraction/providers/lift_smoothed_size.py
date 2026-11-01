"""Back-projection whose sizes follow a smooth along-ray depth of each object (ADR 0029)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.extraction.backprojection import ObjectSummary
from physverify.extraction.lift_geometry import GroundFrame
from physverify.extraction.lift_params import SmoothedSizeLiftParams
from physverify.extraction.mask_border import border_cut
from physverify.extraction.providers._backproject import BackprojectLift
from physverify.extraction.providers._common import expect_params
from physverify.physics.running_median import EdgeMode, running_median
from physverify.schema.perception import MaskTrack
from physverify.schema.states import PrimitiveKind

FloatArray = NDArray[np.float64]


class SmoothedSizeLift(BackprojectLift):
    """``lift/backproject`` with each row's lengths re-referred to a running median of the
    object's own surface depth: L'_t = L_t z_s / z_t, z_s = runmed_W(z) over the object's rows.

    Every LIFT size is linear in its row's surface depth z (image widths and the square root of
    the image area, times z), so the per-row depth noise of the depth model enters it as one
    factor, whatever the primitive. A rigid body moves its depth continuously, and a change of
    size changes the silhouette's area, which this leaves alone. So it applies to every
    primitive; where the depth is steady the ratio is close to 1.

    Only the lengths of the summaries change (extents, silhouette widths and area); the centre
    and its pixel stay, so the position moves only through the centre's offset behind the
    surface, a fraction of the size. Rows cut by the image border do not enter the median and
    keep their size.

    ``lift/ground_anchored`` extends ``run`` and calls ``super().run``; to stack it on this one,
    its class statement names ``SmoothedSizeLift`` instead of ``BackprojectLift`` and its params
    derive from ``SmoothedSizeLiftParams``, one line each."""

    name: ClassVar[str] = "smoothed_size"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = SmoothedSizeLiftParams

    def __init__(self, params: BaseModel) -> None:
        smoothing = expect_params(params, SmoothedSizeLiftParams)
        super().__init__(smoothing)
        self.smoothing = smoothing

    def _summaries(
        self,
        track: MaskTrack,
        z: FloatArray,
        valid: NDArray[np.bool_],
        intrinsics: FloatArray,
        frame: GroundFrame | None,
        primitives: list[PrimitiveKind],
    ) -> list[list[ObjectSummary | None]]:
        rows = super()._summaries(track, z, valid, intrinsics, frame, primitives)
        cut = border_cut(track.masks, self.smoothing.border_margin_px)
        window, edges = self.smoothing.depth_window, self.smoothing.depth_edges
        columns = [
            smoothed_sizes([row[n] for row in rows], cut[:, n], window, edges)
            for n in range(len(primitives))
        ]
        return [[column[t] for column in columns] for t in range(len(rows))]


def smoothed_sizes(
    column: Sequence[ObjectSummary | None],
    excluded: NDArray[np.bool_],
    window: int,
    edges: EdgeMode,
) -> list[ObjectSummary | None]:
    """One object's summaries over time (``None`` = not visible), each rescaled by
    runmed(z) / z; rows that are missing, ``excluded`` or have no usable ratio stay as they are."""
    depth = np.array(
        [
            math.nan if summary is None or skip else float(summary.centre[2])
            for summary, skip in zip(column, excluded, strict=True)
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = running_median(depth, window, edges) / depth
    return [
        summary if summary is None or not (math.isfinite(r) and r > 0.0) else rescaled(summary, r)
        for summary, r in zip(column, ratio.tolist(), strict=True)
    ]


def rescaled(summary: ObjectSummary, ratio: float) -> ObjectSummary:
    """``summary`` with the lengths its silhouette gives at ``ratio`` times its depth (widths
    times ratio, area times ratio squared)."""
    view = summary.view
    if view is not None:
        view = replace(
            view, across=view.across * ratio, along=view.along * ratio, area=view.area * ratio**2
        )
    return replace(summary, extents=summary.extents * ratio, view=view)


PLUGINS = (SmoothedSizeLift,)
