"""R4_size v3_window_depth_unoccluded: v3_window_depth without the rows whose mask box a nearer
tracked object covers."""

from __future__ import annotations

from typing import ClassVar

from physverify.checks._base import CheckInputs
from physverify.checks._size import BoolArray
from physverify.checks.variants._window_depth_size import WindowDepthSizeCheck


class UnoccludedWindowDepthSizeCheck(WindowDepthSizeCheck):
    """R4_size v3_window_depth_unoccluded [rel_size_change]: v3_window_depth that also leaves
    out, for the object and as a reference, the rows where the mask of another tracked object
    nearer the camera covers part of the object's mask box (``nearer_cover`` > 0, from the lift).
    There the object's mask may be cut by that object, so its extent is not the object's.

    A NaN cover (no mask or no depth) keeps the row, as an unknown occlusion does in v3. A
    ground-truth LIFT (by provenance) and a table without ``nearer_cover`` are left as
    v3_window_depth sees them. The price is a declared blind spot: an untracked occluder still
    cuts the mask unseen, and a nearer object that overlaps the box but not the silhouette drops
    a sound row."""

    variant: ClassVar[str] = "v3_window_depth_unoccluded"
    # 2: sigma combines the extent sigmas as correlated axes by default.
    version: ClassVar[str] = "2"

    def _whole_rows(self, inputs: CheckInputs) -> BoolArray:
        whole = super()._whole_rows(inputs)
        cover = inputs.states.nearer_cover
        if cover is None or self._exact_lift(inputs):
            return whole
        return whole & ~(cover > 0)


PLUGINS = (UnoccludedWindowDepthSizeCheck,)
