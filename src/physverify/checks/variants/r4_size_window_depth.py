"""R4_size v3_window_depth: v3 on extents rescaled to the object's running-median depth."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from physverify.checks._base import CheckInputs
from physverify.checks.variants._detrended_size import DetrendedSizeCheck
from physverify.physics.measures.window_depth import window_depth_factor
from physverify.physics.row_time import odd_rows_in
from physverify.schema.checks import CheckResult
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame


class WindowDepthSizeCheck(DetrendedSizeCheck):
    """R4_size v3_window_depth [rel_size_change]: v3 with every extent and its sigma multiplied by
    runmed_W(z) / z (``window_depth_factor``), z the object's camera depth and W v3's own
    ``window_s``; sigma scales with the extent, since extents are homogeneous of degree 1 in z.

    A lift's extent is the silhouette in pixels times the per-row depth over the focal, so the
    per-row noise of a monocular depth reads as size changes; the silhouette still enters row by
    row, so a real change of size stays. A ground-truth LIFT (by provenance) takes no size from a
    depth estimate, and states outside the camera frame have no depth axis: both are left as v3
    sees them. The price is a declared blind spot: a change of the object's depth that the
    running median does not follow and the silhouette does not show is not corrected."""

    variant: ClassVar[str] = "v3_window_depth"
    version: ClassVar[str] = "1"

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        return super().evaluate(self._at_window_depth(inputs))

    def _at_window_depth(self, inputs: CheckInputs) -> CheckInputs:
        states = inputs.states
        exact = inputs.provenance.get(Stage.LIFT.value) == GROUND_TRUTH
        if exact or states.frame is not CoordFrame.CAMERA_OPENCV:
            return inputs
        window = odd_rows_in(self.params.window_s, inputs.row_seconds)
        factor = window_depth_factor(states.position[..., 2], states.visible, window)[..., None]
        scaled = replace(states, size=states.size * factor, size_sigma=states.size_sigma * factor)
        return replace(inputs, states=scaled)


PLUGINS = (WindowDepthSizeCheck,)
