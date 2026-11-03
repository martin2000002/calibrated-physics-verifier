"""Rest of an object in the image plane; not a plugin.

The pixel track of each object's mask (``_image_motion.py``) is tested as in
``_velocity_tests.py`` with two components, with the plan's Savitzky-Golay window and order (the
defaults when KINEMATICS is not ``savgol``).
Under a fixed camera flight and rest differ by orders of magnitude here, while the 3D test,
whose sigma_v scales with the object's own speed, passes the apex of a throw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._image_motion import (
    PixelError,
    mask_motion,
    object_images,
    pixel_error,
)
from physverify.extraction.providers._savgol import SavgolParams
from physverify.extraction.providers._velocity_tests import rest_runs
from physverify.physics.kinematics import SavgolSpec
from physverify.physics.row_time import seconds_per_row
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ImageRest:
    """Rest runs of each object's mask; an object without a mask has no rest."""

    track: MaskTrack
    error: PixelError
    spec: SavgolSpec

    @property
    def known(self) -> bool:
        return self.error.known

    def mask(self, states: ObjectStateTable, alpha: float, min_rows: int) -> BoolArray:
        rest = np.zeros(states.visible.shape, dtype=np.bool_)
        if not self.known:
            return rest
        for n, image in enumerate(object_images(self.track, states)):
            if image is None:
                continue
            motion = mask_motion(image, states.t, self.spec, self.error)
            sigma = motion.velocity_sigma[:, 0]
            for run in rest_runs(motion.velocity, image.valid, sigma, alpha, min_rows):
                rest[run, n] = True
        return rest


def savgol_params(ctx: StageContext) -> SavgolParams:
    """The plan's KINEMATICS window and order; the defaults when that stage is not ``savgol``."""
    params = ctx.stage_params.get(Stage.KINEMATICS)
    return params if isinstance(params, SavgolParams) else SavgolParams()


def image_rest(ctx: StageContext, states: ObjectStateTable) -> ImageRest:
    """The image rest of the clip's masks, with the kinematics window at its row rate."""
    row_seconds = seconds_per_row(states.frame_index, ctx.sampling.seconds_per_frame)
    return ImageRest(
        track=ctx.require(Stage.TRACK, MaskTrack),
        error=pixel_error(ctx),
        spec=savgol_params(ctx).spec(row_seconds),
    )
