"""Rest of an object in the image plane; not a plugin.

The mask's centroid and equivalent radius r_eq = sqrt(area / pi) give a pixel track with
sigma_px = e_c r_eq per component, e_c the profile's TRACK centroid error. Its velocity and linear
sigma come from the Savitzky-Golay differentiation of the kinematics stage, with that stage's
window and order; rows and runs are tested as in ``_velocity_tests.py`` with two components.
Under a fixed camera flight and rest differ by orders of magnitude here, while the 3D test,
whose sigma_v scales with the object's own speed, passes the apex of a throw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._velocity_tests import rest_runs
from physverify.extraction.providers.kinematics_savgol import Params as SavgolParams
from physverify.physics.kinematics import SavgolSpec, TrackDerivatives, differentiate
from physverify.physics.row_time import seconds_per_row
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import CENTROID_PX_REL, ErrorEntry
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PixelError:
    """Relative centroid error e_c (NaN when unknown); sigma_px = e_c * r_eq per component."""

    relative: float

    @classmethod
    def of(cls, entry: ErrorEntry | None) -> PixelError:
        return cls(entry_rms(entry))

    @property
    def known(self) -> bool:
        return math.isfinite(self.relative)

    def sigma(self, r_eq: FloatArray) -> FloatArray:
        return np.asarray(self.relative * np.asarray(r_eq, dtype=np.float64), dtype=np.float64)


def mask_motion(
    masks: BoolArray, visible: BoolArray, t: FloatArray, spec: SavgolSpec, error: PixelError
) -> tuple[TrackDerivatives, BoolArray]:
    """Velocity (T, 2) of the mask centroid with its linear sigma, and the rows that have a
    mask; rows without one are outside every run."""
    masks = np.asarray(masks, dtype=np.bool_)
    area = masks.sum(axis=(1, 2)).astype(np.float64)
    valid = np.asarray(visible, dtype=np.bool_) & (area > 0)
    height, width = masks.shape[1:]
    safe = np.where(area > 0, area, 1.0)
    cx = (masks.sum(axis=1) * np.arange(width, dtype=np.float64)).sum(axis=1) / safe
    cy = (masks.sum(axis=2) * np.arange(height, dtype=np.float64)).sum(axis=1) / safe
    sigma = np.where(valid, error.sigma(np.sqrt(area / math.pi)), np.nan)
    position = np.where(valid[:, None], np.stack([cx, cy], axis=-1), np.nan)
    sigmas = np.repeat(sigma[:, None], 2, axis=1)
    return differentiate(np.asarray(t, dtype=np.float64), position, sigmas, valid, spec), valid


def track_rows(track: MaskTrack, states: ObjectStateTable) -> IntArray:
    """Row of ``track`` at each row of ``states`` by frame index; -1 where it has none."""
    at = {int(frame): row for row, frame in enumerate(track.sampling.frame_indices.tolist())}
    return np.array([at.get(int(f), -1) for f in states.frame_index.tolist()], dtype=np.int64)


@dataclass(frozen=True)
class ImageRest:
    """Rest runs of each object's mask, matched to the states by object id and frame index; an
    object without a mask has no rest."""

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
        rows = track_rows(self.track, states)
        present = rows >= 0
        columns = [str(i) for i in self.track.object_ids.tolist()]
        for n, object_id in enumerate(states.object_ids.tolist()):
            if str(object_id) not in columns:
                continue
            masks = self.object_masks(columns.index(str(object_id)), rows, present)
            visible = states.visible[:, n] & present
            motion, valid = mask_motion(masks, visible, states.t, self.spec, self.error)
            sigma = motion.velocity_sigma[:, 0]
            for run in rest_runs(motion.velocity, valid, sigma, alpha, min_rows):
                rest[run, n] = True
        return rest

    def object_masks(self, column: int, rows: IntArray, present: BoolArray) -> BoolArray:
        height, width = self.track.masks.shape[2:]
        masks = np.zeros((rows.shape[0], height, width), dtype=np.bool_)
        masks[present] = self.track.masks[rows[present], column]
        return masks


def image_rest(ctx: StageContext, states: ObjectStateTable) -> ImageRest:
    """The image rest of the clip's masks, with the kinematics stage's window at its row rate."""
    row_seconds = seconds_per_row(states.frame_index, ctx.sampling.seconds_per_frame)
    return ImageRest(
        track=ctx.require(Stage.TRACK, MaskTrack),
        error=PixelError.of(ctx.error_entry(Stage.TRACK, CENTROID_PX_REL)),
        spec=SavgolParams().spec(row_seconds),
    )
