"""Pixel tracks of the objects' masks; not a plugin.

Each object's mask, matched to the states by object id and frame index, gives per row its
centroid and equivalent radius r_eq = sqrt(area / pi), with sigma_px = max(e_c r_eq, 1 / sqrt(12))
per component, e_c the profile's TRACK centroid error (zero for a ground-truth TRACK). Its
velocity and linear sigma come from the Savitzky-Golay differentiation of the kinematics stage.
``ImageProjection`` gives the image direction of a 3D direction at an object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._impact_attribution import camera_from_world, camera_positions
from physverify.physics.kinematics import SavgolSpec, TrackDerivatives, differentiate
from physverify.physics.tracks import PIXEL_QUANTIZATION_SIGMA
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import CENTROID_PX_REL, ErrorEntry
from physverify.schema.perception import CameraModel, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PixelError:
    """Relative centroid error e_c (NaN when unknown); sigma_px = max(e_c r_eq, 1 / sqrt(12))
    per component."""

    relative: float

    @classmethod
    def of(cls, entry: ErrorEntry | None) -> PixelError:
        return cls(entry_rms(entry))

    @property
    def known(self) -> bool:
        return math.isfinite(self.relative)

    def sigma(self, r_eq: FloatArray) -> FloatArray:
        # Masks are whole pixels: an exact track still carries the quantization of its centroid.
        scaled = self.relative * np.asarray(r_eq, dtype=np.float64)
        return np.asarray(np.maximum(scaled, PIXEL_QUANTIZATION_SIGMA), dtype=np.float64)


def pixel_error(ctx: StageContext) -> PixelError:
    return PixelError.of(ctx.error_entry(Stage.TRACK, CENTROID_PX_REL))


@dataclass(frozen=True)
class ObjectImage:
    """One object's masks (T, H, W) on the rows of the states, the rows that have one
    (``valid``), and per row the centroid (T, 2), NaN elsewhere, and r_eq (T,)."""

    masks: BoolArray
    valid: BoolArray
    centroid: FloatArray
    radius: FloatArray

    @classmethod
    def of(cls, masks: BoolArray, visible: BoolArray) -> ObjectImage:
        masks = np.asarray(masks, dtype=np.bool_)
        area = masks.sum(axis=(1, 2)).astype(np.float64)
        valid = np.asarray(visible, dtype=np.bool_) & (area > 0)
        height, width = masks.shape[1:]
        safe = np.where(area > 0, area, 1.0)
        cx = (masks.sum(axis=1) * np.arange(width, dtype=np.float64)).sum(axis=1) / safe
        cy = (masks.sum(axis=2) * np.arange(height, dtype=np.float64)).sum(axis=1) / safe
        centroid = np.where(valid[:, None], np.stack([cx, cy], axis=-1), np.nan)
        return cls(masks, valid, centroid, np.sqrt(area / math.pi))


def mask_motion(
    image: ObjectImage, t: FloatArray, spec: SavgolSpec, error: PixelError
) -> TrackDerivatives:
    """Velocity (T, 2) of the mask centroid with its linear sigma; rows without a mask are
    outside every run."""
    sigma = np.where(image.valid, error.sigma(image.radius), np.nan)
    sigmas = np.repeat(sigma[:, None], 2, axis=1)
    return differentiate(np.asarray(t, dtype=np.float64), image.centroid, sigmas, image.valid, spec)


def track_rows(track: MaskTrack, states: ObjectStateTable) -> IntArray:
    """Row of ``track`` at each row of ``states`` by frame index; -1 where it has none."""
    at = {int(frame): row for row, frame in enumerate(track.sampling.frame_indices.tolist())}
    return np.array([at.get(int(f), -1) for f in states.frame_index.tolist()], dtype=np.int64)


def object_images(track: MaskTrack, states: ObjectStateTable) -> tuple[ObjectImage | None, ...]:
    """The image of every object of the states; None for an object the track does not hold."""
    rows = track_rows(track, states)
    present = rows >= 0
    columns = [str(i) for i in track.object_ids.tolist()]
    height, width = track.masks.shape[2:]
    images: list[ObjectImage | None] = []
    for n, object_id in enumerate(states.object_ids.tolist()):
        if str(object_id) not in columns:
            images.append(None)
            continue
        masks = np.zeros((rows.shape[0], height, width), dtype=np.bool_)
        masks[present] = track.masks[rows[present], columns.index(str(object_id))]
        images.append(ObjectImage.of(masks, states.visible[:, n] & present))
    return tuple(images)


@dataclass(frozen=True)
class ImageProjection:
    """Positions (T, N, 3) and rotations (T, 3, 3) in CAMERA_OPENCV, None without a camera pose
    that fits the rows, and the focal lengths (fx, fy) in pixels."""

    positions: FloatArray | None
    rotations: FloatArray | None
    focal: FloatArray

    @classmethod
    def of(cls, states: ObjectStateTable, camera: CameraModel) -> ImageProjection:
        transform = camera_from_world(states, camera)
        rotations = None if transform is None else transform[:, :3, :3]
        return cls(camera_positions(states, camera), rotations, np.diag(camera.K)[:2])

    def direction(self, n: int, row: int, vector: FloatArray) -> FloatArray | None:
        """Unit image direction (pixels, y down) of the 3D ``vector`` at object n: the
        derivative of the pinhole projection, (fx (u_x z - x u_z), fy (u_y z - y u_z)) / z^2, at
        the nearest row where the object lies in front of the camera; None without one."""
        if self.positions is None or self.rotations is None:
            return None
        position = self.positions[:, n]
        ahead = np.flatnonzero(np.all(np.isfinite(position), axis=-1) & (position[:, 2] > 0.0))
        if ahead.size == 0:
            return None
        nearest = int(ahead[np.argmin(np.abs(ahead - row))])
        x, y, z = position[nearest]
        u = self.rotations[nearest] @ np.asarray(vector, dtype=np.float64)
        moved = self.focal * np.array([u[0] * z - x * u[2], u[1] * z - y * u[2]]) / (z * z)
        norm = float(np.linalg.norm(moved))
        return moved / norm if norm > 0.0 else None
