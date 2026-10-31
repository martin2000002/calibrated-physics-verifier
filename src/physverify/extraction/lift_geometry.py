"""Upright pose and local size of a lifted object from one view (LIFT v5).

Frames: the camera frame is OpenCV (x right, y down, z forward). The ground frame has columns
(h1, h2, up): up = -gravity, h1 = the camera x axis made horizontal, h2 = up x h1. It is
``assumed_upright_pose`` with the camera x axis as yaw axis, so a yaw psi about up gives the
local frame x = cos(psi) h1 + sin(psi) h2, y = up x x, z = up, the convention ``place_body``
already applies to an object without a quaternion. The quaternion (w, x, y, z) maps local to
camera coordinates; ``size`` is then the full extent along those local axes.

Silhouette lengths are kept in normalised image units (x/z, y/z) and become metric once
multiplied by the object's depth, so a depth or focal change rescales them like the extents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.physics.contact_manifold import convex_hull_indices
from physverify.physics.geometry import assumed_upright_pose, quaternion_from_matrix
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import ErrorEntry
from physverify.schema.states import PrimitiveKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CAMERA_X = np.array([1.0, 0.0, 0.0])
MIN_FOOTPRINT_POINTS = 6
MIN_COS_ELEVATION = 0.25
"""Steeper than about 75 degrees the image height no longer shows the object's height, which is
then assumed equal to its horizontal size."""
MIN_HEIGHT_FRAC = 0.1
"""Height floor relative to the horizontal size, against a negative estimate from mask noise."""
MAD_TO_SIGMA = 1.4826
MIN_UP_IMAGE_NORM = 1e-6


@dataclass(frozen=True)
class GroundFrame:
    """Columns (h1, h2, up) in camera coordinates."""

    basis: FloatArray

    @property
    def up(self) -> FloatArray:
        up: FloatArray = self.basis[:, 2]
        return up

    def rotation(self, yaw: float) -> FloatArray:
        """Local-to-camera rotation of an upright object at ``yaw`` (radians about up)."""
        c, s = math.cos(yaw), math.sin(yaw)
        about_up = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotation: FloatArray = self.basis @ about_up
        return rotation

    def horizontal_angle(self, vector: FloatArray) -> float:
        """Angle about up of ``vector``'s horizontal part, from h1 towards h2."""
        return math.atan2(float(vector @ self.basis[:, 1]), float(vector @ self.basis[:, 0]))


def ground_frame(gravity_dir: FloatArray) -> GroundFrame | None:
    """None when gravity is unknown (non-finite)."""
    if not bool(np.all(np.isfinite(gravity_dir))):
        return None
    pose = assumed_upright_pose(np.zeros(3), gravity_dir, CAMERA_X)
    return GroundFrame(basis=pose.rotation)


def minimum_area_rectangle(points: FloatArray) -> tuple[float, FloatArray] | None:
    """(angle in [0, pi/2) of one side, side lengths) of the smallest enclosing rectangle;
    None when the points are collinear. One side lies on a hull edge (rotating calipers)."""
    hull = np.asarray(points, dtype=np.float64)[convex_hull_indices(points)]
    if hull.shape[0] < 3:
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2.0)
    best: tuple[float, float, FloatArray] | None = None
    for angle in np.unique(angles):
        c, s = math.cos(angle), math.sin(angle)
        rotated = hull @ np.array([[c, -s], [s, c]])
        sides = np.ptp(rotated, axis=0)
        area = float(sides[0] * sides[1])
        if best is None or area < best[0]:
            best = (area, float(angle), sides)
    if best is None or best[0] <= 0.0:
        return None
    return best[1], best[2]


def footprint_yaw(points: FloatArray, frame: GroundFrame, trim_mads: float) -> float:
    """Yaw in [0, pi/2) of a box's footprint: the smallest rectangle around the visible surface
    projected on the ground plane, after dropping points farther from the median than
    median + ``trim_mads`` robust sigma (depth bleeding at the silhouette). NaN when degenerate."""
    flat = points @ frame.basis[:, :2]
    radius = np.linalg.norm(flat - np.median(flat, axis=0), axis=1)
    centre_r = float(np.median(radius))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(radius - centre_r)))
    kept = flat[radius <= centre_r + trim_mads * spread]
    if kept.shape[0] < MIN_FOOTPRINT_POINTS:
        return math.nan
    rectangle = minimum_area_rectangle(kept)
    return math.nan if rectangle is None else rectangle[0]


@dataclass(frozen=True)
class SilhouetteView:
    """The silhouette against the vertical, in normalised image units.

    ``across`` and ``along`` are its widths perpendicular to and along the image of the up
    direction at the object; ``area`` is its area; ``sin_elevation`` is |ray . gravity|;
    ``yaw`` the footprint yaw (NaN when unknown) and ``ray_yaw`` the viewing ray's, both about
    up in the ground frame."""

    across: float
    along: float
    area: float
    sin_elevation: float
    yaw: float
    ray_yaw: float


def image_up(centre: FloatArray, up: FloatArray) -> FloatArray | None:
    """Unit direction, in normalised image coordinates, of the image of ``up`` at ``centre``:
    d(x/z, y/z) = (u_x - x0 u_z, u_y - y0 u_z) / z. None when up points along the ray."""
    x0, y0 = centre[0] / centre[2], centre[1] / centre[2]
    direction = np.array([up[0] - x0 * up[2], up[1] - y0 * up[2]])
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > MIN_UP_IMAGE_NORM else None


def percentile_width(
    coords: FloatArray, direction: FloatArray, percentiles: tuple[float, float], pixel: FloatArray
) -> float:
    """Percentile spread of the pixel centres along ``direction`` plus one pixel's own extent."""
    values = coords @ direction
    lo, hi = np.percentile(values, list(percentiles))
    return float(hi - lo) + float(np.abs(direction) @ pixel)


def silhouette_view(
    mask: BoolArray,
    intrinsics: FloatArray,
    centre: FloatArray,
    surface: FloatArray,
    frame: GroundFrame,
    percentiles: tuple[float, float],
    trim_mads: float,
) -> SilhouetteView | None:
    """The view of one silhouette; None when up is seen end-on."""
    along_dir = image_up(centre, frame.up)
    if along_dir is None:
        return None
    across_dir = np.array([-along_dir[1], along_dir[0]])
    focal = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    principal = np.array([intrinsics[0, 2], intrinsics[1, 2]])
    rows, cols = np.nonzero(mask)
    coords = (np.column_stack([cols, rows]) + 0.5 - principal) / focal
    pixel = 1.0 / focal
    ray = centre / float(np.linalg.norm(centre))
    return SilhouetteView(
        across=percentile_width(coords, across_dir, percentiles, pixel),
        along=percentile_width(coords, along_dir, percentiles, pixel),
        area=float(rows.size) / float(focal[0] * focal[1]),
        sin_elevation=min(abs(float(ray @ frame.up)), 1.0),
        yaw=footprint_yaw(surface, frame, trim_mads),
        ray_yaw=frame.horizontal_angle(ray),
    )


@dataclass(frozen=True)
class LocalGeometry:
    """Full extents along the local axes and the local-to-camera quaternion (w, x, y, z)."""

    size: FloatArray
    quaternion: FloatArray


def _height(along: float, footprint_depth: float, sin_e: float, horizontal: float) -> float:
    """h from the image height of an upright solid, along = h cos(e) + D sin(e), where D is
    the footprint's extent along the horizontal viewing ray (orthographic)."""
    cos_e = math.sqrt(max(1.0 - sin_e**2, 0.0))
    if cos_e < MIN_COS_ELEVATION:
        return horizontal
    return max((along - footprint_depth * sin_e) / cos_e, MIN_HEIGHT_FRAC * horizontal)


def local_geometry(
    primitive: PrimitiveKind, view: SilhouetteView, depth: float, frame: GroundFrame
) -> LocalGeometry | None:
    """Upright solid seen in ``view`` at ``depth``; None for a box whose yaw is unknown.

    Spheres and cylinders are symmetric about up, so their yaw is 0. A box (and an unknown
    shape) is assumed to have a square footprint of side s: its silhouette width across the ray
    is s (|cos phi| + |sin phi|), phi being its yaw from the across-ray direction, and so is
    its footprint's extent along the ray."""
    across, along = view.across * depth, view.along * depth
    if primitive is PrimitiveKind.SPHERE:
        return _upright(np.full(3, across), 0.0, frame)
    if primitive is PrimitiveKind.CYLINDER:
        height = _height(along, across, view.sin_elevation, across)
        return _upright(np.array([across, across, height]), 0.0, frame)
    if not math.isfinite(view.yaw):
        return None
    phi = view.yaw - view.ray_yaw - math.pi / 2.0
    side = across / (abs(math.cos(phi)) + abs(math.sin(phi)))
    height = _height(along, across, view.sin_elevation, side)
    return _upright(np.array([side, side, height]), view.yaw, frame)


def _upright(size: FloatArray, yaw: float, frame: GroundFrame) -> LocalGeometry:
    return LocalGeometry(size=size, quaternion=quaternion_from_matrix(frame.rotation(yaw)))


def orientation_sigma(entry: ErrorEntry | None, gravity_sigma_rad: float) -> float:
    """Radians: the measured orientation error and the gravity error in quadrature; NaN when
    either is unknown."""
    return math.hypot(entry_rms(entry), gravity_sigma_rad)
