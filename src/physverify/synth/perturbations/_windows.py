"""Frame windows, rigid solids and projected silhouettes over ground-truth state tables.

Pixel coordinates follow Kubric's ``project_point``: pixel (u, v) covers [u, u + 1) x [v, v + 1),
so the image is the rectangle [0, W] x [0, H] and pixel centres sit at half-integers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import (
    CYLINDER_SIDES,
    Primitive,
    Solid,
    convex_hull_indices,
    cross_2d,
    polygon_area,
    pose_from_quaternion,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.layout import project

Vec = NDArray[np.float64]
Mask = NDArray[np.bool_]
UP: Vec = np.array([0.0, 0.0, 1.0])
ON_TOP_COS = math.cos(math.radians(30.0))
SPHERE_RIM_POINTS = 64


def solid_at(states: ObjectStateTable, index: int, frame: int, offset: Vec | None = None) -> Solid:
    """The rigid primitive of object ``index`` at ``frame`` (optionally translated)."""
    position = states.position[frame, index] + (0.0 if offset is None else offset)
    primitive = Primitive(PrimitiveKind(str(states.primitive[index])), states.size[frame, index])
    return Solid(primitive, pose_from_quaternion(position, states.quaternion[frame, index]))


def records_of(object_id: str, contacts: ContactTable) -> list[ContactRecord]:
    return [r for r in contacts.records if object_id in (r.object_a, r.object_b)]


def other_of(record: ContactRecord, object_id: str) -> str:
    return record.object_b if record.object_a == object_id else record.object_a


def flight_mask(object_id: str, contacts: ContactTable, n_frames: int, margin: int = 1) -> Mask:
    """Frames with no contact record of the object within [t_start - margin, t_end + margin]."""
    mask = np.ones(n_frames, dtype=bool)
    for record in records_of(object_id, contacts):
        lo = max(0, math.floor(record.t_start) - margin)
        hi = min(n_frames - 1, math.ceil(record.t_end) + margin)
        mask[lo : hi + 1] = False
    return mask


def segments(mask: Mask, min_len: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs of True with at least ``min_len`` frames."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for k, flag in enumerate([*np.asarray(mask, dtype=bool).tolist(), False]):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            if k - start >= min_len:
                runs.append((start, k))
            start = None
    return runs


def rest_mask(states: ObjectStateTable, index: int, v_tol: float = 0.02) -> Mask:
    speed = np.linalg.norm(states.velocity[:, index], axis=1)
    out: Mask = speed < v_tol
    return out


def impact_frames(object_id: str, contacts: ContactTable, n_frames: int) -> Mask:
    """Rows k whose jump_ratio stencil (frames k-1..k+2) touches an impact record of the object:
    floor(t_start) - 1 <= k <= ceil(t_end) (the only rows jump_ratio skips)."""
    mask = np.zeros(n_frames, dtype=bool)
    for record in records_of(object_id, contacts):
        if record.kind is ContactKind.IMPACT:
            lo = max(0, math.floor(record.t_start) - 1)
            hi = min(n_frames - 1, math.ceil(record.t_end))
            mask[lo : hi + 1] = True
    return mask


def on_top_pairs(contacts: ContactTable, up: Vec = UP) -> set[tuple[str, str]]:
    """(upper, lower) for every resting record whose normal, oriented lower -> upper, is within
    30 degrees of ``up``; ``lower`` may be the environment."""
    pairs: set[tuple[str, str]] = set()
    for record in contacts.records:
        if record.kind is not ContactKind.RESTING:
            continue
        along = float(np.asarray(record.normal) @ up)
        if along >= ON_TOP_COS:
            pairs.add((record.object_a, record.object_b))
        elif -along >= ON_TOP_COS and record.object_b != ENV_ID:
            pairs.add((record.object_b, record.object_a))
    return pairs


def supports_nothing(object_id: str, contacts: ContactTable) -> bool:
    return all(lower != object_id for _, lower in on_top_pairs(contacts))


def touching(object_id: str, contacts: ContactTable) -> set[str]:
    return {other_of(r, object_id) for r in records_of(object_id, contacts)}


def silhouette_points(solid: Solid, eye: Vec) -> Vec:
    """World points whose projected convex hull is the solid's silhouette seen from ``eye``."""
    primitive, pose = solid.primitive, solid.pose
    if primitive.shape == PrimitiveKind.SPHERE.value:
        return _sphere_rim(pose.position, primitive.radius, eye)
    if primitive.shape == PrimitiveKind.CYLINDER.value:
        angles = np.arange(CYLINDER_SIDES) * 2.0 * math.pi / CYLINDER_SIDES
        ring = np.column_stack([np.cos(angles), np.sin(angles)]) * primitive.radius
        half_h = float(primitive.half[2])
        local = np.vstack(
            [np.column_stack([ring, np.full(len(ring), z)]) for z in (-half_h, half_h)]
        )
        return pose.to_world(local)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return pose.to_world(signs * primitive.half)


def _sphere_rim(center: Vec, radius: float, eye: Vec) -> Vec:
    """Tangency circle of the viewing cone: at r^2/d from the centre, radius r sqrt(1-r^2/d^2)."""
    axis = eye - center
    d = float(np.linalg.norm(axis))
    axis /= d
    helper = UP if abs(float(axis @ UP)) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    rho = radius * math.sqrt(max(0.0, 1.0 - (radius / d) ** 2))
    phi = np.arange(SPHERE_RIM_POINTS) * 2.0 * math.pi / SPHERE_RIM_POINTS
    rim_center = center + axis * radius**2 / d
    return rim_center + rho * (np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))


def projected_silhouette(K: Vec, world_from_camera: Vec, solid: Solid) -> Vec:  # noqa: N803
    """Convex polygon (M, 2) in pixels; empty when any silhouette point is behind the camera."""
    uv, z = project(silhouette_points(solid, world_from_camera[:3, 3]), K, world_from_camera)
    if bool(np.any(z <= 1e-6)):
        return np.zeros((0, 2))
    hull: Vec = uv[convex_hull_indices(uv)]
    return hull


def polygon_area_2d(polygon: Vec) -> float:
    """Area of a convex pixel polygon (``polygon_area`` on the image plane)."""
    return polygon_area(np.column_stack([polygon, np.zeros(polygon.shape[0])]), UP)


def clip_to_rect(polygon: Vec, width: float, height: float) -> Vec:
    """Sutherland-Hodgman clip of a convex polygon to [0, width] x [0, height]."""
    out = polygon
    for axis, bound, keep_below in (
        (0, 0.0, False),
        (0, width, True),
        (1, 0.0, False),
        (1, height, True),
    ):
        out = _clip_half_plane(out, axis, bound, keep_below)
        if out.shape[0] == 0:
            break
    return out


def _clip_half_plane(polygon: Vec, axis: int, bound: float, keep_below: bool) -> Vec:
    def inside(p: Vec) -> bool:
        return bool(p[axis] <= bound) if keep_below else bool(p[axis] >= bound)

    kept: list[Vec] = []
    for k in range(polygon.shape[0]):
        cur, prev = polygon[k], polygon[k - 1]
        if inside(cur):
            if not inside(prev):
                kept.append(_cross_point(prev, cur, axis, bound))
            kept.append(cur)
        elif inside(prev):
            kept.append(_cross_point(prev, cur, axis, bound))
    return np.array(kept, dtype=np.float64).reshape(-1, 2)


def _cross_point(a: Vec, b: Vec, axis: int, bound: float) -> Vec:
    s = (bound - a[axis]) / (b[axis] - a[axis])
    out: Vec = a + s * (b - a)
    return out


def in_image_fraction(polygon: Vec, size: tuple[int, int]) -> float:
    """Share of the silhouette area inside the image (0 for an empty silhouette)."""
    area = polygon_area_2d(polygon)
    if area <= 0.0:
        return 0.0
    return polygon_area_2d(clip_to_rect(polygon, float(size[0]), float(size[1]))) / area


def polygons_overlap(a: Vec, b: Vec) -> bool:
    """Separating-axis test for two convex polygons (touching counts as overlap)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return False
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for normal in np.column_stack([-edges[:, 1], edges[:, 0]]):
            pa, pb = a @ normal, b @ normal
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def silhouette_mask(polygon: Vec, size: tuple[int, int]) -> Mask:
    """(H, W) pixels whose centre lies inside the convex polygon."""
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    if polygon.shape[0] < 3:
        return mask
    x0, y0 = np.clip(np.floor(polygon.min(axis=0)).astype(int), 0, [width, height])
    x1, y1 = np.clip(np.ceil(polygon.max(axis=0)).astype(int), 0, [width, height])
    if x1 <= x0 or y1 <= y0:
        return mask
    xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    ccw = cross_2d(polygon[0], polygon[1], polygon[2]) > 0
    inside = np.ones(xs.shape, dtype=bool)
    for k in range(polygon.shape[0]):
        a, b = polygon[k - 1], polygon[k]
        cross = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        inside &= cross >= 0 if ccw else cross <= 0
    mask[y0:y1, x0:x1] = inside
    return mask


def centre_in_image(uv: Vec, size: tuple[int, int]) -> bool:
    return bool(0.0 <= uv[0] <= size[0] and 0.0 <= uv[1] <= size[1])


def all_ids(states: ObjectStateTable) -> Iterable[tuple[int, str]]:
    return enumerate(states.object_ids.astype(str).tolist())
