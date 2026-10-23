"""Kubric's camera in the OpenCV convention: intrinsics, world-from-camera pose, per-object
visibility and the CAMERA-frame copies of the state and contact tables.

Kubric's camera is OpenGL (x right, y up, looking along -z) with intrinsics normalized to the
image size: u = fx * x / (-z) + 0.5 and v = -fy * y / (-z) + 0.5, times W and H in pixels.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.physics.finite_diff import central_difference
from physverify.physics.geometry import matrix_from_quaternion, quaternion_from_matrix
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.render import RenderLayers
from physverify.schema.states import CoordFrame, ObjectStateTable
from physverify.synth.perturbations._frame_windows import (
    in_image_fraction,
    projected_silhouette,
    silhouette_mask,
    solid_at,
)

Vec = NDArray[np.float64]
IN_FRAME_MIN_FRACTION = 0.5
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclasses.dataclass(frozen=True)
class Visibility:
    visible: NDArray[np.bool_]
    in_frame: NDArray[np.bool_]
    occlusion: Vec


def opencv_intrinsics(k_norm: Any, width: int, height: int) -> Vec:
    """Pixel K (x right, y down, z forward) from Kubric's normalized OpenGL intrinsics."""
    k = np.asarray(k_norm, dtype=np.float64)
    return np.array(
        [
            [k[0, 0] * width, 0.0, -k[0, 2] * width],
            [0.0, -k[1, 1] * height, -k[1, 2] * height],
            [0.0, 0.0, 1.0],
        ]
    )


def opencv_world_from_camera(matrix_world: Any) -> Vec:
    """Kubric's camera ``matrix_world`` with the OpenGL camera axes flipped to OpenCV."""
    return np.asarray(matrix_world, dtype=np.float64) @ OPENGL_TO_OPENCV


def visibility(
    layers: RenderLayers,
    states: ObjectStateTable,
    K: Vec,  # noqa: N803
    world_from_camera: Vec,
) -> Visibility:
    """``visible``: at least one segmentation pixel. ``in_frame``: at least half of the projected
    silhouette inside the image. ``occlusion``: share of the in-image silhouette pixels covered by
    another id nearer than the object's centre (NaN when unknown or fully outside)."""
    n_t, n_obj = states.n_frames, states.n_objects
    height, width = layers.segmentation.shape[1:]
    seg_ids = np.arange(1, n_obj + 1, dtype=np.uint32)
    visible = (layers.segmentation[:, None, :, :] == seg_ids[None, :, None, None]).any(axis=(2, 3))
    in_frame = np.zeros((n_t, n_obj), dtype=bool)
    occlusion = np.full((n_t, n_obj), np.nan)
    for t in range(n_t):
        for k in range(n_obj):
            pose_known = (
                np.isfinite(states.position[t, k]).all()
                and np.isfinite(states.quaternion[t, k]).all()
            )
            if not pose_known:
                continue
            polygon = projected_silhouette(K, world_from_camera, solid_at(states, k, t))
            in_frame[t, k] = in_image_fraction(polygon, (width, height)) >= IN_FRAME_MIN_FRACTION
            occlusion[t, k] = _occlusion(
                silhouette_mask(polygon, (width, height)),
                layers.segmentation[t],
                layers.z[t],
                int(seg_ids[k]),
                _depth(world_from_camera, states.position[t, k]),
            )
    return Visibility(visible=visible, in_frame=in_frame, occlusion=occlusion)


def _depth(world_from_camera: Vec, point: Vec) -> float:
    return float((point - world_from_camera[:3, 3]) @ world_from_camera[:3, 2])


def _occlusion(
    mask: NDArray[np.bool_], segmentation: NDArray[np.uint32], z: Any, own: int, depth: float
) -> float:
    area = int(mask.sum())
    if area == 0:
        return float("nan")
    z_mask = np.asarray(z, dtype=np.float64)[mask]
    nearer = (segmentation[mask] != own) & np.isfinite(z_mask) & (z_mask > 0) & (z_mask < depth)
    return float(nearer.sum()) / area


def world_states(
    states: ObjectStateTable,
    layers: RenderLayers,
    K: Vec,  # noqa: N803
    world_from_camera: Vec,
) -> ObjectStateTable:
    """The variant states as exact ground truth: zero sigmas, acceleration by central differences
    of the velocity (NaN at both ends) and visibility measured on the render."""
    seen = visibility(layers, states, K, world_from_camera)
    zeros3 = np.zeros(states.position.shape)
    zeros = np.zeros(states.position.shape[:2])
    return dataclasses.replace(
        states,
        position_sigma=zeros3,
        velocity_sigma=zeros3,
        acceleration=central_difference(states.velocity, states.t),
        acceleration_sigma=zeros3.copy(),
        size_sigma=zeros3.copy(),
        orientation_sigma=zeros,
        visible=seen.visible,
        in_frame=seen.in_frame,
        occlusion=seen.occlusion,
        confidence=np.ones(states.position.shape[:2]),
        gravity_dir_sigma=0.0,
    )


def _rotate(vectors: Vec, rotation: Vec) -> Vec:
    """Rows of ``vectors`` (any leading shape) rotated by ``rotation``: v' = R v."""
    rotated: Vec = np.asarray(vectors, dtype=np.float64) @ rotation.T
    return rotated


def camera_states(states: ObjectStateTable, world_from_camera: Vec) -> ObjectStateTable:
    """The WORLD table expressed in the OpenCV camera frame."""
    camera_from_world = np.linalg.inv(world_from_camera)
    rotation = camera_from_world[:3, :3]
    quaternion = np.full(states.quaternion.shape, np.nan)
    for t, k in np.ndindex(*states.quaternion.shape[:2]):
        q = states.quaternion[t, k]
        if np.isfinite(q).all():
            quaternion[t, k] = quaternion_from_matrix(rotation @ matrix_from_quaternion(q))
    return dataclasses.replace(
        states,
        frame=CoordFrame.CAMERA_OPENCV,
        position=_rotate(states.position, rotation) + camera_from_world[:3, 3],
        quaternion=quaternion,
        velocity=_rotate(states.velocity, rotation),
        acceleration=_rotate(states.acceleration, rotation),
        angular_velocity=_rotate(states.angular_velocity, rotation),
        gravity_dir=_rotate(states.gravity_dir, rotation),
    )


def camera_contacts(contacts: ContactTable, world_from_camera: Vec) -> ContactTable:
    camera_from_world = np.linalg.inv(world_from_camera)
    rotation = camera_from_world[:3, :3]

    def moved(record: ContactRecord) -> ContactRecord:
        normals = record.point_normals
        return dataclasses.replace(
            record,
            normal=_rotate(record.normal, rotation),
            points=_rotate(record.points, rotation) + camera_from_world[:3, 3],
            point_normals=None if normals is None else _rotate(normals, rotation),
        )

    return dataclasses.replace(
        contacts,
        frame=CoordFrame.CAMERA_OPENCV,
        records=tuple(moved(r) for r in contacts.records),
    )
