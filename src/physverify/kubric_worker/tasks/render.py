"""Render one clip per job with Blender (through Kubric) from keyframes of its variant states.

Job keys: clip_id, spec_path, states_path, layers_path, meta_path, failed_path, floor. Each
object gets position, quaternion and scale keyframes per native frame; where it is not visible
it sits under the opaque floor. ``kubric_meta.json`` is written last and marks the job done.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.kubric_worker.build_scene import (
    clip_inputs,
    guarded,
    hidden_position,
    kubric_frame,
    kubric_scale,
    open_scene,
    probe_points,
    save_record,
    write_json,
)
from physverify.schema.render import RenderLayers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physverify.kubric_worker.build_scene import BuiltScene
    from physverify.schema.states import ObjectStateTable

LAYERS = ("rgba", "segmentation", "z")
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


def run(job: dict[str, Any]) -> None:
    guarded(job, render_clip)


def render_clip(job: dict[str, Any]) -> None:
    spec, states = clip_inputs(job)
    with open_scene(spec, job["floor"], render=True) as built:
        for k, (asset, bounds) in enumerate(zip(built.objects, built.bounds)):
            extents = spec.objects[k].extents
            for i, (position, quaternion, scale) in enumerate(
                pose_keyframes(states, k, bounds, extents)
            ):
                _keyframe(asset, kubric_frame(i), position, quaternion, scale)
        frame_seconds = _render_frames(built.renderer, spec.n_frames)
        start = time.perf_counter()
        out = built.renderer.postprocess(built.renderer.scratch_dir, return_layers=LAYERS)
        segmentation = object_segmentation(out["segmentation"], built)
        postprocess_seconds = time.perf_counter() - start
        camera = built.scene.camera
        probes = probe_points(spec.camera.look_at)
        projections = [np.asarray(camera.project_point(p), dtype=np.float64) for p in probes]
        meta = {
            "clip_id": job["clip_id"],
            "frame_seconds": frame_seconds,
            "postprocess_seconds": postprocess_seconds,
            "k_norm": np.asarray(camera.intrinsics, dtype=np.float64).tolist(),
            "matrix_world": np.asarray(camera.matrix_world, dtype=np.float64).tolist(),
            "probe_points": probes.tolist(),
            "probe_projections": [p.tolist() for p in projections],
            "resolution": list(spec.resolution),
            "spp": spec.spp,
            "background": spec.background,
        }
    layers = RenderLayers(
        clip_id=job["clip_id"],
        frame_indices=np.arange(spec.n_frames, dtype=np.int64),
        rgba=to_uint8(out["rgba"]),
        segmentation=segmentation,
        z=np.asarray(out["z"])[..., 0].astype(np.float32),
        object_ids=np.asarray([o.object_id for o in spec.objects], dtype=str),
    )
    save_record(layers, Path(job["layers_path"]))
    write_json(Path(job["meta_path"]), meta)


def object_segmentation(raw: Any, built: BuiltScene) -> np.ndarray:
    """Kubric's postprocess labels pixels with the 1-based index in ``scene.assets``; this maps
    them to each asset's ``segmentation_id`` (k + 1 for object k, 0 for floor and dome)."""
    import kubric as kb

    ids = kb.adjust_segmentation_idxs(np.asarray(raw), built.scene.assets, built.objects)
    return np.asarray(ids)[..., 0].astype(np.uint32)


def pose_keyframes(
    states: ObjectStateTable,
    k: int,
    bounds: Sequence[Sequence[float]],
    extents: Sequence[float],
) -> list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, float, float]]]:
    """Position, quaternion (wxyz) and Kubric scale of object ``k`` per native frame: a hidden
    pose where it is not visible or its position is unknown, the identity for an unknown
    orientation and the spec extents for an unknown size."""
    keys = []
    for t in range(states.t.shape[0]):
        position = states.position[t, k]
        shown = bool(states.visible[t, k]) and bool(np.isfinite(position).all())
        quaternion = states.quaternion[t, k]
        size = states.size[t, k]
        keys.append(
            (
                tuple(float(v) for v in position) if shown else hidden_position(position),
                tuple(float(v) for v in quaternion)
                if np.isfinite(quaternion).all()
                else IDENTITY_WXYZ,
                kubric_scale(size if np.isfinite(size).all() else extents, *bounds),
            )
        )
    return keys


def to_uint8(rgba: Any) -> np.ndarray:
    """Kubric reads the PNG at its bit depth; the schema stores 8 bits."""
    data = np.asarray(rgba)
    if data.dtype == np.uint16:
        return np.round(data / 257.0).astype(np.uint8)
    return data.astype(np.uint8)


def _keyframe(
    asset: Any,
    frame: int,
    position: Sequence[float],
    quaternion: Sequence[float],
    scale: Sequence[float],
) -> None:
    asset.position = tuple(position)
    asset.quaternion = tuple(quaternion)
    asset.scale = tuple(scale)
    for attribute in ("position", "quaternion", "scale"):
        asset.keyframe_insert(attribute, frame)


def _render_frames(renderer: Any, n_frames: int) -> list[float]:
    """Kubric's ``Blender.render`` loop, timed per frame (it only reports the total)."""
    import bpy
    from kubric.redirect_io import RedirectStream

    scratch = Path(renderer.scratch_dir)
    renderer.set_exr_output_path(scratch / "exr" / "frame_")
    seconds = []
    with RedirectStream(stream=sys.stdout, disabled=renderer.verbose):
        for i in range(n_frames):
            frame = kubric_frame(i)
            start = time.perf_counter()
            bpy.context.scene.frame_set(frame)
            bpy.context.scene.render.filepath = str(scratch / "images" / f"frame_{frame:04d}.png")
            bpy.ops.render.render(animation=False, write_still=True)
            seconds.append(time.perf_counter() - start)
    return seconds
