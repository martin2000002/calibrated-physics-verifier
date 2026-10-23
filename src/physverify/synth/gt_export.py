"""Canonical ground truth of every rendered clip: PNG frames, WORLD and
CAMERA state and contact tables, OpenCV camera, final perturbation record, ``gt_meta.json`` (the
step marker, written last) and the clip manifest; ``manifest.jsonl`` indexes the exported clips.
``video.mp4`` is a lossy copy of the frames for viewing; extraction reads the PNG frames.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from physverify.core.atomic import write_bytes_atomic
from physverify.core.cache_keys import params_hash
from physverify.core.digests import file_sha256
from physverify.core.video import write_mp4
from physverify.schema.clip import ClipManifest, ObjectQuery, TimeSource
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.io import build, dumps, envelope, open_envelope
from physverify.schema.render import RenderLayers
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.schema.stages import STAGE_ORDER
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.gt_camera import (
    camera_contacts,
    camera_states,
    opencv_intrinsics,
    opencv_world_from_camera,
    world_states,
)
from physverify.synth.magnitudes import bounce_restitution, record_cuts
from physverify.synth.perturb import load_simulation, measured
from physverify.synth.store import (
    CAMERA,
    CLIP_MANIFEST,
    CONTACTS_CAMERA,
    CONTACTS_WORLD,
    FRAMES_DIR,
    GT_META,
    KUBRIC_META,
    LAYERS,
    RECORD,
    SPEC,
    STATES_CAMERA,
    STATES_WORLD,
    VARIANT_STATES,
    VIDEO,
    DatasetStore,
    Step,
    atomic_save,
    atomic_write_json,
    read_json,
)
from physverify.synth.truth_measures._base import TruthMeasure, measure_truth

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physverify.synth.plan import DatasetPlan, VariantPlan
    from physverify.synth.settings import DatasetSettings

DATASET_NAME = "kubric_synthetic"
LICENSE = "CC BY-SA 4.0 (KuBasic)"
HDRI_LICENSE = " + CC0 (HDRI Haven)"
FRAME_NAME = "{:06d}.png"
# Every file the ground-truth store reads; the frames are the PNG encoding of layers.rgba.
GT_FILES = (LAYERS, CAMERA, STATES_WORLD, STATES_CAMERA, CONTACTS_WORLD, CONTACTS_CAMERA)


def clean_restitution(
    store: DatasetStore, variant: VariantPlan, ds: DatasetSettings
) -> dict[str, float]:
    """Restitution of each sphere's first floor bounce in the unperturbed simulation of the scene
    (NaN without a bounce); it checks that the simulator adds no energy (e <= 1)."""
    if not store.is_done(Step.SIMULATE, variant.scene_id):
        return {}
    scene, states, contacts = load_simulation(store, variant.scene_id, variant.clip_id, ds)
    g = np.asarray(scene.gravity, dtype=np.float64)
    out: dict[str, float] = {}
    for index, object_id in enumerate(states.object_ids.astype(str).tolist()):
        if str(states.primitive[index]) != PrimitiveKind.SPHERE.value:
            continue
        impacts = sorted(
            r.t_start
            for r in contacts.records
            if r.kind is ContactKind.IMPACT and r.object_a == object_id and r.object_b == ENV_ID
        )
        out[object_id] = (
            bounce_restitution(states, index, impacts[0], g) if impacts else float("nan")
        )
    return out


def restitution_meta(e_clean: dict[str, float]) -> dict[str, Any]:
    """Unclamped ``e_clean`` plus the spheres whose clean bounce gained energy (e > 1): a
    simulator artifact measured on the clean clip, cause not verified."""
    gains = sorted(k for k, e in e_clean.items() if math.isfinite(e) and e > 1.0)
    return {"e_clean": e_clean, "e_clean_above_one": gains}


def final_record(
    plugin: Any,
    record: PerturbationRecord,
    variant: VariantPlan,
    states: ObjectStateTable,
    contacts: ContactTable,
) -> PerturbationRecord:
    """The record with the magnitude measured on the final truth; the planned target stays in
    ``params``."""
    magnitude = measured(plugin, states, contacts, record)
    params = {**record.params, "target_magnitude": variant.target_magnitude}
    return dataclasses.replace(record, magnitude=magnitude, params=params)


def write_frames(rgba: NDArray[np.uint8], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(rgba):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"cannot encode frame {index} of {directory}")
        write_bytes_atomic(directory / FRAME_NAME.format(index), encoded.tobytes())


def video_path(frames_dir: Path, data_dir: Path) -> str:
    """The frames directory relative to ``paths.data_dir`` (absolute when outside it)."""
    try:
        return frames_dir.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return frames_dir.resolve().as_posix()


def clip_manifest(
    variant: VariantPlan,
    scene: SceneSpec,
    record: PerturbationRecord,
    video: str,
) -> ClipManifest:
    counts = Counter(o.label for o in scene.objects)
    return ClipManifest(
        dataset=DATASET_NAME,
        clip_id=variant.clip_id,
        split=variant.split,
        video_path=video,
        n_frames=scene.n_frames,
        width=int(scene.resolution[0]),
        height=int(scene.resolution[1]),
        container_fps=float(scene.frame_rate),
        seconds_per_frame=1.0 / scene.frame_rate,
        time_source=TimeSource.GROUND_TRUTH,
        object_queries=tuple(
            ObjectQuery(phrase=label, expected_count=n) for label, n in sorted(counts.items())
        ),
        scene_id=variant.scene_id,
        perturbation=record,
        gt_stages=STAGE_ORDER,
        license=LICENSE + (HDRI_LICENSE if scene.background == "hdri" else ""),
    )


def content_sha256(path: Path) -> str:
    """sha256 of a file's content. For an NPZ it hashes each member's uncompressed bytes by name,
    so the zip container (compression, headers, the numpy that wrote it) does not change it."""
    if path.suffix != ".npz":
        return file_sha256(path)
    with zipfile.ZipFile(path) as archive:
        members = {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
        }
    return params_hash(members)


def gt_version(version: str, scene: SceneSpec, record: PerturbationRecord, clip_dir: Path) -> str:
    """Hash of the spec, the final record and the content of every exported ground-truth file:
    any change in the truth a consumer reads changes it."""
    return params_hash(
        {
            "version": version,
            "spec": scene.to_dict(),
            "record": record.to_dict(),
            "files": {name: content_sha256(clip_dir / name) for name in GT_FILES},
        }
    )


def export_clip(
    variant: VariantPlan,
    store: DatasetStore,
    plugin: Any,
    truth: Mapping[str, type[TruthMeasure]],
    ds: DatasetSettings,
    data_dir: Path,
) -> ClipManifest:
    clip_dir = store.clip_dir(variant.clip_id)
    scene = SceneSpec.load_json(store.scene_dir(variant.sim_id) / SPEC)
    layers = RenderLayers.load_npz(clip_dir / LAYERS)
    meta = read_json(clip_dir / KUBRIC_META)
    width, height = int(scene.resolution[0]), int(scene.resolution[1])
    K = opencv_intrinsics(meta["k_norm"], width, height)  # noqa: N806
    world_from_camera = opencv_world_from_camera(meta["matrix_world"])
    states = world_states(
        ObjectStateTable.load_npz(clip_dir / VARIANT_STATES), layers, K, world_from_camera
    )
    contacts = ContactTable.load_json(clip_dir / CONTACTS_WORLD)
    record = final_record(
        plugin, PerturbationRecord.load_json(clip_dir / RECORD), variant, states, contacts
    )
    write_frames(layers.rgba, clip_dir / FRAMES_DIR)
    write_mp4(layers.rgba[..., :3], clip_dir / VIDEO, float(scene.frame_rate))
    atomic_save(states, clip_dir / STATES_WORLD)
    atomic_save(camera_states(states, world_from_camera), clip_dir / STATES_CAMERA)
    atomic_save(camera_contacts(contacts, world_from_camera), clip_dir / CONTACTS_CAMERA)
    atomic_write_json(
        clip_dir / CAMERA,
        {
            "K": K.tolist(),
            "world_from_camera": world_from_camera.tolist(),
            "width": width,
            "height": height,
        },
    )
    manifest = clip_manifest(variant, scene, record, video_path(clip_dir / FRAMES_DIR, data_dir))
    atomic_save(manifest, clip_dir / CLIP_MANIFEST)
    atomic_write_json(
        clip_dir / GT_META,
        {
            "gt_version": gt_version(ds.version, scene, record, clip_dir),
            "sim_id": variant.sim_id,
            "record": record.to_dict(),
            "measures": measure_truth(
                truth, scene, states, contacts, ds.contacts.tol_m, record_cuts(record)
            ),
            **restitution_meta(clean_restitution(store, variant, ds)),
        },
    )
    return manifest


def export_all(
    plan: DatasetPlan,
    store: DatasetStore,
    plugins: Mapping[str, Any],
    truth: Mapping[str, type[TruthMeasure]],
    ds: DatasetSettings,
    data_dir: Path,
) -> list[str]:
    """Exports every rendered clip with contact truth that is not exported yet and rewrites
    ``manifest.jsonl``."""
    exported: list[str] = []
    for variant in plan.variants:
        clip_id = variant.clip_id
        if store.is_skipped(clip_id) or store.is_done(Step.EXPORT, clip_id):
            continue
        if not (store.is_done(Step.RENDER, clip_id) and store.is_done(Step.PERTURB, clip_id)):
            continue
        export_clip(variant, store, plugins[variant.perturbation], truth, ds, data_dir)
        exported.append(clip_id)
    write_index(store)
    return exported


def exported_manifests(store: DatasetStore) -> list[ClipManifest]:
    """Manifests of the clips whose export finished, sorted by clip id."""
    return [
        ClipManifest.load_json(path)
        for path in sorted((store.root / "clips").glob(f"*/{CLIP_MANIFEST}"))
        if (path.parent / GT_META).is_file()
    ]


def write_index(store: DatasetStore) -> None:
    """``manifest.jsonl``: one exported clip manifest (JSON envelope) per line."""
    lines = [dumps(envelope(m, m.to_dict())) + "\n" for m in exported_manifests(store)]
    write_bytes_atomic(store.manifest_path, "".join(lines).encode("utf-8"))


def read_index(store: DatasetStore) -> list[ClipManifest]:
    path = store.manifest_path
    if not path.is_file():
        return []
    return [
        build(ClipManifest, open_envelope(json.loads(line), ClipManifest.__name__))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
