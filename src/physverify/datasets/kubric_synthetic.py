"""``kubric_synthetic``: the clips exported by ``physverify synth``.

Clips are generated locally, never downloaded. ``z`` at or beyond ``render.far_z_m`` is the
background (Blender's miss value) and becomes 0, the "no surface" value of ``NativeGroundTruth``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from physverify.core.errors import MissingInputError, UsageError
from physverify.core.video import FrameDirVideoSource
from physverify.datasets._base import ArrayGroundTruthStore, NativeGroundTruth
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.contacts import ContactTable
from physverify.schema.render import RenderLayers
from physverify.schema.scene import SceneSpec
from physverify.schema.states import CoordFrame, ObjectStateTable
from physverify.synth.generate import load_plan
from physverify.synth.gt_export import DATASET_NAME, read_index
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import (
    CAMERA,
    CONTACTS_CAMERA,
    CONTACTS_WORLD,
    GT_META,
    LAYERS,
    SPEC,
    STATES_CAMERA,
    STATES_WORLD,
    DatasetStore,
    read_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physverify.config.models import Settings

SPLITS = frozenset(s.value for s in Split if s is not Split.ALL)


def load_native(store: DatasetStore, clip_id: str, far_z_m: float) -> NativeGroundTruth:
    clip_dir = store.clip_dir(clip_id)
    meta = read_json(clip_dir / GT_META)
    camera = read_json(clip_dir / CAMERA)
    layers = RenderLayers.load_npz(clip_dir / LAYERS)
    states = ObjectStateTable.load_npz(clip_dir / STATES_WORLD)
    z = np.where(np.isfinite(layers.z) & (layers.z < far_z_m), layers.z, 0.0).astype(np.float32)
    return NativeGroundTruth(
        version=str(meta["gt_version"]),
        frame_indices=layers.frame_indices,
        segmentation=layers.segmentation,
        z=z,
        object_ids=layers.object_ids,
        labels=states.labels,
        K=np.asarray(camera["K"], dtype=np.float64),
        world_from_camera=np.asarray(camera["world_from_camera"], dtype=np.float64),
        states={
            CoordFrame.WORLD_ZUP: states,
            CoordFrame.CAMERA_OPENCV: ObjectStateTable.load_npz(clip_dir / STATES_CAMERA),
        },
        contacts={
            CoordFrame.WORLD_ZUP: ContactTable.load_json(clip_dir / CONTACTS_WORLD),
            CoordFrame.CAMERA_OPENCV: ContactTable.load_json(clip_dir / CONTACTS_CAMERA),
        },
        scene=SceneSpec.load_json(store.scene_dir(str(meta["sim_id"])) / SPEC),
    )


class KubricSynthetic:
    name: ClassVar[str] = DATASET_NAME

    def __init__(self, settings: Settings, root: Path | None = None) -> None:
        """``root`` defaults to the configured dataset version (``dataset_root``)."""
        ds = load_dataset_settings(settings)
        self._store = DatasetStore(root or dataset_root(settings, ds))
        self._far_z_m = ds.render.far_z_m
        self._data_dir = settings.paths.data_dir
        self._max_frame_bytes = settings.runtime.max_frame_bytes

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        """Exported clips: all (None), one split ("calib"/"eval") or one planned subset."""
        exported = [self._own(c) for c in read_index(self._store)]
        if subset is None:
            return exported
        if subset in SPLITS:
            return [c for c in exported if c.split.value == subset]
        plan = load_plan(self._store)
        if plan is None:
            raise MissingInputError(f"no dataset plan in {self._store.root}: run `synth generate`")
        wanted = {v.clip_id for v in plan.subset(subset).variants}
        return [c for c in exported if c.clip_id in wanted]

    def _own(self, clip: ClipManifest) -> ClipManifest:
        """Labelled with this plugin's name, which keys the stage cache and the sealed sets."""
        return clip if clip.dataset == self.name else replace(clip, dataset=self.name)

    def video(self, clip: ClipManifest) -> FrameDirVideoSource:
        path = Path(clip.video_path)
        directory = path if path.is_absolute() else self._data_dir / path
        return FrameDirVideoSource(directory, clip.real_fps, self._max_frame_bytes)

    def ground_truth(self, clip: ClipManifest) -> ArrayGroundTruthStore:
        if not (self._store.clip_dir(clip.clip_id) / GT_META).is_file():
            raise MissingInputError(
                f"{clip.clip_id}: ground truth not exported (`synth export-gt`)"
            )
        return ArrayGroundTruthStore(load_native(self._store, clip.clip_id, self._far_z_m))

    def fetch(self, subset: str | None, dest: Path) -> None:
        raise UsageError(
            f"{self.name} is generated locally, not downloaded: "
            f"physverify synth build --subset {subset or 'local'}"
        )


PLUGINS = (KubricSynthetic,)
