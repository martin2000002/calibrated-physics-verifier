"""Dataset contracts: frame access, ground truth per stage, and dataset discovery.

``ArrayGroundTruthStore`` is the one place where native-resolution ground truth is resampled to
a ``FrameSampling`` (nearest neighbour for masks and depth, K scaled by the resize factors).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from physverify.config.plugin_context import PluginKind
from physverify.core.download import BYTES_PER_MB, MANIFEST_FILE, read_manifest
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.output import Echo
from physverify.core.resample import resize_depth, resize_masks, scale_intrinsics
from physverify.schema.clip import ClipManifest, FrameSampling, ObjectQuery, Split
from physverify.schema.contacts import ContactTable
from physverify.schema.perception import (
    CameraModel,
    DepthKind,
    DepthMap,
    Detections,
    IntrinsicsSource,
    MaskTrack,
)
from physverify.schema.scene import SceneSpec
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import CoordFrame, ObjectStateTable

if TYPE_CHECKING:
    from physverify.config.models import Settings


@runtime_checkable
class VideoSource(Protocol):
    """Decoded RGB frames; ``frames`` reduces each frame before keeping it."""

    @property
    def real_fps(self) -> float: ...

    @property
    def n_frames(self) -> int: ...

    @property
    def native_size(self) -> tuple[int, int]: ...

    def frames(
        self, start: int, stop: int, step: int = 1, max_side: int | None = None
    ) -> NDArray[np.uint8]:
        """(T', h, w, 3) uint8 RGB for native indices range(start, stop, step)."""
        ...


@runtime_checkable
class GroundTruthStore(Protocol):
    """Canonical ground truth of one clip; pixel outputs follow the requested sampling."""

    @property
    def version(self) -> str: ...

    def stages(self) -> tuple[Stage, ...]: ...

    def detections(self, sampling: FrameSampling) -> Detections: ...

    def masks(self, sampling: FrameSampling) -> MaskTrack: ...

    def depth(self, sampling: FrameSampling) -> DepthMap:
        """Metric z layer (``DepthKind.METRIC_Z``)."""
        ...

    def camera(self, sampling: FrameSampling) -> CameraModel: ...

    def states(self, frame: CoordFrame) -> ObjectStateTable:
        """Native-rate states in meters with zero sigma."""
        ...

    def contacts(self, frame: CoordFrame) -> ContactTable: ...

    def scene(self) -> SceneSpec | None: ...


@runtime_checkable
class DatasetSource(Protocol):
    """A dataset plugin; constructed as ``cls(settings)`` and registered by ``name``."""

    name: ClassVar[str]

    def __init__(self, settings: Settings) -> None: ...

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]: ...

    def video(self, clip: ClipManifest) -> VideoSource: ...

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None: ...

    def fetch(self, subset: str | None, dest: Path) -> None: ...


DATASETS = PluginKind[type[DatasetSource]]("datasets")


DEFAULT_LABEL_SOURCE: Final = "videophy"


@dataclass(frozen=True)
class LabelledVideo:
    """One generated video with its published labels. ``human_raw`` and ``reference_raw`` keep
    the published values (NaN when the table has none); binarisation is the comparison's job."""

    part: str
    version: str
    split: Split
    clip_id: str
    video_url: str
    caption: str
    generator: str
    human_raw: float
    human_cut: float
    reference_raw: float
    reference_cut: float
    reference_name: str
    violated_rules: str


class HumanLabelSource(DatasetSource, Protocol):
    """A dataset of generated videos with published human labels and a reference model's
    scores, which the human-agreement study and the generator comparison read."""

    config_section: ClassVar[str]
    """The configuration section the source reads; the study's pre-registration hashes it."""
    fetch_hint: ClassVar[str]
    """The command that downloads the label tables and videos."""

    @property
    def queries_path(self) -> Path: ...

    def rows(self, subset: str | None = None) -> list[LabelledVideo]: ...

    def queries(self) -> dict[str, tuple[ObjectQuery, ...]]: ...

    def object_queries(
        self, row: LabelledVideo, queries: Mapping[str, tuple[ObjectQuery, ...]]
    ) -> tuple[ObjectQuery, ...]: ...

    def video_file(self, row: LabelledVideo, root: Path | None = None) -> Path: ...

    def recorded_subsets(self) -> dict[str, str | None]:
        """The recorded sha256 of each hand-reviewed part; empty before the review."""
        ...

    def reference_table(self, part: str) -> Path:
        """The table of the reference model's published scores of ``part``."""
        ...

    def reference_records(self, part: str) -> list[dict[str, str]]: ...

    def review_subset(self, subset: str, *, show: bool, echo: Echo) -> None:
        """Write the pending hand review of ``subset`` and the draft queries of its captions."""
        ...

    def record_review(self, subset: str, *, echo: Echo) -> None: ...

    def check_review(self, *, echo: Echo) -> None: ...


def select_clips(clips: Sequence[ClipManifest], wanted: Sequence[str]) -> list[ClipManifest]:
    """``wanted`` in its first-seen order, or every clip when empty."""
    if not wanted:
        return list(clips)
    by_id = {clip.clip_id: clip for clip in clips}
    unknown = sorted(set(wanted) - set(by_id))
    if unknown:
        raise UsageError(f"unknown clip ids {unknown}")
    return [by_id[clip_id] for clip_id in dict.fromkeys(wanted)]


def dataset_from(
    registry: Mapping[str, type[DatasetSource]], name: str, settings: Settings
) -> DatasetSource:
    """Dataset ``name`` of ``registry`` built from ``settings``."""
    if name not in registry:
        raise UsageError(f"unknown dataset {name!r}; known: {sorted(registry)}")
    return registry[name](settings)


def fetch_summary(dest: Path, elapsed_s: float) -> str:
    manifest = read_manifest(dest / MANIFEST_FILE)
    if manifest is None:
        return f"{dest.name}: no manifest written ({elapsed_s:.1f} s)"
    size_mb = sum(f.size for f in manifest.files) / BYTES_PER_MB
    return (
        f"{manifest.dataset} [{manifest.subset}]: {len(manifest.clips)} clips, "
        f"{len(manifest.files)} files, {size_mb:.1f} MB, {elapsed_s:.1f} s -> {dest}"
    )


def fetch_all(
    classes: Sequence[type[Any]],
    settings: Settings,
    subset: str | None,
    *,
    echo: Echo,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    for cls in classes:
        dest = settings.paths.raw_dir / cls.name
        started = clock()
        cls(settings).fetch(subset, dest)
        echo(fetch_summary(dest, clock() - started))


@dataclass(frozen=True, eq=False)
class NativeGroundTruth:
    """Ground-truth layers at native resolution for the native frames ``frame_indices``.

    ``segmentation`` value k > 0 is ``object_ids[k - 1]``; ``z`` is metric depth along the
    optical axis (0 or non-finite = no surface); ``K`` is OpenCV intrinsics in native pixels.
    """

    version: str
    frame_indices: NDArray[np.int64]
    segmentation: NDArray[np.uint32]
    z: NDArray[np.float32]
    object_ids: NDArray[np.str_]
    labels: NDArray[np.str_]
    K: NDArray[np.float64]
    world_from_camera: NDArray[np.float64]
    states: Mapping[CoordFrame, ObjectStateTable]
    contacts: Mapping[CoordFrame, ContactTable]
    scene: SceneSpec | None = None


class ArrayGroundTruthStore:
    """GroundTruthStore over in-memory native layers."""

    def __init__(self, native: NativeGroundTruth) -> None:
        self._native = native

    @property
    def version(self) -> str:
        return self._native.version

    def stages(self) -> tuple[Stage, ...]:
        return STAGE_ORDER

    def _rows(self, sampling: FrameSampling) -> NDArray[np.int64]:
        native = self._native.frame_indices.tolist()
        wanted = sampling.frame_indices.tolist()
        missing = sorted(set(wanted) - set(native))
        if missing:
            raise MissingInputError(f"ground truth lacks native frames {missing[:5]}")
        return np.array([native.index(i) for i in wanted], dtype=np.int64)

    def _object_masks(self, rows: NDArray[np.int64]) -> NDArray[np.bool_]:
        labels = np.arange(1, self._native.object_ids.shape[0] + 1, dtype=np.uint32)
        seg = self._native.segmentation[rows]
        masks: NDArray[np.bool_] = seg[:, None, :, :] == labels[None, :, None, None]
        return masks

    def detections(self, sampling: FrameSampling) -> Detections:
        masks = self.masks(sampling).masks[0]
        visible = [k for k in range(masks.shape[0]) if bool(masks[k].any())]
        boxes = np.array([_box(masks[k]) for k in visible], dtype=np.float64).reshape(-1, 4)
        return Detections(
            sampling=sampling,
            frame_index=int(sampling.frame_indices[0]),
            boxes_xyxy=boxes,
            scores=np.ones(len(visible)),
            labels=self._native.labels[visible].astype(str),
            object_ids=self._native.object_ids[visible].astype(str),
        )

    def masks(self, sampling: FrameSampling) -> MaskTrack:
        masks = resize_masks(self._object_masks(self._rows(sampling)), sampling.size)
        return MaskTrack(
            sampling=sampling,
            object_ids=self._native.object_ids,
            labels=self._native.labels,
            masks=masks,
            scores=masks.any(axis=(2, 3)).astype(np.float64),
        )

    def depth(self, sampling: FrameSampling) -> DepthMap:
        z = self._native.z[self._rows(sampling)]
        valid = np.isfinite(z) & (z > 0)
        depth, valid = resize_depth(
            np.where(valid, z, 0.0).astype(np.float32), valid, sampling.size
        )
        return DepthMap(sampling=sampling, depth=depth, kind=DepthKind.METRIC_Z, valid=valid)

    def camera(self, sampling: FrameSampling) -> CameraModel:
        K = scale_intrinsics(self._native.K, sampling.native_size, sampling.size)  # noqa: N806
        return CameraModel(
            sampling=sampling,
            K=K,
            source=IntrinsicsSource.GROUND_TRUTH,
            fov_deg=math.degrees(2.0 * math.atan(sampling.width / (2.0 * K[0, 0]))),
            fov_sigma_deg=0.0,
            world_from_camera=self._native.world_from_camera,
        )

    def states(self, frame: CoordFrame) -> ObjectStateTable:
        return self._native.states[frame]

    def contacts(self, frame: CoordFrame) -> ContactTable:
        return self._native.contacts[frame]

    def scene(self) -> SceneSpec | None:
        return self._native.scene


def _box(mask: NDArray[np.bool_]) -> list[float]:
    """Tight xyxy box in continuous pixel coordinates (pixel i spans [i, i + 1])."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]
