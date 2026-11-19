"""The camera motion side stage: its cache key, derived from the TRACK key, and load-or-compute.

``root/<dataset>/<clip_id>/camera_motion/<estimator>-<key[:12]>/{key.txt,output.json}``: no
``meta.json``, so ``StageCache.metas`` never lists it. This module registers no estimator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from physverify.core.atomic import write_text_atomic
from physverify.datasets._base import VideoSource
from physverify.extraction.camera_motion._base import CameraMotionEstimator
from physverify.schema.camera_motion import CameraMotion
from physverify.schema.errors import SchemaError
from physverify.schema.io import SCHEMA_VERSION, dumps, envelope, stable_hash
from physverify.schema.perception import MaskTrack

CAMERA_MOTION_DIR = "camera_motion"
KEY_FILE = "key.txt"
OUTPUT_FILE = "output.json"


def camera_motion_key(track_key: str, estimator: CameraMotionEstimator, dataset: str) -> str:
    """Every routed plan that shares the main plan's TRACK shares its camera motion."""
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": CAMERA_MOTION_DIR,
            "name": estimator.name,
            "version": estimator.version,
            "params": estimator.params.model_dump(mode="json"),
            "track_key": track_key,
            "dataset": dataset,
        }
    )


class CameraMotionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def directory(self, dataset: str, clip_id: str, estimator: str, key: str) -> Path:
        return self.root / dataset / clip_id / CAMERA_MOTION_DIR / f"{estimator}-{key[:12]}"

    def load(self, dataset: str, clip_id: str, estimator: str, key: str) -> CameraMotion | None:
        directory = self.directory(dataset, clip_id, estimator, key)
        try:
            if (directory / KEY_FILE).read_text(encoding="utf-8") != key:
                return None
            return CameraMotion.load_json(directory / OUTPUT_FILE)
        except (OSError, ValueError, KeyError, SchemaError):
            return None

    def save(self, dataset: str, clip_id: str, key: str, motion: CameraMotion) -> None:
        """Atomic per file, key first: a reader sees a complete output or none."""
        directory = self.directory(dataset, clip_id, motion.estimator, key)
        write_text_atomic(directory / KEY_FILE, key)
        write_text_atomic(directory / OUTPUT_FILE, dumps(envelope(motion, motion.to_dict())))


def ensure_camera_motion(
    estimator: CameraMotionEstimator,
    store: CameraMotionStore,
    dataset: str,
    clip_id: str,
    track_key: str,
    inputs: Callable[[], tuple[VideoSource, MaskTrack]],
) -> CameraMotion:
    """The cached motion, else the estimate of ``inputs`` (cached); a failed estimate is
    ``measured=False`` and is not cached."""
    key = camera_motion_key(track_key, estimator, dataset)
    cached = store.load(dataset, clip_id, estimator.name, key)
    if cached is not None:
        return cached
    try:
        motion = estimator.estimate(*inputs())
    except Exception:  # noqa: BLE001 - an unmeasurable clip is never gated, only counted
        return CameraMotion.unmeasured(estimator.name)
    store.save(dataset, clip_id, key, motion)
    return motion


PLUGINS: tuple[type[CameraMotionEstimator], ...] = ()
