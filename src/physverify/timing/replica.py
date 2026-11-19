"""Replicas of a clip under new ids, so one pipeline call computes the same clip twice.

The extraction pipeline loads a stage's model on the first uncached cell and frees it at the end
of the stage; running a ``warm`` and a ``timed`` replica of one clip in a single call gives the
steady-state cost (model already loaded) and, by difference, the first-cell overhead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

from physverify.core.errors import UsageError
from physverify.datasets._base import DatasetSource, GroundTruthStore, VideoSource
from physverify.schema.clip import ClipManifest

WARM: Final = "warm"
TIMED: Final = "timed"
SEPARATOR: Final = "__"


def replica_id(clip_id: str, tag: str) -> str:
    return f"{clip_id}{SEPARATOR}{tag}"


def replica(clip: ClipManifest, tag: str) -> ClipManifest:
    return replace(clip, clip_id=replica_id(clip.clip_id, tag))


def replicas(clip: ClipManifest, tags: Sequence[str]) -> dict[str, ClipManifest]:
    """Replica id -> replica, in ``tags`` order."""
    return {copy.clip_id: copy for copy in (replica(clip, tag) for tag in tags)}


class ReplicaDataset:
    """A ``DatasetSource`` view that serves every replica from its original clip; not registered.

    ``originals`` maps a replica id to its original clip; any other clip is passed through."""

    def __init__(self, inner: DatasetSource, originals: Mapping[str, ClipManifest]) -> None:
        self.name = inner.name
        self._inner = inner
        self._originals = dict(originals)

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return list({clip.clip_id: clip for clip in self._originals.values()}.values())

    def video(self, clip: ClipManifest) -> VideoSource:
        return self._inner.video(self._original(clip))

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return self._inner.ground_truth(self._original(clip))

    def fetch(self, subset: str | None, dest: Path) -> None:
        raise UsageError("a replica dataset is a timing view and cannot be fetched")

    def _original(self, clip: ClipManifest) -> ClipManifest:
        return self._originals.get(clip.clip_id, clip)


def replica_source(inner: DatasetSource, originals: Mapping[str, ClipManifest]) -> DatasetSource:
    """The view typed as a ``DatasetSource``: its ``name`` is the inner dataset's, set per
    instance, where the protocol declares a class variable."""
    return cast(DatasetSource, ReplicaDataset(inner, originals))
