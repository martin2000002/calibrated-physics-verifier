"""Replica ids and the replica dataset view used by the extraction target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from physverify.core.errors import UsageError
from physverify.schema.clip import ClipManifest
from physverify.timing.replica import (
    SEPARATOR,
    TIMED,
    WARM,
    ReplicaDataset,
    replica,
    replica_id,
    replica_source,
    replicas,
)

ClipFactory = Callable[..., ClipManifest]


class RecordingDataset:
    name: ClassVar[str] = "inner_ds"

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        raise AssertionError("the view lists its own clips")

    def video(self, clip: ClipManifest) -> Any:
        self.asked.append(("video", clip.clip_id))
        return clip.clip_id

    def ground_truth(self, clip: ClipManifest) -> Any:
        self.asked.append(("ground_truth", clip.clip_id))
        return clip.clip_id

    def fetch(self, subset: str | None, dest: Path) -> None:
        raise AssertionError("never fetched")


def test_replica_changes_only_the_id(clip_factory: ClipFactory) -> None:
    clip = clip_factory("c0")
    copy = replica(clip, WARM)
    assert copy.clip_id == replica_id("c0", WARM) == f"c0{SEPARATOR}{WARM}"
    assert replace(copy, clip_id="c0") == clip


def test_replicas_keep_tag_order(clip_factory: ClipFactory) -> None:
    copies = replicas(clip_factory("c0"), (WARM, TIMED))
    assert list(copies) == [replica_id("c0", WARM), replica_id("c0", TIMED)]
    assert [copy.clip_id for copy in copies.values()] == list(copies)


def test_view_serves_every_replica_from_its_original(clip_factory: ClipFactory) -> None:
    clip = clip_factory("c0")
    copies = replicas(clip, (WARM, TIMED))
    inner = RecordingDataset()
    view = ReplicaDataset(inner, dict.fromkeys(copies, clip))
    assert view.name == inner.name
    assert list(view.clips()) == [clip]
    for copy in copies.values():
        view.video(copy)
    view.ground_truth(copies[replica_id("c0", TIMED)])
    assert inner.asked == [("video", "c0"), ("video", "c0"), ("ground_truth", "c0")]


def test_view_passes_unknown_clips_through(clip_factory: ClipFactory) -> None:
    inner = RecordingDataset()
    ReplicaDataset(inner, {}).video(clip_factory("other"))
    assert inner.asked == [("video", "other")]


def test_view_cannot_be_fetched(clip_factory: ClipFactory, tmp_path: Path) -> None:
    view = replica_source(RecordingDataset(), {})
    with pytest.raises(UsageError, match="cannot be fetched"):
        view.fetch(None, tmp_path)
