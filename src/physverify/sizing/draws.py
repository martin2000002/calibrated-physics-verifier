"""The seeded Objectron draws of A9 (ADR 0010): the per-class draw of each official split and the
end-to-end candidates. ``physverify size`` makes them; the ``objectron_static`` control replays
them to leave every video A9 looked at out of its own selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from physverify.config.models import Settings
from physverify.sizing import objectron
from physverify.sizing.settings import EndToEndSettings, SizingSettings

DATASET_DIR = "objectron"
TRAIN, TEST = objectron.SPLITS


def objectron_dir(settings: Settings) -> Path:
    return settings.paths.raw_dir / DATASET_DIR


def official_split(
    settings: Settings, sizing: SizingSettings, split: str, reader: objectron.PrefixReader
) -> dict[str, list[objectron.VideoRef]]:
    """The whole official index of ``split`` per class (cached on disk)."""
    index_dir = objectron_dir(settings) / "index"
    return {
        label: objectron.read_index(reader, sizing.bucket_url, label, split, index_dir)
        for label in sizing.classes
    }


def selection(
    settings: Settings, sizing: SizingSettings, split: str, reader: objectron.PrefixReader
) -> list[objectron.VideoRef]:
    """The seeded per-class draw of the official ``split``, refused if the official train and
    test indices share a video."""
    indices = {name: official_split(settings, sizing, name, reader) for name in (TRAIN, TEST)}
    objectron.require_disjoint(
        (r for refs in indices[TRAIN].values() for r in refs),
        (r for refs in indices[TEST].values() for r in refs),
    )
    count = sizing.per_class_train if split == TRAIN else sizing.per_class_test
    return [
        ref
        for label, refs in indices[split].items()
        for ref in objectron.select(refs, label, count, sizing.seed)
    ]


@dataclass(frozen=True)
class EndToEndDraw:
    taken: tuple[objectron.VideoRef, ...]
    skipped: tuple[objectron.VideoRef, ...]

    @property
    def examined(self) -> frozenset[str]:
        return frozenset(r.video_id for r in (*self.taken, *self.skipped))


def e2e_draw(
    settings: Settings,
    sizing: SizingSettings,
    plan: EndToEndSettings,
    reader: objectron.PrefixReader,
    fits: Callable[[objectron.VideoRef], bool],
) -> EndToEndDraw:
    """The end-to-end candidates in draw order: per class, the seeded permutation of the test
    index, taking a video when ``fits`` (``size e2e-prepare`` prepares it there) until the
    class quota or the total is met; the rest examined are skipped."""
    taken: list[objectron.VideoRef] = []
    skipped: list[objectron.VideoRef] = []
    quota = math.ceil(plan.videos / max(1, len(plan.classes)))
    for label in plan.classes:
        index = objectron.read_index(
            reader, sizing.bucket_url, label, TEST, objectron_dir(settings) / "index"
        )
        count = 0
        for ref in objectron.select(index, label, len(index), sizing.seed):
            if count == quota or len(taken) == plan.videos:
                break
            if fits(ref):
                taken.append(ref)
                count += 1
            else:
                skipped.append(ref)
    return EndToEndDraw(tuple(taken), tuple(skipped))
