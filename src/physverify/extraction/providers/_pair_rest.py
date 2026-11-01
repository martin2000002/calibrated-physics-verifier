"""Rest of object pairs for ``contacts/kinematic_scene``; not a plugin.

A pair rest test gives a clip's RESTING pair records and marks the rows where a body rests on
another, so that those rows are not tested against the environment. ``proximity``: the rows of a
proximity interval of the pair (``contacts/proximity`` geometry without a ground plane) where
both objects rest; the upper one is the one with the larger up . median position. A new test is a
module with a class that meets ``PairRest``, a builder from the stage context and an entry in
``PAIR_RESTS``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._impact_attribution import KinematicScene
from physverify.extraction.providers._proximity import FrameContact, nan_median
from physverify.extraction.providers._scene_records import (
    SOURCE,
    Interval,
    PairRest,
    SceneInputs,
)
from physverify.schema.contacts import ContactKind, ContactRecord
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PairRestName = Literal["proximity"]


class PairRestBuilder(Protocol):
    def __call__(self, ctx: StageContext, states: ObjectStateTable, /) -> PairRest: ...


@dataclass(frozen=True)
class ProximityPairs:
    """Both objects of a proximity interval at rest."""

    def pairs(
        self, inputs: SceneInputs, up: FloatArray | None, covered: BoolArray
    ) -> list[ContactRecord]:
        return list(resting_pairs(inputs.scene, inputs.intervals, inputs.rest, up, covered))


def proximity_pairs(_ctx: StageContext, _states: ObjectStateTable) -> ProximityPairs:
    return ProximityPairs()


def resting_pairs(
    scene: KinematicScene,
    intervals: Sequence[Interval],
    rest: BoolArray,
    up: FloatArray | None,
    covered: BoolArray,
) -> Iterator[ContactRecord]:
    """RESTING pair records over the stretches where both objects rest; marks in ``covered``
    the rows where the upper object (larger up . median position) rests on the pair."""
    min_rows = scene.proximity.min_rest_rows
    for pair, interval in intervals:
        assert pair.b is not None
        both = [c for c in interval if rest[c.frame, pair.a] and rest[c.frame, pair.b]]
        for stretch in consecutive(both):
            if len(stretch) < min_rows:
                continue
            rows = [c.frame for c in stretch]
            if up is not None:
                covered[rows, upper(scene, pair.a, pair.b, rows, up)] = True
            record = scene.proximity.record(pair, stretch)
            yield replace(record, kind=ContactKind.RESTING, source=SOURCE)


def consecutive(contacts: Sequence[FrameContact]) -> Iterator[list[FrameContact]]:
    current: list[FrameContact] = []
    for contact in contacts:
        if current and contact.frame != current[-1].frame + 1:
            yield current
            current = []
        current.append(contact)
    if current:
        yield current


def upper(scene: KinematicScene, a: int, b: int, rows: Sequence[int], up: FloatArray) -> int:
    heights = [nan_median((scene.states.position[rows, n] @ up).tolist()) for n in (a, b)]
    return b if heights[1] > heights[0] else a


PAIR_RESTS: Mapping[PairRestName, PairRestBuilder] = MappingProxyType(
    {"proximity": proximity_pairs}
)
