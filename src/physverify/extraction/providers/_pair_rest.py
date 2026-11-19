"""Rest of object pairs for ``contacts/kinematic_scene``; not a plugin.

A pair rest test gives a clip's RESTING pair records and marks the rows where a body rests on
another, so that those rows are not tested against the environment. ``proximity``: the rows of a
proximity interval of the pair (``contacts/proximity`` geometry without a ground plane) where
both objects rest; the upper one is the one with the larger up . median position. ``image``: the
rows of the pair's proximity intervals where both rest, plus the rows where both rest and one
mask lies on the other in the image (shared columns, vertical gap <= K hypot(sigma_px)) with the
bodies within K hypot(sigma_gap) in 3D, K = z(alpha / 2). An image-stacked row needs the upper
body's lowest point above the lower one's by more than K hypot(sigma_h), so that bodies side by
side are not stacked; the upper body of a stretch is the image-upper one of most of its rows,
else the 3D one, and it is covered only when the stretch is stacked. A new test is a module with
a class that meets ``PairRest``, a builder from the stage context and an entry in ``PAIR_RESTS``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._image_motion import (
    ImageProjection,
    ObjectImage,
    PixelError,
    object_images,
    pixel_error,
)
from physverify.extraction.providers._impact_attribution import KinematicScene
from physverify.extraction.providers._proximity import FrameContact, Pair, nan_median
from physverify.extraction.providers._scene_plane import height
from physverify.extraction.providers._scene_records import (
    SOURCE,
    Interval,
    PairRest,
    SceneInputs,
)
from physverify.extraction.providers._velocity_tests import two_sided_quantile
from physverify.schema.contacts import ContactKind, ContactRecord
from physverify.schema.perception import CameraModel, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
PairRestName = Literal["image", "proximity"]
Key = tuple[int, int]


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


@dataclass(frozen=True)
class PairRow:
    """A row's 3D contact of a pair and its image-upper body (None for a proximity row)."""

    contact: FrameContact
    upper: int | None = None


@dataclass(frozen=True)
class ImagePairs:
    """Both objects at rest in a proximity interval or stacked in the image."""

    images: tuple[ObjectImage | None, ...]
    projection: ImageProjection
    error: PixelError

    def pairs(
        self, inputs: SceneInputs, up: FloatArray | None, covered: BoolArray
    ) -> list[ContactRecord]:
        rows = interval_rows(inputs)
        if up is not None:
            for key, found in self.stack_rows(inputs, up).items():
                mine = rows.setdefault(key, {})
                for row, stack in found.items():
                    assert stack.upper is not None
                    lower = sum(key) - stack.upper
                    if row not in mine and stacked(inputs, up, stack.upper, lower, [row]):
                        mine[row] = stack
        return list(pair_records(inputs, rows, up, covered))

    def stack_rows(self, inputs: SceneInputs, up: FloatArray) -> dict[Key, dict[int, PairRow]]:
        """Rows where both rest and one mask lies on the other, keyed in proximity order."""
        found: dict[Key, dict[int, PairRow]] = {}
        if not (inputs.gap_error.known and self.error.known):
            return found
        k = two_sided_quantile(inputs.alpha)
        for a, b in combinations(np.argsort(inputs.scene.states.object_ids).tolist(), 2):
            pair = stack_pair(inputs, a, b, k)
            for row in np.flatnonzero(inputs.rest[:, a] & inputs.rest[:, b]).tolist():
                upper = self.image_upper(a, b, row, up, k)
                contact = None if upper is None else inputs.scene.proximity.contact(pair, row)
                if contact is not None:
                    found.setdefault((a, b), {})[row] = PairRow(contact, upper)
        return found

    def image_upper(self, a: int, b: int, row: int, up: FloatArray, k: float) -> int | None:
        """The body whose mask lies on the other's within K hypot(sigma_px), or None."""
        image_a, image_b = self.images[a], self.images[b]
        if image_a is None or image_b is None or not (image_a.valid[row] and image_b.valid[row]):
            return None
        up_image = self.projection.direction(a, row, up)
        if up_image is None:
            return None
        rising = float((image_a.centroid[row] - image_b.centroid[row]) @ up_image) > 0.0
        (upper, top), (_, bottom) = (
            ((a, image_a), (b, image_b)) if rising else ((b, image_b), (a, image_a))
        )
        gap = image_gap(top.masks[row], bottom.masks[row])
        sigma = self.error.sigma(np.array([image_a.radius[row], image_b.radius[row]]))
        return upper if gap is not None and gap <= k * math.hypot(*sigma.tolist()) else None


def image_pairs(ctx: StageContext, states: ObjectStateTable) -> ImagePairs:
    track = ctx.require(Stage.TRACK, MaskTrack)
    camera = ctx.require(Stage.CAMERA, CameraModel)
    return ImagePairs(
        object_images(track, states), ImageProjection.of(states, camera), pixel_error(ctx)
    )


def interval_rows(inputs: SceneInputs) -> dict[Key, dict[int, PairRow]]:
    """Rows of every pair's proximity intervals where both objects rest."""
    rows: dict[Key, dict[int, PairRow]] = {}
    for pair, interval in inputs.intervals:
        assert pair.b is not None
        for contact in interval:
            if inputs.rest[contact.frame, pair.a] and inputs.rest[contact.frame, pair.b]:
                rows.setdefault((pair.a, pair.b), {})[contact.frame] = PairRow(contact)
    return rows


def stack_pair(inputs: SceneInputs, a: int, b: int, k: float) -> Pair:
    """The pair geometry with its tolerance widened to K hypot(sigma_gap)."""
    lengths = inputs.scene.proximity.lengths
    la, lb = float(lengths[a]), float(lengths[b])
    spread = k * math.hypot(inputs.gap_error.sigma(la), inputs.gap_error.sigma(lb))
    length = min(la, lb)
    return Pair(a=a, b=b, tol=max(inputs.scene.tol_rel * length, spread), length=length)


def image_gap(upper: BoolArray, lower: BoolArray) -> float | None:
    """Least vertical pixel gap from the upper mask's bottom to the lower mask's top over
    their shared columns (y down); None without one."""
    columns = np.flatnonzero(upper.any(axis=0) & lower.any(axis=0))
    if columns.size == 0:
        return None
    bottom = (upper.shape[0] - 1) - np.argmax(upper[::-1, columns], axis=0)
    top = np.argmax(lower[:, columns], axis=0)
    return float(np.min(top - bottom))


def stacked(
    inputs: SceneInputs, up: FloatArray, upper: int, lower: int, rows: Sequence[int]
) -> bool:
    """Median over the rows of the upper body's lowest point above the lower one's beyond
    K hypot(sigma_h(L_upper), sigma_h(L_lower))."""
    proximity = inputs.scene.proximity
    bodies = [(proximity.bodies[r][upper], proximity.bodies[r][lower]) for r in rows]
    rises = [
        height(hi, up) - height(lo, up) for hi, lo in bodies if hi is not None and lo is not None
    ]
    rises = [r for r in rises if math.isfinite(r)]
    if not rises:
        return False
    sigmas = [inputs.height_error.sigma(float(proximity.lengths[n])) for n in (upper, lower)]
    bound = two_sided_quantile(inputs.alpha) * math.hypot(*sigmas)
    return float(np.median(rises)) > bound


def pair_records(
    inputs: SceneInputs,
    rows: Mapping[Key, Mapping[int, PairRow]],
    up: FloatArray | None,
    covered: BoolArray,
) -> Iterator[ContactRecord]:
    scene = inputs.scene
    proximity = scene.proximity
    for (a, b), found in rows.items():
        length = float(min(proximity.lengths[a], proximity.lengths[b]))
        pair = Pair(a=a, b=b, tol=scene.tol_rel * length, length=length)
        for stretch in consecutive([found[r].contact for r in sorted(found)]):
            if len(stretch) < proximity.min_rest_rows:
                continue
            frames = [c.frame for c in stretch]
            if up is not None:
                upper = stretch_upper(scene, (a, b), [found[r] for r in frames], up)
                if stacked(inputs, up, upper, a + b - upper, frames):
                    covered[frames, upper] = True
            record = proximity.record(pair, stretch)
            yield replace(record, kind=ContactKind.RESTING, source=SOURCE)


def stretch_upper(scene: KinematicScene, key: Key, rows: Sequence[PairRow], up: FloatArray) -> int:
    """The image-upper body of most rows, else the 3D upper one."""
    uppers = [r.upper for r in rows if r.upper is not None]
    if 2 * len(uppers) > len(rows):
        return max(sorted(set(uppers)), key=uppers.count)
    return upper(scene, *key, [r.contact.frame for r in rows], up)


PAIR_RESTS: Mapping[PairRestName, PairRestBuilder] = MappingProxyType(
    {"image": image_pairs, "proximity": proximity_pairs}
)
