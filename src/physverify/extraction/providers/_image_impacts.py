"""Impacts in the image plane for ``contacts/kinematic_scene``; not a plugin.

The candidates are the kinks of the mask-centroid pixel tracks (``_image_kinks.py``). Each kink
goes, as in ``_impact_attribution.py`` but in the image, to the candidate whose image normal
best aligns with its change dv, with a positive cosine: (a) an opposite kink of another object
within one row step whose image boxes touch, normal along dv_a - dv_b; (b) another object with a
proximity record around the kink, its normal projected at the object; (c) the environment, up
projected. Pairs through (a) are accepted first and consume both objects' kinks.
An environment kink is an impact when du = dv . up_img > z(alpha / m) sigma_du and it bounces
(moving down before, not moving down after beyond z(alpha) sigma), on a scene plane P_n made of
the other objects' rest runs and bounces. A pair kink is an impact when its du along the image
normal is significant and its 3D normal is known: the record's for (b), the direction between the
two centres for (a). Records are placed in 3D as the velocity test places them. With an unknown
centroid error no kink is tested and none is found.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from physverify.extraction._base import StageContext
from physverify.extraction.providers._image_kinks import ImageKink, clip_kinks
from physverify.extraction.providers._image_motion import (
    ImageProjection,
    ObjectImage,
    PixelError,
    object_images,
    pixel_error,
)
from physverify.extraction.providers._impact_attribution import (
    PAIR_MARGIN,
    Choice,
    Event,
    KinematicScene,
    boxes_touch,
    cosine,
    nearest_row,
    record_of,
    unit,
)
from physverify.extraction.providers._impacts import Impact, ImpactRows
from physverify.extraction.providers._scene_plane import SupportPlane, height
from physverify.extraction.providers._scene_records import (
    SOURCE,
    FoundImpacts,
    SceneEvidence,
    SceneInputs,
    env_impact,
    planes_from,
    rest_heights,
)
from physverify.extraction.providers._velocity_tests import NormalChange
from physverify.physics.tracks import contact_window
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactRecord
from physverify.schema.perception import CameraModel, MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImageChoice:
    """The surface a kink is attributed to: ``other`` (None = environment) with its image
    normal and, when known before placing it, its 3D normal; ``partner`` is the other object's
    opposite kink of a candidate (a)."""

    kink: ImageKink
    other: int | None
    image_normal: FloatArray
    cosine: float
    normal: FloatArray | None = None
    partner: ImageKink | None = None


@dataclass(frozen=True)
class ImageScene:
    """The kinematic scene with the image projection and up (None without gravity)."""

    scene: KinematicScene
    projection: ImageProjection
    up: FloatArray | None

    def image_up(self, kink: ImageKink) -> FloatArray | None:
        if self.up is None:
            return None
        return self.projection.direction(kink.index, kink.row, self.up)


def impulse_candidates(
    view: ImageScene, kink: ImageKink, left: Sequence[ImageKink]
) -> Iterator[ImageChoice]:
    """(a): opposite kinks of other objects within one row step whose image boxes touch."""
    scene = view.scene
    for other in left:
        if other.index == kink.index or abs(other.time - kink.time) > scene.row_seconds:
            continue
        if not float(kink.delta_v @ other.delta_v) < 0.0:
            continue
        normal = unit(kink.delta_v - other.delta_v)
        if normal is not None and boxes_touch(scene, kink.index, other.index, kink.time):
            yield ImageChoice(
                kink, other.index, normal, cosine(kink.delta_v, normal), partner=other
            )


def proximity_candidates(view: ImageScene, kink: ImageKink) -> Iterator[ImageChoice]:
    """(b): objects with a proximity record whose window (margin 1) holds the kink."""
    scene = view.scene
    ids = scene.states.object_ids
    own = str(ids[kink.index])
    frame = kink.time / scene.seconds_per_frame
    for record in scene.records:
        if own not in (record.object_a, record.object_b) or ENV_ID in (
            record.object_a,
            record.object_b,
        ):
            continue
        first, last = contact_window(record, scene.step, PAIR_MARGIN)
        if first <= frame <= last:
            other = record.object_b if record.object_a == own else record.object_a
            normal = record.normal if record.object_a == own else -record.normal
            image = view.projection.direction(kink.index, kink.row, normal)
            if image is not None:
                index = int(np.flatnonzero(ids == other)[0])
                yield ImageChoice(kink, index, image, cosine(kink.delta_v, image), normal)


def environment_candidate(view: ImageScene, kink: ImageKink) -> Iterator[ImageChoice]:
    """(c): the environment, up projected, when gravity is known."""
    image = view.image_up(kink)
    if image is not None and view.up is not None:
        yield ImageChoice(kink, None, image, cosine(kink.delta_v, image), view.up)


def choose(view: ImageScene, kink: ImageKink, left: Sequence[ImageKink]) -> ImageChoice | None:
    candidates = [
        *impulse_candidates(view, kink, left),
        *proximity_candidates(view, kink),
        *environment_candidate(view, kink),
    ]
    usable = [c for c in candidates if np.isfinite(c.cosine) and c.cosine > 0.0]
    return max(usable, key=lambda c: c.cosine) if usable else None


def resolve(view: ImageScene, kinks: Sequence[ImageKink]) -> list[ImageChoice]:
    """Final choices, as ``_impact_attribution.resolve``: the pair through (a) with the largest
    cosine consumes both objects' kinks within one row step, and the others choose again."""
    consumed: set[tuple[int, int]] = set()
    final: list[ImageChoice] = []
    while True:
        left = [k for k in kinks if k.key not in consumed]
        choices = [c for k in left if (c := choose(view, k, left)) is not None]
        pairs = [c for c in choices if c.partner is not None]
        if not pairs:
            return final + choices
        best = max(pairs, key=lambda c: c.cosine)
        final.append(best)
        objects = {best.kink.index, best.other}
        consumed |= {
            k.key
            for k in kinks
            if k.index in objects and abs(k.time - best.kink.time) <= view.scene.row_seconds
        }


def bounces(kink: ImageKink, up_image: FloatArray, alpha: float) -> bool:
    """Significant upward change, moving down before and not down after beyond z(alpha)."""
    change = NormalChange(float(kink.delta_v @ up_image), kink.sigma, kink.tests)
    descending = float(kink.before @ up_image) < 0.0
    lifted = float(kink.after @ up_image) >= -float(norm.isf(alpha)) * kink.sigma_after
    return change.significant(alpha) and descending and lifted


def event_of(scene: KinematicScene, kink: ImageKink) -> Event:
    """The kink as a 3D event at the object's position on the nearest row; its 3D change is
    unknown (zero), only its time and place are used."""
    position = np.asarray(scene.states.position[nearest_row(scene, kink.time), kink.index])
    return Event(kink.index, Impact(kink.row, kink.time, np.zeros(3), position, kink.score))


def bounce_planes(
    inputs: SceneInputs, view: ImageScene, evidence: SceneEvidence, kinks: Sequence[ImageKink]
) -> tuple[SupportPlane | None, ...]:
    """P_n from the rest runs and the bounces of the other objects: the height of a bounce is
    its body's lowest point at the kink. These planes only test environment kinks."""
    up = view.up
    if up is None:
        return (None,) * inputs.scene.states.n_objects
    heights = rest_heights(inputs, up, evidence.env_runs)
    bodies = inputs.scene.proximity.bodies
    for kink in kinks:
        image = view.image_up(kink)
        if image is not None and bounces(kink, image, inputs.alpha):
            event = event_of(inputs.scene, kink)
            body = bodies[nearest_row(inputs.scene, kink.time)][kink.index]
            heights[kink.index].append(height(body, up, event.impact.position))
    return planes_from(inputs, up, heights)


def environment_record(
    inputs: SceneInputs, choice: ImageChoice, planes: Sequence[SupportPlane | None]
) -> ContactRecord | None:
    kink, support = choice.kink, planes[choice.kink.index]
    if support is None or not bounces(kink, choice.image_normal, inputs.alpha):
        return None
    change = NormalChange(float(kink.delta_v @ choice.image_normal), kink.sigma, kink.tests)
    return env_impact(inputs.scene, event_of(inputs.scene, kink), support, change)


def pair_record(inputs: SceneInputs, choice: ImageChoice) -> ContactRecord | None:
    """du along the image normal, of dv_a - dv_b with hypot(sigma_a, sigma_b) and max(m_a, m_b)
    for (a), of dv for (b); placed with the 3D normal."""
    scene, kink, partner = inputs.scene, choice.kink, choice.partner
    assert choice.other is not None
    delta_v, sigma, tests, normal = kink.delta_v, kink.sigma, kink.tests, choice.normal
    if partner is not None:
        delta_v = delta_v - partner.delta_v
        sigma, tests = float(np.hypot(sigma, partner.sigma)), max(tests, partner.tests)
        position = scene.states.position[nearest_row(scene, kink.time)]
        normal = unit(position[kink.index] - position[choice.other])
    change = NormalChange(float(delta_v @ choice.image_normal), sigma, tests)
    if normal is None or not np.all(np.isfinite(normal)) or not change.significant(inputs.alpha):
        return None
    paired = None if partner is None else event_of(scene, partner)
    found = Choice(event_of(scene, kink), choice.other, normal, choice.cosine, paired)
    record = record_of(scene, found)
    return None if record is None else replace(record, confidence=change.confidence, source=SOURCE)


@dataclass(frozen=True)
class ImageImpacts:
    """Kinks of the masks' centroid tracks, attributed and tested in the image."""

    images: tuple[ObjectImage | None, ...]
    projection: ImageProjection
    error: PixelError

    def impacts(self, inputs: SceneInputs, evidence: SceneEvidence) -> FoundImpacts:
        if not self.error.known:
            return FoundImpacts(())
        scene = inputs.scene
        rows = ImpactRows.at(inputs.side_s, scene.row_seconds)
        kinks = clip_kinks(self.images, scene.states.t, rows, self.error, inputs.alpha)
        view = ImageScene(scene, self.projection, evidence.up)
        planes = bounce_planes(inputs, view, evidence, kinks)
        records = [
            r
            for c in resolve(view, kinks)
            if (
                r := environment_record(inputs, c, planes)
                if c.other is None
                else pair_record(inputs, c)
            )
        ]
        return FoundImpacts(tuple(records))


def image_impacts(ctx: StageContext, states: ObjectStateTable) -> ImageImpacts:
    track = ctx.require(Stage.TRACK, MaskTrack)
    camera = ctx.require(Stage.CAMERA, CameraModel)
    return ImageImpacts(
        object_images(track, states), ImageProjection.of(states, camera), pixel_error(ctx)
    )
