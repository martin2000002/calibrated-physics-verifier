"""An impact of two spheres whose approach is seen, with no third body near, up to the row the
pair first comes near (route D). Every predicate reads rows up to that onset only.

Near at row k: the image centres are within ``kappa`` times the sum of the apparent radii, or the
closing speed of the last step closes the gap d - S within one more step. The radius at k is
``apparent_radius`` over rows k - 2 ``radius_rows``..k: the check's window, made causal. The
onset k* is the first near row, a stopping time (whether k* = k depends on rows <= k only), so
rows rewritten after the impact cannot move the decision. ``approach_seen``: the
``side_min_rows`` rows ending ``gap_frames`` before k* have centres. ``isolated_before``: no other
body is near either one within ``ScopeLimits.reach_frames`` up to k*. ``matched`` only says the
event is this pair's onset: within the same reach of it."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from physverify.checks.variants._image_impulse import (
    BoolArray,
    Event,
    FloatArray,
    IntArray,
    apparent_radius,
    image_track,
)
from physverify.checks.variants.event_scopes._base import ScopeLimits, SidedEvent, both_spheres
from physverify.physics.tracks import row_step
from physverify.schema.states import ObjectStateTable


class PreContactApproach:
    name: ClassVar[str] = "pre_contact_approach"

    def applies(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> bool:
        return not self.failed(states, event, others, gates)

    def failed(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> tuple[str, ...]:
        e = event.event
        pair = (states.object_index(e.a), states.object_index(e.b))
        tracks = Tracks.of(states, gates)
        onset = tracks.onset(*pair)
        holds = {"spheres": both_spheres(states, e), "onset": onset is not None}
        if onset is not None:
            holds |= _onset_predicates(states, tracks, pair, onset, e.t, gates)
        else:
            holds |= dict.fromkeys(("approach_seen", "isolated_before", "matched"), False)
        return tuple(name for name, ok in holds.items() if not ok)


@dataclass(frozen=True)
class Tracks:
    """Each body's image centres and causal apparent radii, one row per frame."""

    centres: tuple[FloatArray, ...]
    radii: tuple[FloatArray, ...]
    kappa: float

    @classmethod
    def of(cls, states: ObjectStateTable, gates: ScopeLimits) -> Tracks:
        n = range(states.n_objects)
        return cls(
            tuple(image_track(states, i) for i in n),
            tuple(_causal_radius(states, i, gates.radius_rows) for i in n),
            gates.kappa,
        )

    def near(self, i: int, j: int) -> BoolArray:
        d = np.linalg.norm(self.centres[i] - self.centres[j], axis=1)
        s = self.radii[i] + self.radii[j]
        before = np.concatenate(([math.nan], d[:-1]))
        with np.errstate(invalid="ignore"):
            closing = np.isfinite(before) & (before - d >= d - s)
            near: BoolArray = np.isfinite(d) & np.isfinite(s) & ((d <= self.kappa * s) | closing)
        return near

    def onset(self, i: int, j: int) -> int | None:
        """The first row the pair is near; None if it never is."""
        hits = np.flatnonzero(self.near(i, j))
        return int(hits[0]) if hits.size else None


def _causal_radius(states: ObjectStateTable, index: int, half: int) -> FloatArray:
    return np.array(
        [apparent_radius(states, index, k - half, half) for k in range(states.n_frames)]
    )


def _onset_predicates(
    states: ObjectStateTable,
    tracks: Tracks,
    pair: tuple[int, int],
    onset: int,
    t: float,
    gates: ScopeLimits,
) -> dict[str, bool]:
    frames = np.asarray(states.frame_index, dtype=np.int64)
    step = row_step(frames)
    reach = gates.reach_frames(step)
    return {
        "approach_seen": _approach_seen(
            tracks, pair, onset, math.ceil(gates.gap_frames / step), gates
        ),
        "isolated_before": _isolated_before(tracks, pair, frames, onset, reach),
        "matched": abs(t - float(frames[onset])) <= reach,
    }


def _approach_seen(
    tracks: Tracks, pair: tuple[int, int], onset: int, gap_rows: int, gates: ScopeLimits
) -> bool:
    last = onset - gap_rows
    rows = range(last - gates.side_min_rows + 1, last + 1)
    return all(
        row >= 0 and bool(np.all(np.isfinite(tracks.centres[i][row]))) for row in rows for i in pair
    )


def _isolated_before(
    tracks: Tracks, pair: tuple[int, int], frames: IntArray, onset: int, reach: float
) -> bool:
    window = (frames >= frames[onset] - reach) & (frames <= frames[onset])
    third = (c for c in range(len(tracks.centres)) if c not in pair)
    return not any(tracks.near(i, c)[window].any() for c in third for i in pair)


PLUGINS = (PreContactApproach,)
