"""Greedy attribution of velocity changes to surfaces; not a plugin.

Shared by the 3D impacts (``_impact_attribution.py``) and the image kinks
(``_image_impacts.py``): each item picks the candidate surface with the largest positive cosine,
and pairs through (a) are accepted first, best cosine first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

EventKey = tuple[int, int]


class Attributable(Protocol):
    """An object's velocity change to attribute: 3D impact or image kink."""

    @property
    def key(self) -> EventKey: ...

    @property
    def index(self) -> int: ...

    @property
    def time(self) -> float: ...


class Attribution(Protocol):
    """A candidate surface for an Attributable; ``partner`` is set for a candidate (a)."""

    @property
    def other(self) -> int | None: ...

    @property
    def cosine(self) -> float: ...

    @property
    def partner(self) -> object | None: ...


def best_choice[ChoiceT: Attribution](candidates: Iterable[ChoiceT]) -> ChoiceT | None:
    """The candidate with the largest cosine, if that cosine is positive."""
    usable = [c for c in candidates if math.isfinite(c.cosine) and c.cosine > 0.0]
    return max(usable, key=lambda c: c.cosine) if usable else None


def resolve_greedy[ItemT: Attributable, ChoiceT: Attribution](
    items: Sequence[ItemT],
    choose: Callable[[ItemT, Sequence[ItemT]], ChoiceT | None],
    row_seconds: float,
) -> list[ChoiceT]:
    """Final choices. The pair through (a) with the largest cosine is accepted and consumes both
    objects' items within one row step; the items left choose again among themselves, so an
    item whose best partner was consumed falls back to its next surface (three bodies meeting
    at once). Without pairs left, the remaining items keep their choices."""
    consumed: set[EventKey] = set()
    final: list[ChoiceT] = []
    while True:
        left = [i for i in items if i.key not in consumed]
        choices = [(i, c) for i in left if (c := choose(i, left)) is not None]
        pairs = [(i, c) for i, c in choices if c.partner is not None]
        if not pairs:
            return final + [c for _, c in choices]
        item, best = max(pairs, key=lambda pair: pair[1].cosine)
        final.append(best)
        objects = {item.index, best.other}
        consumed |= {
            i.key for i in items if i.index in objects and abs(i.time - item.time) <= row_seconds
        }
