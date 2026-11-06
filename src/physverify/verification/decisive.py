"""The subject whose magnitude a multi-subject check reports, shared by the run table and the
calibrated clip outcome. Only the ordering and the subjects a set is made of live here; the
verdict of a set is ``runner.row_verdict``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from physverify.schema.checks import AbstainReason, Verdict


class Judged(Protocol):
    @property
    def verdict(self) -> Verdict: ...

    @property
    def magnitude(self) -> float | None: ...

    @property
    def abstain_reason(self) -> AbstainReason | None: ...


def size(magnitude: float | None) -> float:
    """The ordering value of a magnitude: None and NaN rank below every number."""
    if magnitude is None or math.isnan(magnitude):
        return -math.inf
    return magnitude


def applicable[T: Judged](items: Sequence[T]) -> list[T]:
    """The subjects a set is made of: one outside its check's declared scope (NOT_APPLICABLE)
    neither votes nor abstains."""
    return [item for item in items if item.abstain_reason is not AbstainReason.NOT_APPLICABLE]


def decisive[T: Judged](items: Sequence[T]) -> T | None:
    """The largest ``applicable`` FAIL; else the largest non-ABSTAIN subject; else the largest
    one. A NaN magnitude ranks above None, so a set measured only as NaN reports NaN, not None."""
    return max(_pool(applicable(items)), key=_rank, default=None)


def _pool[T: Judged](items: Sequence[T]) -> Sequence[T]:
    failed = [item for item in items if item.verdict is Verdict.FAIL]
    if failed:
        return failed
    return [item for item in items if item.verdict is not Verdict.ABSTAIN] or items


def _rank(item: Judged) -> tuple[float, bool]:
    return size(item.magnitude), item.magnitude is not None
