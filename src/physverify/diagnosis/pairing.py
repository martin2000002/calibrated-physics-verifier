"""Pairing of check results across input levels (Levels 2 and 3).

A reference result (the oracle side) and another result describe the same event when they share
the clip, the check, the subject (perceived ids mapped to ground-truth ids) and overlapping native
frame spans. Within each (clip, check, subject) group the events are assigned one to one by
maximum span IoU; what stays alone is missing (reference) or spurious (other).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.core.errors import PhysverifyError
from physverify.extraction.matching import hungarian, match_masks
from physverify.schema.checks import CheckResult
from physverify.schema.matching import IdMap
from physverify.schema.perception import MaskTrack

Span = tuple[int, int]
Subject = tuple[str, ...]
GroupKey = tuple[str, str, Subject]
SubjectMapper = Callable[[Subject], Subject | None]

NO_SPAN: Span = (-1, -1)
"""Sort key of an unlocalized result (``frame_span`` None)."""


class UnitMismatchError(PhysverifyError):
    """Results that are compared (or LIFT outputs across cells) are in different units."""


def common_unit(label: str, units: Mapping[str, Iterable[str]]) -> str:
    """The one unit shared by every source; raises ``UnitMismatchError`` otherwise."""
    seen = {source: sorted(set(values)) for source, values in units.items()}
    distinct = {unit for values in seen.values() for unit in values}
    if len(distinct) != 1:
        detail = "; ".join(f"{source}: {values}" for source, values in sorted(seen.items()))
        raise UnitMismatchError(f"{label}: expected a single unit, got {detail or 'none'}")
    return distinct.pop()


@dataclass(frozen=True)
class ResultPair:
    """``oracle`` is the reference side; ``subject`` is its (ground-truth) subject."""

    check_id: str
    clip_id: str
    subject: Subject
    oracle: CheckResult
    other: CheckResult

    @property
    def delta(self) -> float | None:
        """m_other - m_oracle, when both sides have a magnitude."""
        if self.oracle.magnitude is None or self.other.magnitude is None:
            return None
        return self.other.magnitude - self.oracle.magnitude


@dataclass(frozen=True)
class Paired:
    pairs: tuple[ResultPair, ...] = ()
    missing: tuple[CheckResult, ...] = ()
    spurious: tuple[CheckResult, ...] = ()

    @property
    def check_ids(self) -> tuple[str, ...]:
        ids = {p.check_id for p in self.pairs}
        ids |= {r.check_id for r in (*self.missing, *self.spurious)}
        return tuple(sorted(ids))

    def of_check(self, check_id: str) -> Paired:
        return Paired(
            pairs=tuple(p for p in self.pairs if p.check_id == check_id),
            missing=tuple(r for r in self.missing if r.check_id == check_id),
            spurious=tuple(r for r in self.spurious if r.check_id == check_id),
        )


def merge(parts: Iterable[Paired]) -> Paired:
    """Concatenation of several clips' pairings, in the given order."""
    items = tuple(parts)
    return Paired(
        pairs=tuple(p for part in items for p in part.pairs),
        missing=tuple(r for part in items for r in part.missing),
        spurious=tuple(r for part in items for r in part.spurious),
    )


def subject_map(pred_track: MaskTrack, truth_track: MaskTrack) -> IdMap:
    """Perceived track ids to ground-truth ids by mask IoU (``match_masks``)."""
    return match_masks(pred_track, truth_track)


def span_iou(a: Span | None, b: Span | None, unlocalized: float) -> float:
    """IoU of two inclusive native frame spans.

    An unlocalized result (None) is compatible with any event of its group: it scores 1 against
    another unlocalized one and ``unlocalized`` (the acceptance floor) against a span, so a real
    overlap is never displaced by it.
    """
    if a is None or b is None:
        return 1.0 if a is None and b is None else unlocalized
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def _mapper(subjects: IdMap | None) -> SubjectMapper:
    if subjects is None:
        return lambda subject: subject
    to_truth = subjects.pred_to_gt()
    unmatched = frozenset(subjects.unmatched_pred)

    def mapped(subject: Subject) -> Subject | None:
        if any(item in unmatched for item in subject):
            return None
        return tuple(to_truth.get(item, item) for item in subject)

    return mapped


def _key(result: CheckResult, subject: Subject) -> GroupKey:
    """Subjects are compared as sets, so a pair (i, j) matches (j, i)."""
    return (result.clip_id, result.check_id, tuple(sorted(subject)))


def _order(result: CheckResult) -> tuple[str, str, Subject, Span]:
    return (result.clip_id, result.check_id, result.subject, result.frame_span or NO_SPAN)


def _assign(
    refs: Sequence[CheckResult], others: Sequence[CheckResult], min_span_iou: float
) -> tuple[tuple[int, int], ...]:
    scores = np.array(
        [[span_iou(r.frame_span, o.frame_span, min_span_iou) for o in others] for r in refs],
        dtype=np.float64,
    ).reshape(len(refs), len(others))
    return hungarian(scores, min_span_iou)


def pair_results(
    oracle: Sequence[CheckResult],
    other: Sequence[CheckResult],
    subjects: IdMap | None,
    min_span_iou: float,
) -> Paired:
    """Pair ``other`` with the reference ``oracle`` results.

    ``subjects`` maps the other side's ids to the reference ids (None: same ids). An id the map
    left unmatched makes that result spurious; ids the map does not know (e.g. the environment)
    are kept as they are.
    """
    mapper = _mapper(subjects)
    groups: dict[GroupKey, tuple[list[CheckResult], list[CheckResult]]] = {}
    spurious: list[CheckResult] = []
    for result in sorted(oracle, key=_order):
        groups.setdefault(_key(result, result.subject), ([], []))[0].append(result)
    for result in sorted(other, key=_order):
        subject = mapper(result.subject)
        if subject is None:
            spurious.append(result)
            continue
        groups.setdefault(_key(result, subject), ([], []))[1].append(result)
    pairs: list[ResultPair] = []
    missing: list[CheckResult] = []
    for key in sorted(groups):
        refs, others = groups[key]
        assigned = _assign(refs, others, min_span_iou)
        pairs += [
            ResultPair(key[1], key[0], refs[i].subject, refs[i], others[j]) for i, j in assigned
        ]
        missing += [r for i, r in enumerate(refs) if i not in {a for a, _ in assigned}]
        spurious += [o for j, o in enumerate(others) if j not in {b for _, b in assigned}]
    pairs.sort(key=lambda pair: _order(pair.oracle))
    return Paired(
        tuple(pairs), tuple(sorted(missing, key=_order)), tuple(sorted(spurious, key=_order))
    )
