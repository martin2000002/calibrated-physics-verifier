"""The review of a VideoPhy subset and the draft queries of its captions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from physverify.datasets._queries import derive_queries
from physverify.datasets._videophy_review import REVIEWED, SubsetReview, load_review, template
from physverify.datasets._videophy_tables import VideophyRow

if TYPE_CHECKING:
    from physverify.config.models import Settings
    from physverify.datasets.videophy import VideophyDataset


def reviewed_candidates(dataset: VideophyDataset, subset: str) -> dict[str, list[VideophyRow]]:
    reviewed = dataset.settings.reviewed_subsets
    return {p: rows for p, rows in dataset.candidates(subset).items() if p in reviewed}


def audit(dataset: VideophyDataset, subset: str) -> SubsetReview:
    """The pending review over the lexical rule's captions (a recorded review is kept as is)."""
    previous = load_review(dataset.subset_path)
    if previous is not None and previous.status == REVIEWED:
        return previous
    captions = {
        p: [r.caption for r in rows] for p, rows in reviewed_candidates(dataset, subset).items()
    }
    return template(captions, dataset.settings.filter_version, previous)


SUBJECT_ENDS = ("is", "are", "was", "were", "gets", "being", "and", "on", "onto", "into", "off")


def draft_queries(
    existing: Mapping[str, object], captions: Sequence[str], rigid_event: Sequence[str]
) -> dict[str, object]:
    """``{caption: [phrase]}``: the words before the first event verb or ``SUBJECT_ENDS`` word,
    a draft for the hand review; entries already written are kept."""
    drafts: dict[str, object] = dict(existing)
    ends = (*rigid_event, *SUBJECT_ENDS)
    for caption in sorted(set(captions)):
        if caption not in drafts:
            drafts[caption] = [q.phrase for q in derive_queries(caption, ends)]
    return dict(sorted(drafts.items()))


def uncovered(queries: Mapping[str, object], labels: frozenset[str]) -> list[str]:
    """Query phrases with no size: neither the phrase nor its head noun (last word) is sized."""
    phrases = {
        p
        for entries in queries.values()
        if isinstance(entries, list)
        for p in entries
        if isinstance(p, str)
    }
    return sorted(p for p in phrases if p not in labels and p.split()[-1] not in labels)


def sized_labels_or_none(settings: Settings) -> frozenset[str] | None:
    """A9's sized labels, or None when A9's model is not installed yet."""
    try:
        from physverify.sizing.model import sized_labels
    except ImportError:
        return None
    return sized_labels(settings)
