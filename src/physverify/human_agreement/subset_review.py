"""The review of a VideoPhy subset and the draft queries of its captions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physverify.core.atomic import write_text_atomic
from physverify.core.errors import ConfigError
from physverify.core.output import Echo
from physverify.datasets._queries import derive_queries
from physverify.datasets._videophy_review import (
    REVIEWED,
    SubsetReview,
    load_review,
    record,
    require_recorded,
    template,
    write_review,
)
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is invalid: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return loaded


def _write_json(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_draft_queries(
    dataset: VideophyDataset,
    settings: Settings,
    captions: list[str],
    echo: Echo,
) -> None:
    path = dataset.queries_path
    drafts = draft_queries(_read_json(path), captions, dataset.settings.filter.rigid_event)
    _write_json(path, drafts)
    labels = sized_labels_or_none(settings)
    if labels is None:
        echo("queries: sizing model not available, size coverage not checked")
        return
    missing = uncovered(drafts, labels)
    echo(f"queries: {path} ({len(drafts)} captions, {len(missing)} phrases without size)")
    for phrase in missing:
        echo(f"  no size: {phrase}")


def record_subsets(dataset: VideophyDataset, chosen: str, echo: Echo) -> None:
    review = load_review(dataset.subset_path)
    if review is None:
        raise ConfigError(f"{dataset.subset_path} is missing; run `agreement subset --audit`")
    recorded = record(review, reviewed_candidates(dataset, chosen))
    write_review(dataset.subset_path, recorded)
    for part, stored in recorded.subsets.items():
        echo(f"{part}: n={stored.n} sha256={stored.sha256}")


def check_recorded(dataset: VideophyDataset, echo: Echo) -> None:
    require_recorded(load_review(dataset.subset_path), dataset.settings.filter_version)
    echo(f"review: {dataset.subset_path} ({REVIEWED})")


def review_subset(
    dataset: VideophyDataset,
    settings: Settings,
    chosen: str,
    *,
    show: bool,
    echo: Echo,
) -> None:
    """Write the pending review and the draft queries of its captions, then list the review."""
    review = audit(dataset, chosen)
    write_review(dataset.subset_path, review)
    captions = [c for table in review.decisions.values() for c in table]
    write_draft_queries(dataset, settings, captions, echo)
    echo(f"review: {dataset.subset_path} ({review.status})")
    for part, table in review.decisions.items():
        echo(f"{part}: {len(table)} captions")
        if show:
            for caption, decision in table.items():
                echo(f"  [{decision or '?'}] {caption}")
