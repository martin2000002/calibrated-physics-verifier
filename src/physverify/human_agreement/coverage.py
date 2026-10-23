"""Rows of a subset the chain cannot see: no object query, or no video in the source (an explicit
abstention). Read from the rows, the queries and the files on disk only; labels never enter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from physverify.datasets._videophy_tables import VideophyRow

if TYPE_CHECKING:
    from physverify.datasets.videophy import VideophyDataset

NO_QUERY = "no_query"
NO_VIDEO = "no_video"

type RowTest = Callable[[VideophyRow], bool]


@dataclass(frozen=True)
class InputGaps:
    """``no_video`` counts every row without its video, with a query or not; ``by_part`` and
    ``by_generator`` hold the same two counts per part and per ``version:generator``."""

    no_query: int
    no_query_with_video: int
    no_video: int
    by_part: dict[str, dict[str, int]] = field(default_factory=dict)
    by_generator: dict[str, dict[str, int]] = field(default_factory=dict)
    no_query_captions: tuple[str, ...] = ()


def generator_key(row: VideophyRow) -> str:
    return f"{row.version}:{row.generator}"


def _tally(groups: dict[str, dict[str, int]], key: str, name: str) -> None:
    counts = groups.setdefault(key, {NO_QUERY: 0, NO_VIDEO: 0})
    counts[name] += 1


def input_gaps(rows: Sequence[VideophyRow], has_query: RowTest, has_video: RowTest) -> InputGaps:
    by_part: dict[str, dict[str, int]] = {}
    by_generator: dict[str, dict[str, int]] = {}
    no_query = with_video = no_video = 0
    captions: set[str] = set()
    for row in rows:
        query, video = has_query(row), has_video(row)
        for name, gap in ((NO_QUERY, not query), (NO_VIDEO, not video)):
            if gap:
                _tally(by_part, row.part, name)
                _tally(by_generator, generator_key(row), name)
        if not query:
            no_query += 1
            with_video += video
            captions.add(row.caption)
        no_video += not video
    return InputGaps(no_query, with_video, no_video, by_part, by_generator, tuple(sorted(captions)))


def dataset_tests(dataset: VideophyDataset) -> tuple[RowTest, RowTest]:
    """``(has_query, has_video)`` as extraction sees them: the queries the clip would be
    extracted with, and its video file on disk."""
    queries = dataset.queries()
    return (
        lambda row: bool(dataset.object_queries(row, queries)),
        lambda row: dataset.video_file(row).is_file(),
    )


def dataset_gaps(dataset: VideophyDataset, rows: Sequence[VideophyRow]) -> InputGaps:
    return input_gaps(rows, *dataset_tests(dataset))
