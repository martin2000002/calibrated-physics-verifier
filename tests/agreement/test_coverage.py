"""Input gaps: rows without an object query or a video, counted apart from the abstentions."""

from __future__ import annotations

import json
import math

from physverify.agreement.coverage import InputGaps, dataset_gaps, input_gaps
from physverify.datasets._videophy_tables import VideophyRow
from physverify.schema.clip import Split

QUERYLESS = "A spinning top made of wood."


def _row(clip_id: str, part: str, generator: str, caption: str = "A ball rolls.") -> VideophyRow:
    version = part.split("_")[0]
    return VideophyRow(
        part=part,
        version=version,
        split=Split.EVAL,
        clip_id=clip_id,
        video_url=f"https://videos.example.org/{clip_id}.mp4",
        caption=caption,
        generator=generator,
        human_raw=1.0,
        human_cut=1.0,
        reference_raw=math.nan,
        reference_cut=0.5,
        reference_name="",
        violated_rules="",
    )


ROWS = (
    _row("a", "v1_test", "lavie"),
    _row("b", "v1_test", "lavie"),
    _row("c", "v2_test", "ray2", QUERYLESS),
    _row("d", "v2_test", "ray2", QUERYLESS),
    _row("e", "v2_test", "ray2"),
    _row("f", "v2_test", "wan"),
)
NO_VIDEO = {"d", "e"}


def test_six_rows_with_each_gap_are_counted_per_part_and_generator() -> None:
    gaps = input_gaps(ROWS, lambda r: r.caption != QUERYLESS, lambda r: r.clip_id not in NO_VIDEO)
    assert (gaps.no_query, gaps.no_query_with_video, gaps.no_video) == (2, 1, 2)
    assert gaps.by_generator == {"v2:ray2": {"no_query": 2, "no_video": 2}}
    assert gaps.by_part == {"v2_test": {"no_query": 2, "no_video": 2}}
    assert gaps.no_query_captions == (QUERYLESS,)


def test_no_gap_gives_zero_counts() -> None:
    assert input_gaps(ROWS, lambda r: True, lambda r: True) == InputGaps(0, 0, 0)


def test_the_dataset_reads_the_queries_file_and_the_videos_on_disk(world) -> None:
    """An empty entry falls back to the caption heuristic: 'ball' for the ball, nothing before
    'spinning' for the top."""
    world.keep_all()
    box, ball = (
        next(r for r in world.dataset.rows("h2_test") if r.caption == c)
        for c in (world.BOX, world.BALL)
    )
    path = world.dataset.video_file(box)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    queries = world.dataset.queries_path
    queries.parent.mkdir(parents=True, exist_ok=True)
    queries.write_text(json.dumps({world.BALL: [], QUERYLESS: []}), encoding="utf-8")
    top = _row("t", "v2_test", "ray2", QUERYLESS)
    gaps = dataset_gaps(world.dataset, [box, ball, top])
    assert (gaps.no_query, gaps.no_query_with_video, gaps.no_video) == (1, 0, 2)
    assert gaps.no_query_captions == (QUERYLESS,)
