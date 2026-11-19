"""The judge score cache: one JSONL line per clip, resumed by stamp and video bytes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from physverify.baselines.score_cache import ClipVideo, ScoreCache, read_scores, score_file
from physverify.core.errors import PhysverifyError
from physverify.generator_comparison.judges._base import JudgeInput, JudgeScore


class FakeJudge:
    name = "fake"

    def __init__(self, stamp: str = "s1") -> None:
        self.calls: list[tuple[str, ...]] = []
        self._stamp = stamp

    def unavailable(self) -> str | None:
        return None

    def stamp(self, task: str) -> str:
        return f"{self._stamp}:{task}"

    def judge(self, items: Sequence[JudgeInput], task: str) -> Mapping[str, JudgeScore]:
        self.calls.append(tuple(i.video_id for i in items))
        return {
            i.video_id: JudgeScore(None if i.video_id == "bad" else 3.0, (1, 5), self.stamp(task))
            for i in items
        }


def clips(*ids: str, sha: str = "v1") -> list[ClipVideo]:
    return [ClipVideo(i, Path(f"/videos/{i}.mp4"), sha, "a caption") for i in ids]


def test_a_cache_scores_in_chunks_and_resumes(tmp_path: Path) -> None:
    path = score_file(tmp_path, "fake", "set")
    judge = FakeJudge()
    assert ScoreCache(path, judge, "pc", 2).fill(clips("a", "b", "bad")) == 3
    assert judge.calls == [("a", "b"), ("bad",)]
    assert ScoreCache(path, judge, "pc", 2).fill(clips("a", "b", "c")) == 3
    assert judge.calls[-1] == ("c",)
    found = read_scores(path)
    assert found["a"].value == 3.0 and found["bad"].value is None
    assert path == tmp_path / "fake" / "set.jsonl"


def test_another_stamp_or_other_video_bytes_score_again(tmp_path: Path) -> None:
    path = score_file(tmp_path, "fake", "set")
    ScoreCache(path, FakeJudge(), "pc", 10).fill(clips("a", "b"))
    changed = FakeJudge()
    ScoreCache(path, changed, "pc", 10).fill([*clips("a"), *clips("b", sha="v2")])
    assert changed.calls == [("b",)]
    restamped = FakeJudge("s2")
    ScoreCache(path, restamped, "pc", 10).fill(clips("a"))
    assert restamped.calls == [("a",)] and read_scores(path)["a"].stamp == "s2:pc"


def test_a_malformed_line_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(PhysverifyError, match=r"set\.jsonl:1"):
        read_scores(path)
