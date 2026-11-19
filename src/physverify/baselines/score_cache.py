"""A judge's scores of one dataset's clips in a JSONL file: one line per scored clip, appended
under an exclusive ``flock``. A clip already scored under the judge's current stamp from the same
video bytes is not scored again, so an interrupted run resumes where it stopped."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.core.errors import PhysverifyError
from physverify.core.models import FrozenModel
from physverify.generator_comparison.judges._base import Judge, JudgeInput

LOCK_SUFFIX: Final = ".lock"
SCORES_SUFFIX: Final = ".jsonl"


class CachedScore(FrozenModel):
    clip_id: str
    video_sha256: str
    value: float | None
    scale: tuple[float, float]
    stamp: str
    reason: str | None = None


@dataclass(frozen=True)
class ClipVideo:
    clip_id: str
    video: Path
    sha256: str
    caption: str


def score_file(scores: Path, judge: str, dataset: str) -> Path:
    return scores / judge / f"{dataset}{SCORES_SUFFIX}"


def read_scores(path: Path) -> dict[str, CachedScore]:
    """The last line of every clip; an empty mapping when the file does not exist."""
    if not path.is_file():
        return {}
    found: dict[str, CachedScore] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            score = CachedScore.model_validate_json(line)
        except ValueError as exc:
            raise PhysverifyError(f"{path}:{number}: not a score line: {exc}") from exc
        found[score.clip_id] = score
    return found


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_name(path.name + LOCK_SUFFIX), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class ScoreCache:
    def __init__(self, path: Path, judge: Judge, task: str, per_call: int) -> None:
        self._path, self._judge, self._task, self._per_call = path, judge, task, per_call

    @property
    def path(self) -> Path:
        return self._path

    def fill(self, clips: Sequence[ClipVideo]) -> int:
        """Score the clips without a current line, ``per_call`` at a time; the number of clips
        with a current line afterwards."""
        stamp = self._judge.stamp(self._task)
        with _locked(self._path):
            have = read_scores(self._path)
            todo = [c for c in clips if not _current(have.get(c.clip_id), stamp, c.sha256)]
            for start in range(0, len(todo), self._per_call):
                self._append(todo[start : start + self._per_call])
            have = read_scores(self._path)
        return sum(_current(have.get(c.clip_id), stamp, c.sha256) for c in clips)

    def _append(self, chunk: Sequence[ClipVideo]) -> None:
        inputs = [JudgeInput(c.clip_id, c.video, c.caption) for c in chunk]
        scores = self._judge.judge(inputs, self._task)
        lines = [
            CachedScore(
                clip_id=c.clip_id,
                video_sha256=c.sha256,
                value=scores[c.clip_id].value,
                scale=scores[c.clip_id].scale,
                stamp=scores[c.clip_id].stamp,
                reason=scores[c.clip_id].reason,
            ).model_dump_json()
            for c in chunk
        ]
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write("".join(f"{line}\n" for line in lines))
            handle.flush()
            os.fsync(handle.fileno())


def _current(found: CachedScore | None, stamp: str, sha256: str) -> bool:
    return found is not None and found.stamp == stamp and found.video_sha256 == sha256
