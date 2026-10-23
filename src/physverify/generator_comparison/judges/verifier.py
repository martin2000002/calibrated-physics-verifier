"""The verifier as a judge: the stored calibrated score of each candidate, on [0, 1]."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Final

from physverify.finetune.records import CandidateScore
from physverify.generator_comparison.judges._base import JudgeInput, JudgeScore

SCALE: Final = (0.0, 1.0)


class VerifierJudge:
    """Reads scores already computed by ``compare-generators score``; it never runs the verifier
    itself."""

    name: ClassVar[str] = "verifier"

    def __init__(self, scores: Mapping[str, CandidateScore], stamp: str) -> None:
        self._scores = scores
        self._stamp = stamp

    def unavailable(self) -> str | None:
        if self._scores:
            return None
        return "no stored verifier scores; run `physverify compare-generators score`"

    def stamp(self, task: str) -> str:
        return self._stamp

    def judge(self, items: Sequence[JudgeInput], task: str) -> Mapping[str, JudgeScore]:
        return {item.video_id: self._one(item.video_id) for item in items}

    def _one(self, video_id: str) -> JudgeScore:
        score = self._scores.get(video_id)
        if score is None:
            return JudgeScore(None, SCALE, self._stamp, "missing_score")
        return JudgeScore(
            score.score, SCALE, self._stamp, score.reason if score.score is None else None
        )


PLUGINS = (VerifierJudge,)
