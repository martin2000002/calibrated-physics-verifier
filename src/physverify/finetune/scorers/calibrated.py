"""Calibrated score: a thin wrapper over the calibration chain's ``score_clip``.

No score formula lives here; the calibration's clip score maps one to one onto
``AggregateScore``. ``k_sigma`` is not passed: ``score_clip`` uses the one the stratum was fitted
with. The stratum is opened through ``_calibrated_gate``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.calibration.api import ScoredClip
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.scorers._calibrated_gate import CalibratedGate
from physverify.schema.checks import CheckResult


def to_aggregate(clip_score: ScoredClip) -> AggregateScore:
    decided = frozenset(str(check) for check in clip_score.decided_checks)
    return AggregateScore(
        score=clip_score.score,
        sigma=clip_score.sigma,
        abstained=bool(clip_score.abstained),
        reason=clip_score.reason,
        n_decided=len(decided),
        n_failed=int(clip_score.n_failed),
        coverage=float(clip_score.coverage),
        decided=decided,
        per_check={str(k): float(v) for k, v in clip_score.per_check.items()},
    )


class CalibratedScorer(CalibratedGate):
    name: ClassVar[str] = "calibrated"
    version: ClassVar[str] = "3"
    calibrated: ClassVar[bool] = True

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        api, stratum = self._opened(results)
        return to_aggregate(api.score_clip(results, stratum, only=only))


PLUGINS = (CalibratedScorer,)
