"""B1, the verifier's ordering score: ``B1 = 1 - S_graded`` over the checks outside
``[check_scope]`` (``exclude_checks``, time-dependent checks kept), under the frozen artifact's
perceived stratum. One definition for the held-out aggregate AUROC, the pair accuracy against the
external judge, the real-video cut u_B1 and the generated-video comparison."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from physverify.finetune.scorers.graded import graded_score, monotonicity_weight, stratum_weights
from physverify.schema.checks import CheckResult

if TYPE_CHECKING:
    from physverify.calibration.api import CalibrationApi
    from physverify.calibration.artifact_io import Stratum
    from physverify.calibration.clip_values import ClipValueOf


class B1:
    """``excluded``: the checks the application's ``excluded_checks`` returns."""

    def __init__(self, api: CalibrationApi[Stratum], excluded: frozenset[str]) -> None:
        self._api, self._excluded = api, excluded

    @property
    def excluded(self) -> frozenset[str]:
        return self._excluded

    def under(self, stratum: Stratum) -> ClipValueOf:
        """The clip value under ``stratum``; None when no check grades the clip."""
        weights = stratum_weights(stratum, monotonicity_weight)

        def value(results: Sequence[CheckResult]) -> float | None:
            only = frozenset(r.check_id for r in results) - self._excluded
            score = graded_score(self._api, results, stratum, weights, only).score
            return None if score is None else 1.0 - score

        return value
