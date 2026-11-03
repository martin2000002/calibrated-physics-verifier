"""Whether a check counts in n_met: the one rule of the dev score and of the held-out evaluation.

A check counts when it is monotone (``h1_pass``), its informedness J has J^lo > 0 and, where the
scenes carry families, its stratified block meets too: stratified AUROC^lo > 0.5 in both readings
and stratified J^lo > 0. Without families the stratified part does not apply. Pure, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from physverify.calibration.artifact import READINGS, Monotonicity, Stratified, Weight
from physverify.calibration.stratified import above

AUROC_FLOOR = 0.5


@dataclass(frozen=True)
class Clause:
    meets_h1: bool
    meets_j: bool
    meets_stratified: bool | None
    """None where the set has no families."""

    @property
    def meets_both(self) -> bool:
        return self.meets_h1 and self.meets_j

    @property
    def meets_clause(self) -> bool:
        return self.meets_both and self.meets_stratified is not False


def meets_stratified(scores: Stratified) -> bool:
    aurocs = all(above(scores.auroc_cv[r].ci.low, AUROC_FLOOR) for r in READINGS)
    return aurocs and scores.j_cv is not None and above(scores.j_cv.informedness_ci.low, 0.0)


def clause(h1: Monotonicity, j: Weight, stratified: Stratified | None) -> Clause:
    return Clause(
        meets_h1=h1.h1_pass,
        meets_j=above(j.informedness_ci.low, 0.0),
        meets_stratified=None if stratified is None else meets_stratified(stratified),
    )
