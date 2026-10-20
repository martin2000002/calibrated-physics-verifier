"""Paired comparison items: human label, reference VLM and verifier decision per video (D3-D5).

Positive class = implausible (what the verifier detects). Every score is oriented as an
implausibility: the verifier's rule score, the VLM's negated plausibility score, the human's
negated raw rating.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from physverify.agreement.decision import ERROR_REASON, ClipDecision, ClipDecisionRule, Decision
from physverify.agreement.run_io import VerifyRun
from physverify.datasets._videophy_tables import VideophyRow
from physverify.verification.runner import TableRow


@dataclass(frozen=True)
class ComparisonItem:
    row: VideophyRow
    human_implausible: bool
    vlm_implausible: bool | None
    decision: Decision
    outcome: ClipDecision
    check_rows: tuple[TableRow, ...] = field(default=(), compare=False)

    @property
    def clip_id(self) -> str:
        return self.row.clip_id

    @property
    def human_score(self) -> float:
        return -self.row.human_raw

    @property
    def vlm_score(self) -> float:
        return -self.row.reference_raw


@dataclass(frozen=True)
class JoinCounts:
    """Rows of the subset and why some are not compared (D3: NaN labels are excluded, counted).
    ``not_verified`` rows have no clip in the run (video missing from the source or unreadable);
    they are compared as ``check_error`` abstentions (D7)."""

    rows: int
    human_missing: int
    not_verified: int
    vlm_missing: int
    compared: int


def implausible(raw: float, cut: float) -> bool | None:
    """Below the published cut is implausible; NaN is no label."""
    return None if math.isnan(raw) else raw < cut


def unverified(clip_id: str) -> Decision:
    """A clip the pipeline failed on: an abstention for ``check_error``, never left out, so the
    as_plausible policy counts it as plausible like any other undecided clip."""
    return Decision(clip_id, ClipDecision.ABSTAIN, math.nan, None, (ERROR_REASON,))


def decide_all(rule: ClipDecisionRule, run: VerifyRun) -> dict[str, Decision]:
    return {
        clip_id: rule.decide(clip_id, rows, run.results.get(clip_id, ()))
        for clip_id, rows in run.rows.items()
    }


def build_items(
    rows: Sequence[VideophyRow],
    decisions: Mapping[str, Decision],
    run: VerifyRun,
    cut: float | None,
) -> tuple[list[ComparisonItem], JoinCounts]:
    items: list[ComparisonItem] = []
    human_missing = not_verified = vlm_missing = 0
    for row in rows:
        human = implausible(row.human_raw, row.human_cut)
        if human is None:
            human_missing += 1
            continue
        decision = decisions.get(row.clip_id)
        if decision is None:
            not_verified += 1
            decision = unverified(row.clip_id)
        vlm = implausible(row.reference_raw, row.reference_cut)
        vlm_missing += vlm is None and bool(row.reference_name)
        outcome = decision.outcome(cut)
        checks = run.rows.get(row.clip_id, ())
        items.append(ComparisonItem(row, human, vlm, decision, outcome, checks))
    counts = JoinCounts(len(rows), human_missing, not_verified, vlm_missing, len(items))
    return items, counts
