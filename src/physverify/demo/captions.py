"""What the demo videos say: the arm, S and the rule in the top bar, and one caption per check.

A check the active rule decided says what fed S: under a calibrated rule its violation
probability p_c, under another rule its verdict and magnitude; only a check that abstained reads
ABSTAIN. Badges go to the checks the prompt targets, the checks the rule decided and every FAIL.
Model tags become arm names (base, fine-tuned, given video) and the rule label a short rule name;
the tags and the label stay in the report files.
"""

from __future__ import annotations

from typing import Final

from physverify.demo.report import (
    NO_VERDICT,
    VIDEO_TAG_PREFIX,
    CheckRow,
    RuleInfo,
    VideoReport,
    plus_minus,
    score_text,
)
from physverify.finetune.records import ADAPTER_TAG_PREFIX, BASE_TAG
from physverify.schema.checks import AbstainReason, Verdict

SEPARATOR: Final = " | "
BASE_ARM: Final = "base"
TUNED_ARM: Final = "fine-tuned"
GIVEN_ARM: Final = "given video"
SHORT_REASONS: Final = {AbstainReason.INTERVAL_CROSSES_THRESHOLD.value: "too close to call"}


def arm_name(model_tag: str) -> str:
    if model_tag == BASE_TAG:
        return BASE_ARM
    if model_tag.startswith(ADAPTER_TAG_PREFIX):
        return TUNED_ARM
    if model_tag.startswith(VIDEO_TAG_PREFIX):
        return GIVEN_ARM
    return model_tag


def _readable(token: str) -> str:
    return token.replace("_", " ")


def rule_text(rule: RuleInfo) -> str:
    if rule.calibrated:
        return f"calibrated rule ({rule.scorer})"
    return "PROVISIONAL rule, not calibrated"


def bar_segments(report: VideoReport) -> list[str]:
    return [
        arm_name(report.model_tag),
        _readable(score_text(report.clip)),
        f"coverage {report.clip.coverage:.2f}",
        rule_text(report.rule),
    ]


def bar_text(report: VideoReport) -> str:
    return SEPARATOR.join(bar_segments(report))


def violation(row: CheckRow, calibrated: bool) -> float | None:
    """The rule's violation probability of a check it decided, under a calibrated rule."""
    return row.rule_term if calibrated and row.rule_decided else None


def _reason(row: CheckRow) -> str:
    if not row.abstain_reasons:
        return ""
    first = row.abstain_reasons[0]
    return f" ({SHORT_REASONS.get(first, _readable(first).lower())})"


def _measured(row: CheckRow) -> str:
    return f"m = {plus_minus(row.magnitude, row.sigma)}"


def badge_text(row: CheckRow, calibrated: bool = False) -> str:
    p = violation(row, calibrated)
    if p is not None:
        return f"{row.check_id} P(violation) {p:.2f}"
    if row.verdict == NO_VERDICT:
        return f"{row.check_id} no event"
    if row.verdict == Verdict.ABSTAIN.value:
        return f"{row.check_id} {row.verdict}{_reason(row)}"
    if row.verdict not in (Verdict.PASS.value, Verdict.FAIL.value):
        return f"{row.check_id} {row.verdict}"
    unused = "" if row.rule_decided else ", not in S"
    return f"{row.check_id} {row.verdict} {_measured(row)}{unused}"


def shown_rows(report: VideoReport, all_checks: bool = False) -> list[CheckRow]:
    if all_checks:
        return list(report.checks)
    targets = set(report.prompt.target_checks)
    return [
        row
        for row in report.checks
        if row.check_id in targets or row.rule_decided or row.verdict == Verdict.FAIL.value
    ]
