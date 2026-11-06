"""A result outside its check's declared scope (NOT_APPLICABLE) leaves the populations of J_cv and
the clip rule; an abstention inside the scope still counts against a positive; the operational
AUROC reading keeps every labelled row."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

from calibration_cases import check_calibration, missing, result, row

from physverify.calibration.oof import check_valued, check_values, in_informedness
from physverify.calibration.table import CalibrationRow, Role
from physverify.calibration.thresholds import fails
from physverify.calibration.verdict import NO_EVENT, Rule, clip_verdict
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import RowStatus, aggregate
from physverify.verification.thresholding import ThresholdRule

RULE = Rule(ThresholdPolicy(), bias=0.0, factor=1.0)
TAU = 0.08
CHECK_ID = "R3b"


class StubCheck:
    check_id: ClassVar[str] = CHECK_ID
    magnitude_unit: ClassVar[str] = "u"


def measured(subject: str, magnitude: float) -> CheckResult:
    return result(CHECK_ID, magnitude=magnitude, subject=subject)


def abstained(subject: str, reason: AbstainReason) -> CheckResult:
    return replace(missing(CHECK_ID, subject=subject), abstain_reason=reason)


def not_applicable(subject: str) -> CheckResult:
    return abstained(subject, AbstainReason.NOT_APPLICABLE)


def made(subjects: Sequence[CheckResult], role: Role, scene: str = "s0") -> CalibrationRow:
    """The calibration row the runner's table gives for ``subjects``."""
    stub: Any = StubCheck()
    table = aggregate("clip", stub, subjects)
    return row(
        table.magnitude,
        role=role,
        scene=scene,
        check_id=CHECK_ID,
        status=table.status,
        sigma=table.sigma,
        subjects=subjects,
    )


def test_a_row_of_not_applicable_events_leaves_both_populations_and_the_clip_rule() -> None:
    for role in (Role.POSITIVE, Role.CLEAN):
        outside = made([not_applicable("a"), not_applicable("b")], role)
        assert outside.status is RowStatus.NO_EVENT
        assert not outside.has_event and not in_informedness(outside)
        assert clip_verdict(outside.subjects, RULE, TAU) == NO_EVENT


def test_a_not_applicable_event_neither_votes_nor_abstains_the_clip() -> None:
    passing = made([not_applicable("a"), measured("b", 0.01)], Role.CLEAN)
    assert passing.scored and in_informedness(passing)
    assert clip_verdict(passing.subjects, RULE, TAU).verdict is Verdict.PASS
    failing = made([not_applicable("a"), measured("b", 0.5)], Role.POSITIVE)
    assert fails(failing, RULE, TAU)
    out_of_scope = [abstained("a", AbstainReason.OUT_OF_SCOPE), measured("b", 0.01)]
    assert clip_verdict(out_of_scope, RULE, TAU).verdict is Verdict.ABSTAIN


def test_an_abstention_inside_the_scope_still_counts_against_a_positive() -> None:
    inside = made(
        [abstained("a", AbstainReason.ILL_CONDITIONED), not_applicable("b")], Role.POSITIVE
    )
    assert inside.status is RowStatus.ABSTAINED
    assert in_informedness(inside) and not fails(inside, RULE, TAU)
    assert clip_verdict(inside.subjects, RULE, TAU).verdict is Verdict.ABSTAIN


def test_the_operational_reading_keeps_the_not_applicable_rows() -> None:
    rows = [
        made([not_applicable("a")], Role.POSITIVE, "s0"),
        made([measured("a", 0.5)], Role.POSITIVE, "s1"),
        made([measured("a", 0.01)], Role.CLEAN, "s2"),
    ]
    values = check_values(rows, check_calibration(tau=TAU), ThresholdRule())
    valued = check_valued(values)
    assert set(valued) == {r.clip_id for r in rows}
    outside = valued[rows[0].clip_id]
    assert outside.value is None and outside.positive
    assert [v.flag for v in values] == [None, True, False]
