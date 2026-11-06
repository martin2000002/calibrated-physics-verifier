"""The decisive subject of a row: NaN and None rank last, abstained subjects only as a fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.verification.decisive import decisive, size
from physverify.verification.runner import RowStatus, aggregate

NAN = math.nan


@dataclass(frozen=True)
class Item:
    verdict: Verdict
    magnitude: float | None
    abstain_reason: AbstainReason | None = None


def test_size_ranks_none_and_nan_below_every_number() -> None:
    assert size(None) == size(NAN) == -math.inf
    assert size(-5.0) == -5.0


def test_nan_first_does_not_hide_a_larger_subject() -> None:
    items = [Item(Verdict.PASS, NAN), Item(Verdict.PASS, 0.2), Item(Verdict.PASS, 0.1)]
    assert decisive(items) is items[1]


def test_the_largest_fail_wins_over_larger_other_subjects() -> None:
    items = [Item(Verdict.ABSTAIN, 9.0), Item(Verdict.FAIL, 0.3), Item(Verdict.PASS, 1.0)]
    assert decisive(items) is items[1]


def test_an_abstained_subject_never_wins_over_a_non_abstained_one() -> None:
    items = [Item(Verdict.ABSTAIN, 9.0), Item(Verdict.PASS, 0.1)]
    assert decisive(items) is items[1]
    assert decisive([Item(Verdict.ABSTAIN, 9.0), Item(Verdict.UNSCORED, NAN)]) == Item(
        Verdict.UNSCORED, NAN
    )


def test_all_abstained_falls_back_to_the_largest_and_nan_beats_none() -> None:
    items = [Item(Verdict.ABSTAIN, None), Item(Verdict.ABSTAIN, NAN), Item(Verdict.ABSTAIN, 0.1)]
    assert decisive(items) is items[2]
    assert decisive(items[:2]) is items[1]
    assert decisive([]) is None


class StubCheck:
    check_id: ClassVar[str] = "R1"
    magnitude_unit: ClassVar[str] = "pen_over_size"


CHECK: Any = StubCheck()


def _results(*specs: tuple[str, float, Verdict]) -> list[CheckResult]:
    return [_result(subject, magnitude, verdict) for subject, magnitude, verdict in specs]


def _result(subject: str, magnitude: float, verdict: Verdict) -> CheckResult:
    reason = AbstainReason.INTERVAL_CROSSES_THRESHOLD if verdict is Verdict.ABSTAIN else None
    return CheckResult(
        check_id=CHECK.check_id,
        clip_id="clip",
        subject=(subject,),
        frame_span=None,
        magnitude=magnitude,
        sigma=0.0,
        unit=CHECK.magnitude_unit,
        verdict=verdict,
        input_level=InputLevel.ORACLE,
        abstain_reason=reason,
    )


def test_a_pass_and_an_abstain_row_stays_abstain_with_the_pass_magnitude() -> None:
    results = _results(("a", 0.9, Verdict.ABSTAIN), ("b", 0.1, Verdict.PASS))
    row = aggregate("clip", CHECK, results)
    assert row.verdict is Verdict.ABSTAIN and row.magnitude == 0.1


def test_a_fail_and_a_nan_row_fails_with_the_fail_magnitude() -> None:
    results = _results(("a", NAN, Verdict.ABSTAIN), ("b", 0.4, Verdict.FAIL))
    row = aggregate("clip", CHECK, results)
    assert row.verdict is Verdict.FAIL and row.magnitude == 0.4


def test_an_all_nan_row_is_scored_with_nan_not_none() -> None:
    results = _results(("a", NAN, Verdict.ABSTAIN), ("b", NAN, Verdict.ABSTAIN))
    row = aggregate("clip", CHECK, results)
    assert row.status is RowStatus.SCORED
    assert row.magnitude is not None and math.isnan(row.magnitude)


def test_a_nan_subject_listed_first_does_not_set_the_row_magnitude() -> None:
    results = _results(("a", NAN, Verdict.PASS), ("b", 0.2, Verdict.PASS))
    row = aggregate("clip", CHECK, results)
    assert row.verdict is Verdict.PASS and row.magnitude == 0.2


def _not_applicable(subject: str) -> CheckResult:
    found = _result(subject, NAN, Verdict.ABSTAIN)
    return replace(found, magnitude=None, abstain_reason=AbstainReason.NOT_APPLICABLE)


def test_a_not_applicable_subject_is_never_decisive() -> None:
    outside = Item(Verdict.ABSTAIN, None, AbstainReason.NOT_APPLICABLE)
    assert decisive([outside]) is None
    assert decisive([outside, Item(Verdict.ABSTAIN, None)]) == Item(Verdict.ABSTAIN, None)


def test_a_row_of_not_applicable_subjects_has_no_event() -> None:
    row = aggregate("clip", CHECK, [_not_applicable("a"), _not_applicable("b")])
    assert row.status is RowStatus.NO_EVENT and row.verdict is None
    assert row.abstain_reason is None and (row.n_subjects, row.n_abstained) == (0, 0)


def test_a_not_applicable_subject_neither_votes_nor_abstains_the_row() -> None:
    passing = aggregate("clip", CHECK, [_not_applicable("a"), *_results(("b", 0.1, Verdict.PASS))])
    assert passing.status is RowStatus.SCORED and passing.verdict is Verdict.PASS
    assert passing.magnitude == 0.1 and passing.n_subjects == 1
    abstained = aggregate(
        "clip", CHECK, [_not_applicable("a"), *_results(("b", 0.1, Verdict.ABSTAIN))]
    )
    assert abstained.status is RowStatus.SCORED and abstained.verdict is Verdict.ABSTAIN
    assert abstained.abstain_reason is AbstainReason.INTERVAL_CROSSES_THRESHOLD
