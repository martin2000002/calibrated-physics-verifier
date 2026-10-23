"""The per-check breakdown follows the compared rule and the primary policy."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from physverify.human_agreement.breakdown import per_check, policy_accuracy
from physverify.human_agreement.clip_decision import ClipDecision, ProvisionalAnyFail
from physverify.human_agreement.comparison_items import ComparisonItem, build_items, decide_all
from physverify.human_agreement.undecided_policy import AS_IMPLAUSIBLE, AS_PLAUSIBLE
from physverify.human_agreement.verify_run_reader import VerifyRun, load_run

THRESHOLDS = {"status": "provisional", "checks": {"R1": {"tau": 0.05}, "R3a": {"tau": 0.05}}}


def _run(world) -> VerifyRun:
    clips = {
        "v1t/a1": [world.FAIL],
        "v1t/a2": [world.CROSSING],
        "v1t/b1": [world.PASS],
        "v1t/b2": [world.NO_EVENT],
    }
    return load_run(world.run("r", clips))


def _items(world, run: VerifyRun | None = None) -> list[ComparisonItem]:
    run = run or _run(world)
    rows = world.dataset.candidates("v1_test")["v1_test"]
    return build_items(rows, decide_all(ProvisionalAnyFail(THRESHOLDS), run), run, None)[0]


def _with(items: Sequence[ComparisonItem], *outcomes: ClipDecision) -> list[ComparisonItem]:
    return [dataclasses.replace(it, outcome=o) for it, o in zip(items, outcomes, strict=True)]


def test_per_check_uses_the_redecided_outcomes_and_the_given_policy(world) -> None:
    items = _items(world)
    human = [it.human_implausible for it in items]
    assert human == [False, True, True, False]
    abstain, implausible = ClipDecision.ABSTAIN, ClipDecision.IMPLAUSIBLE
    redecided = _with(items, abstain, implausible, abstain, abstain)
    seen: list[frozenset[str]] = []

    def redecide(excluded: frozenset[str]) -> list[ComparisonItem]:
        seen.append(excluded)
        return redecided

    checks = {r.check_id for it in items for r in it.check_rows}
    plausible = per_check(items, redecide, checks, AS_PLAUSIBLE)
    pessimistic = per_check(items, redecide, checks, AS_IMPLAUSIBLE)
    assert plausible
    assert all(row["implausible_rate_alone_human_implausible"] == 0.5 for row in plausible)
    assert all(row["implausible_rate_alone_human_plausible"] == 0.0 for row in plausible)
    assert [row["accuracy_alone"] for row in plausible] == [pytest.approx(0.75)] * len(plausible)
    assert [row["accuracy_alone"] for row in pessimistic] == [pytest.approx(0.5)] * len(pessimistic)
    assert all(excluded <= checks for excluded in seen)


def test_per_check_covers_the_rule_check_set_including_result_only_checks(world) -> None:
    run = _run(world)
    items = _items(world, run)
    extra = "R9"
    every = run.check_ids | {extra}
    assert extra not in {r.check_id for it in items for r in it.check_rows}
    seen: list[frozenset[str]] = []

    def redecide(excluded: frozenset[str]) -> list[ComparisonItem]:
        seen.append(excluded)
        return items

    rows = per_check(items, redecide, every, AS_PLAUSIBLE)
    assert [row["check_id"] for row in rows] == sorted(every)
    assert next(row for row in rows if row["check_id"] == extra)["n_scored"] == 0
    assert every - {extra} in seen and frozenset({extra}) in seen


def test_the_run_check_set_is_the_union_of_rows_and_results(world) -> None:
    run = _run(world)
    clip, rows = next(iter(run.rows.items()))
    result = dataclasses.replace(run.results[clip][0], check_id="R9")
    widened = dataclasses.replace(run, results={clip: (*run.results[clip], result)})
    assert {r.check_id for r in rows} <= run.check_ids
    assert widened.check_ids == run.check_ids | {"R9"}


def test_policy_accuracy_counts_undecided_clips_by_the_policy(world) -> None:
    items = _with(_items(world), *[ClipDecision.ABSTAIN] * 4)
    assert policy_accuracy(items, AS_PLAUSIBLE) == pytest.approx(0.5)
    assert policy_accuracy(items, AS_IMPLAUSIBLE) == pytest.approx(0.5)
    decided = _with(items, *[ClipDecision.PLAUSIBLE] * 4)
    assert policy_accuracy(decided, AS_IMPLAUSIBLE) == pytest.approx(0.5)
