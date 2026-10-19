"""The per-check breakdown follows the compared rule and the primary policy (ADR 0011 D9)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from physverify.agreement.abstention import AS_IMPLAUSIBLE, AS_PLAUSIBLE
from physverify.agreement.breakdown import per_check, policy_accuracy
from physverify.agreement.decision import ClipDecision, ProvisionalAnyFail
from physverify.agreement.items import ComparisonItem, build_items, decide_all
from physverify.agreement.run_io import load_run

THRESHOLDS = {"status": "provisional", "checks": {"R1": {"tau": 0.05}, "R3a": {"tau": 0.05}}}


def _items(world) -> list[ComparisonItem]:
    clips = {
        "v1t/a1": [world.FAIL],
        "v1t/a2": [world.CROSSING],
        "v1t/b1": [world.PASS],
        "v1t/b2": [world.NO_EVENT],
    }
    run = load_run(world.run("r", clips))
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

    plausible = per_check(items, redecide, AS_PLAUSIBLE)
    pessimistic = per_check(items, redecide, AS_IMPLAUSIBLE)
    assert plausible and all(row["fail_rate_human_implausible"] == 0.5 for row in plausible)
    assert all(row["fail_rate_human_plausible"] == 0.0 for row in plausible)
    assert [row["accuracy_alone"] for row in plausible] == [pytest.approx(0.75)] * len(plausible)
    assert [row["accuracy_alone"] for row in pessimistic] == [pytest.approx(0.5)] * len(pessimistic)
    check_ids = {row["check_id"] for row in plausible}
    assert all(excluded <= check_ids for excluded in seen)


def test_policy_accuracy_counts_undecided_clips_by_the_policy(world) -> None:
    items = _with(_items(world), *[ClipDecision.ABSTAIN] * 4)
    assert policy_accuracy(items, AS_PLAUSIBLE) == pytest.approx(0.5)
    assert policy_accuracy(items, AS_IMPLAUSIBLE) == pytest.approx(0.5)
    decided = _with(items, *[ClipDecision.PLAUSIBLE] * 4)
    assert policy_accuracy(decided, AS_IMPLAUSIBLE) == pytest.approx(0.5)
