"""Clip decisions, the join with the labels and the abstention policies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physverify.calibration.api import CalibrationApi
from physverify.calibration.apply import violation_probability, vote, weighted_score
from physverify.core.errors import ConfigError, UsageError
from physverify.human_agreement.clip_decision import (
    CALIBRATED_FAIL_ONLY_NAME,
    NO_FAILED_REASON,
    CalibratedRule,
    ClipDecision,
    ProvisionalAnyFail,
    StratumRequest,
    UnusableCalibrationError,
    calibrated_rule,
)
from physverify.human_agreement.comparison_items import build_items, decide_all
from physverify.human_agreement.undecided_policy import (
    AS_IMPLAUSIBLE,
    AS_PLAUSIBLE,
    COVERAGE_MATCHED_VLM,
    DECIDED_ONLY,
    abstention_counts,
    policy_view,
)
from physverify.human_agreement.verify_run_reader import VerifyRun, load_run, run_row_step
from physverify.schema.checks import InputLevel
from physverify.verification.runner import RowStatus
from physverify.verification.thresholding import ThresholdTable

THRESHOLDS = ThresholdTable.provisional({"R1": 0.05, "R3a": 0.05})


def _decide(world, rows, excluded=()):
    run = load_run(world.run("r", {"v1t/a1": rows}))
    rule = ProvisionalAnyFail(THRESHOLDS, excluded)
    return decide_all(rule, run)[world.clip("v1t/a1")]


def test_any_fail_is_implausible_and_scored_by_magnitude_over_tau(world) -> None:
    decision = _decide(world, [world.FAIL, world.NO_EVENT])
    assert decision.fixed is ClipDecision.IMPLAUSIBLE
    assert decision.score == pytest.approx(0.10 / 0.05)
    assert decision.deciding_check == "R1"


def test_a_pass_without_fail_is_plausible(world) -> None:
    assert _decide(world, [world.PASS, world.NO_EVENT]).fixed is ClipDecision.PLAUSIBLE


def test_only_no_event_rows_give_no_evidence(world) -> None:
    decision = _decide(world, [world.NO_EVENT])
    assert decision.fixed is ClipDecision.NO_EVIDENCE and math.isnan(decision.score)


def test_abstentions_and_errors_abstain_with_their_reasons(world) -> None:
    error = world.Row("R3a", None, None, RowStatus.ERROR)
    decision = _decide(world, [world.CROSSING, error])
    assert decision.fixed is ClipDecision.ABSTAIN
    assert decision.reasons == ("INTERVAL_CROSSES_THRESHOLD", "check_error")


def test_excluded_checks_do_not_decide(world) -> None:
    assert _decide(world, [world.FAIL], excluded=("R1",)).fixed is ClipDecision.ABSTAIN


@dataclass(frozen=True)
class Score:
    score: float | None
    reason: str | None = None
    abstained_checks: tuple[str, ...] = ()
    unweighted_checks: tuple[str, ...] = ()
    per_check: Any = field(default_factory=lambda: MappingProxyType({}))
    decided_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()


class FakeStratum:
    source = "calibration.json#perceived/precise@cal-1"

    def tau(self, check_id: str) -> float | None:
        return 0.05

    def score_threshold(self) -> float | None:
        return 0.9

    def check_status(self) -> dict[str, Any]:
        return {"source": self.source, "checks": {"R1": {"status": "ok", "weight": 1.0}}}


class FakeError(Exception):
    pass


class FakeUnusableError(FakeError):
    pass


def _api(score: Score, problem: Exception | None = None) -> tuple[CalibrationApi, list[Any]]:
    calls: list[Any] = []

    def open_stratum(path: Path, level: InputLevel, profile: str, **kwargs: Any) -> FakeStratum:
        calls.append(("open", level, profile, kwargs))
        if problem is not None:
            raise problem
        return FakeStratum()

    def score_clip(
        results: Any, stratum: Any, k_sigma: float | None = None, *, only: Any = None
    ) -> Score:
        calls.append((k_sigma, only))
        return score

    api = CalibrationApi(
        open_stratum=open_stratum,
        default_artifact_path=lambda settings: Path("unused"),
        active_checks_hash=lambda settings, checks: "hash",
        require_calibrated=lambda settings: False,
        score_clip=score_clip,
        unique_row_step=max,
        error=FakeError,
        unusable_error=FakeUnusableError,
        violation_probability=violation_probability,
        vote=vote,
        weighted_score=weighted_score,
    )
    return api, calls


def _request(profile: str = "fast", row_step: int = 1) -> StratumRequest:
    return StratumRequest(Path("calibration.json"), profile, row_step, "hash", True)


def test_the_calibrated_rule_scores_one_minus_s(world) -> None:
    api, calls = _api(Score(0.25, per_check={"R1": 0.7, "R3a": 0.1}, abstained_checks=("R5b",)))
    rule = calibrated_rule(api, _request("precise", 2), ())
    opened = calls.pop(0)
    assert opened == (
        "open",
        InputLevel.PERCEIVED,
        "precise",
        {"checks_hash": "hash", "row_step": 2, "require_calibrated": True},
    )
    assert calls.pop(0) == (None, None)  # the k_sigma probe at opening
    run = load_run(world.run("r", {"v1t/a1": [world.FAIL, world.NO_EVENT]}))
    decision = decide_all(rule, run)[world.clip("v1t/a1")]
    assert decision.fixed is None and decision.score == pytest.approx(0.75)
    assert decision.deciding_check == "R1" and decision.reasons == ("R5b",)
    assert decision.outcome(0.5) is ClipDecision.IMPLAUSIBLE
    assert decision.outcome(0.8) is ClipDecision.PLAUSIBLE
    assert rule.kubric_threshold == 0.9 and "precise@cal-1" in rule.label
    assert rule.status["checks"] == {"R1": {"status": "ok", "weight": 1.0}}
    assert calls == [(None, None)]  # the stratum's own k_sigma
    decide_all(rule.excluding(["R1"]), run)
    assert calls[-1] == (None, {"R3a"})


def test_s_none_is_an_abstention_with_the_scorer_reason(world) -> None:
    rule = calibrated_rule(_api(Score(None, "no_informative_check"))[0], _request(), ())
    run = load_run(world.run("r", {"v1t/a1": [world.CROSSING]}))
    decision = decide_all(rule, run)[world.clip("v1t/a1")]
    assert decision.fixed is ClipDecision.ABSTAIN
    assert decision.reasons == ("no_informative_check",)


def test_a_configuration_problem_is_a_config_error() -> None:
    api, _ = _api(Score(1.0), FakeError("no calibration artifact at x: run `calibrate fit`"))
    with pytest.raises(ConfigError, match="no calibration artifact"):
        calibrated_rule(api, _request(), ())


def test_an_unusable_stratum_is_left_to_the_caller() -> None:
    api, _ = _api(Score(1.0), FakeUnusableError("aggregate not ready"))
    with pytest.raises(UnusableCalibrationError, match="aggregate not ready") as caught:
        calibrated_rule(api, _request(), ())
    assert not isinstance(caught.value, ConfigError)


def _fail_only(world, score: Score, k_sigma: float | None = 2.0):
    api, calls = _api(score)
    rule = calibrated_rule(api, _request(), (), k_sigma=k_sigma, fail_only=True)
    run = load_run(world.run("r", {"v1t/a1": [world.FAIL, world.PASS]}))
    return rule, decide_all(rule, run)[world.clip("v1t/a1")], calls


def test_fail_only_is_implausible_on_a_failed_voting_check(world) -> None:
    score = Score(
        0.2,
        per_check={"R1": 0.9, "R3a": 0.3, "R4": 0.95},
        decided_checks=("R1", "R3a", "R4"),
        failed_checks=("R1", "R3a"),
        abstained_checks=("R5b",),
    )
    rule, decision, calls = _fail_only(world, score)
    assert rule.name == CALIBRATED_FAIL_ONLY_NAME and rule.label.startswith("calibrated_fail_only:")
    assert decision.fixed is ClipDecision.IMPLAUSIBLE and decision.score == pytest.approx(0.8)
    assert decision.deciding_check == "R1" and decision.reasons == ("R5b",)
    assert decision.decided_checks == ("R1", "R3a", "R4")
    assert rule.kubric_threshold is None
    assert {c[0] for c in calls if c[0] != "open"} == {2.0}  # k pinned at opening and per clip


def test_fail_only_never_reads_a_pass_as_plausible(world) -> None:
    score = Score(0.9, per_check={"R1": 0.1}, decided_checks=("R1",))
    _, decision, _ = _fail_only(world, score)
    assert decision.fixed is ClipDecision.ABSTAIN and decision.reasons == (NO_FAILED_REASON,)
    assert decision.outcome(None) is ClipDecision.ABSTAIN and decision.decided_checks == ("R1",)


def test_fail_only_without_any_event_has_no_evidence(world) -> None:
    _, decision, _ = _fail_only(world, Score(None, "no_decided_check"))
    assert decision.fixed is ClipDecision.NO_EVIDENCE and math.isnan(decision.score)


@pytest.mark.parametrize(
    ("score", "reasons"),
    [
        (Score(None, "no_weighted_check", unweighted_checks=("R1",)), (NO_FAILED_REASON,)),
        (Score(None, "no_decided_check", abstained_checks=("R5b", "R2")), ("R2", "R5b")),
    ],
)
def test_fail_only_abstains_on_events_without_a_voting_fail(
    world, score: Score, reasons: tuple[str, ...]
) -> None:
    _, decision, _ = _fail_only(world, score)
    assert decision.fixed is ClipDecision.ABSTAIN and decision.reasons == reasons
    assert decision.decided_checks == ()


def test_fail_only_keeps_its_mode_when_excluding(world) -> None:
    rule, _, _ = _fail_only(world, Score(0.9))
    assert rule.excluding(["R1"]).name == CALIBRATED_FAIL_ONLY_NAME


def test_a_k_sigma_the_stratum_was_not_fitted_with_fails_at_opening() -> None:
    api, _ = _api(Score(1.0))

    def mismatched(results: Any, stratum: Any, k_sigma: float | None = None, **_: Any) -> Score:
        raise FakeError(f"k_sigma {k_sigma} differs from the 2.0 the stratum was calibrated with")

    api = CalibrationApi(**{**api.__dict__, "score_clip": mismatched})
    with pytest.raises(ConfigError, match=r"k_sigma 3\.0 differs"):
        calibrated_rule(api, _request(), (), k_sigma=3.0, fail_only=True)


def test_a_calibrated_decision_needs_the_frozen_cut(world) -> None:
    rule: CalibratedRule = calibrated_rule(_api(Score(0.5))[0], _request(), ())
    run = load_run(world.run("r", {"v1t/a1": [world.FAIL]}))
    with pytest.raises(ConfigError, match="agreement freeze"):
        build_items(
            world.dataset.candidates("v1_test")["v1_test"], decide_all(rule, run), run, None
        )


def _items(world):
    clips = {
        "v1t/a1": [world.FAIL],  # human plausible (1), VLM plausible (0.8)
        "v1t/a2": [world.CROSSING],  # human implausible, VLM implausible
        "v1t/b1": [world.PASS],  # human implausible, VLM implausible
        "v1t/b2": [world.NO_EVENT],  # human plausible, VLM plausible
        "v1t/b3": [world.FAIL],  # no human label
    }
    run = load_run(world.run("r", clips))
    rows = world.dataset.candidates("v1_test")["v1_test"]
    return build_items(rows, decide_all(ProvisionalAnyFail(THRESHOLDS), run), run, None)


def test_the_join_counts_rows_without_label_or_verification(world) -> None:
    items, counts = _items(world)
    assert (counts.rows, counts.human_missing, counts.not_verified, counts.compared) == (5, 1, 0, 4)
    assert [it.human_implausible for it in items] == [False, True, True, False]
    assert [it.vlm_implausible for it in items] == [False, True, True, False]


def test_a_clip_missing_from_the_run_is_compared_as_a_check_error(world) -> None:
    run = load_run(world.run("r", {"v1t/a1": [world.FAIL]}))
    rows = world.dataset.candidates("v1_test")["v1_test"]
    items, counts = build_items(rows, decide_all(ProvisionalAnyFail(THRESHOLDS), run), run, None)
    assert (counts.not_verified, counts.compared) == (3, 4)
    missing = [it for it in items if it.clip_id != world.clip("v1t/a1")]
    assert all(it.outcome is ClipDecision.ABSTAIN for it in missing)
    assert all(it.decision.reasons == ("check_error",) for it in missing)
    assert policy_view(items, AS_PLAUSIBLE).verifier.tolist() == [1, 0, 0, 0]


def test_the_policies_count_undecided_clips_as_declared(world) -> None:
    items, _ = _items(world)
    plausible = policy_view(items, AS_PLAUSIBLE)
    assert plausible.verifier.tolist() == [1, 0, 0, 0]
    assert policy_view(items, AS_IMPLAUSIBLE).verifier.tolist() == [1, 1, 0, 1]
    decided = policy_view(items, DECIDED_ONLY)
    assert decided.index == (0, 2) and decided.paired
    matched = policy_view(items, COVERAGE_MATCHED_VLM)
    assert matched.index == (0, 2) and not matched.paired and len(matched.vlm_index) == 2
    assert np.all(np.isin(matched.vlm_index, plausible.index))


def test_abstention_counts_keep_the_interval_reason_apart(world) -> None:
    counts = abstention_counts(_items(world)[0])
    assert counts["outcomes"] == {"abstain": 1, "implausible": 1, "no_evidence": 1, "plausible": 1}
    assert counts["reasons"] == {"INTERVAL_CROSSES_THRESHOLD": 1}


def _stepped(*steps: int) -> VerifyRun:
    rows = {f"c{i}": [SimpleNamespace(row_step=s)] for i, s in enumerate(steps)}
    return VerifyRun(path=Path("run"), run={}, rows=rows, results={})  # type: ignore[arg-type]


def test_a_run_has_one_time_base_and_mixed_steps_are_refused() -> None:
    assert run_row_step(_stepped(2, 0, 2)) == 2
    assert run_row_step(_stepped(0)) == 0
    with pytest.raises(UsageError, match="one time base per run"):
        run_row_step(_stepped(1, 2))
