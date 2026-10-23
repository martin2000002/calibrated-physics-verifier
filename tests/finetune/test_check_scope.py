"""``[check_scope]`` in the generator comparison: every scorer ``build_scorer`` returns leaves the
excluded checks out."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from cli_runner import CliRunner
from wiring import build_scorer, checks

from physverify.config.models import Settings
from physverify.core.errors import ConfigError, UsageError
from physverify.finetune.scorer_scope import (
    EXCLUDED_TAG,
    ScopedScorer,
    family_targets,
    scorer_exclusion,
)
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.settings import load_finetune
from physverify.schema.checks import CheckResult, Verdict

Result = Callable[..., CheckResult]
TIME_DEPENDENT = ("R4_jump", "R5b")


def _results(make_result: Result) -> list[CheckResult]:
    """R1 passes; both time-dependent checks fail hard."""
    return [
        make_result("R1", Verdict.PASS, 0.0, 1.0),
        make_result("R5b", Verdict.FAIL, 50.0, 1.0),
        make_result("R4_jump", Verdict.FAIL, 50.0, 1.0),
    ]


def test_build_scorer_wraps_the_provisional_scorer(settings: Settings, make_result: Result) -> None:
    scorer = build_scorer("provisional", settings)
    assert isinstance(scorer, ScopedScorer)
    assert (scorer.name, scorer.version, scorer.calibrated) == ("provisional", "2", False)
    inner = ProvisionalScorer(settings, {})
    assert scorer.thresholds() == f"{inner.thresholds()}{EXCLUDED_TAG}R4_jump,R5b"
    score = scorer.score(_results(make_result))
    assert score.decided == frozenset({"R1"}) and score.n_failed == 0
    assert set(score.per_check) == {"R1"}
    assert inner.score(_results(make_result)).n_failed == 2


def test_only_never_lets_an_excluded_check_in(settings: Settings, make_result: Result) -> None:
    scorer = build_scorer("provisional", settings)
    score = scorer.score(_results(make_result), only=frozenset({"R1", "R5b"}))
    assert score.decided == frozenset({"R1"})


def test_an_empty_only_is_an_abstention_not_every_check(
    settings: Settings, make_result: Result
) -> None:
    scorer = build_scorer("provisional", settings)
    score = scorer.score(_results(make_result), only=frozenset())
    assert score.abstained and score.score is None and score.n_decided == 0


def test_the_stamp_changes_with_exclude_time_dependent(
    make_settings: Callable[..., Settings], make_result: Result
) -> None:
    excluded = build_scorer("provisional", make_settings(profile="precise"))
    kept = build_scorer(
        "provisional", make_settings("finetune.exclude_time_dependent=false", profile="precise")
    )
    assert not isinstance(kept, ScopedScorer)
    assert kept.thresholds() != excluded.thresholds()
    assert EXCLUDED_TAG not in kept.thresholds()
    assert kept.score(_results(make_result)).n_failed == 2


def test_exclude_checks_join_the_stamp(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings('check_scope.exclude_checks=["R1"]', profile="precise")
    assert build_scorer("provisional", settings).thresholds().endswith("@excl:R1,R4_jump,R5b")


def test_an_unknown_check_id_is_a_configuration_error(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings('check_scope.exclude_checks=["R4jump"]', profile="precise")
    with pytest.raises(ConfigError, match="R4jump"):
        build_scorer("provisional", settings)


def test_fall_keeps_r5a_by_default(settings: Settings) -> None:
    cfg = load_finetune(settings)
    targets = family_targets(cfg, scorer_exclusion(settings, cfg, checks()))
    assert targets["fall"] == ["R5a"]
    assert all(targets.values())


def test_a_family_without_target_checks_is_refused(
    make_settings: Callable[..., Settings],
) -> None:
    without = 'check_scope.exclude_checks=["R5a"]'
    with pytest.raises(UsageError, match=r"\['fall'\]"):
        build_scorer("provisional", make_settings(without, profile="precise"))
    keep = make_settings(without, "finetune.on_family_without_targets=keep", profile="precise")
    assert build_scorer("provisional", keep).thresholds().endswith("@excl:R4_jump,R5a,R5b")


def test_the_cli_exits_2_on_a_family_without_targets(cli: CliRunner, tmp_path: Path) -> None:
    env = {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path)}
    options = ("--set", 'check_scope.exclude_checks=["R5a"]')
    result = cli(*options, "finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 2 and "keep no target check" in result.err
    unknown = cli("--set", 'check_scope.exclude_checks=["R99"]', "finetune", "pairs", env=env)
    assert unknown.code == 2 and "R99" in unknown.err
