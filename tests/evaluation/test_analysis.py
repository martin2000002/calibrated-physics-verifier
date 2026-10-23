from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from physverify.config.models import Settings
from physverify.finetune.records import UNSEEN_TAG, CandidateScore, PromptRecord
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.settings import load_finetune
from physverify.generator_comparison.analysis_inputs import AnalysisInputs
from physverify.generator_comparison.arms import PromptArms, analysis_prompts, build_arms
from physverify.generator_comparison.best_of_n import pick_best
from physverify.generator_comparison.guards import missing_pair_guard
from physverify.generator_comparison.preregistered_numbers import (
    SECONDARY_TESTS,
    analyse,
    ft_vs_bo8,
    judged_vs_bo8,
    template_means,
    worst_case,
)
from physverify.generator_comparison.settings import H3Settings
from physverify.generator_comparison.stats import TestResult

Score = Callable[..., CandidateScore]


def stored(score: CandidateScore, only: frozenset[str]) -> AggregateScore:
    """The stored score whatever the common checks (enough for the arithmetic here)."""
    return AggregateScore(score.score, None, score.score is None, None, len(only), 0, 1.0, only)


def arm(
    p: PromptRecord,
    make_score: Score,
    base: Mapping[int, float | None],
    ft: Mapping[int, float | None],
) -> PromptArms:
    pid = p.prompt_id
    return PromptArms(
        prompt=p,
        base={k: make_score(f"{pid}__k{k:02d}", v) for k, v in base.items()},
        ft={k: make_score(f"{pid}__k{k:02d}", v, model_tag="lora") for k, v in ft.items()},
    )


def inputs(
    settings: Settings, cfg: H3Settings, arms: list[PromptArms], **update: Any
) -> AnalysisInputs:
    finetune = load_finetune(settings)
    fields: dict[str, Any] = {
        "arms": arms,
        "has_ft": True,
        "k_ft": 4,
        "rescore": stored,
        "rules": finetune.pairs,
        "select": finetune.select,
        "cfg": cfg,
        "pc": {},
        "sa": {},
        "external_status": "not_run",
    }
    return AnalysisInputs(**{**fields, **update})


@pytest.fixture
def better(make_prompt: Callable[..., PromptRecord], make_score: Score) -> list[PromptArms]:
    """Eight prompts where every fine-tuned sample beats the best base one."""
    found = []
    for i in range(8):
        p = make_prompt(f"e{i}", "eval", i, [UNSEEN_TAG] if i % 2 else [])
        base = {k: 0.3 + 0.02 * k for k in range(8)}
        ft = {k: 0.6 + 0.01 * i + 0.01 * k for k in range(4)}
        found.append(arm(p, make_score, base, ft))
    return found


def test_build_arms_keeps_only_the_eval_prompts_and_the_paired_seeds(
    make_prompt: Callable[..., PromptRecord], make_score: Score
) -> None:
    prompts = [make_prompt("e0", "eval", 0), make_prompt("x", "eval", 1, ["demo"])]
    prompts.append(make_prompt("d0", "dev", 0))
    analysed = analysis_prompts(prompts, ["demo"])
    assert [p.prompt_id for p in analysed] == ["e0"]
    base = [make_score(f"e0__k{k:02d}", 0.5) for k in range(9)]
    arms = build_arms(analysed, base, [make_score("e0__k05", 0.5)], 8, 4)
    assert sorted(arms[0].base) == list(range(8))
    assert arms[0].ft == {}


def test_ft_vs_bo8_averages_pairs_and_drops_thin_prompts(
    make_prompt: Callable[..., PromptRecord], make_score: Score, h3_settings: Settings
) -> None:
    rules = load_finetune(h3_settings).pairs
    full = arm(make_prompt("a", "eval", 0), make_score, {0: 0.5, 1: 0.7}, {0: 0.8, 1: 0.6})
    thin = arm(make_prompt("b", "eval", 1), make_score, {0: 0.5}, {0: 0.9})
    picks = {a.prompt_id: pick_best(a.base, rules) for a in (full, thin)}
    values, dropped = ft_vs_bo8([full, thin], picks, 4, stored, min_pairs=2)
    assert values == {"a": pytest.approx(0.0)}
    assert dropped == 1


def test_worst_case_imputes_zero_and_leaves_out_unscored_bo8(
    make_prompt: Callable[..., PromptRecord], make_score: Score, h3_settings: Settings
) -> None:
    rules = load_finetune(h3_settings).pairs
    missing = arm(make_prompt("a", "eval", 0), make_score, {0: 0.6}, {0: 0.8, 1: None})
    unscored = arm(make_prompt("b", "eval", 1), make_score, {0: None}, {0: 0.8})
    picks = {a.prompt_id: pick_best(a.base, rules) for a in (missing, unscored)}
    values, left_out = worst_case([missing, unscored], picks, 4, stored)
    # Four planned FT samples: k0 pairs (0.8 - 0.6), k1 (unscored) and k2, k3 (missing) get -0.6.
    assert values == {"a": pytest.approx((0.2 - 3 * 0.6) / 4)}
    assert left_out == 1


def test_g5_runs_only_after_a_supported_primary(h3_cfg: H3Settings) -> None:
    weak = TestResult(n=8, p=0.5, hl=0.1, mean=0.1, ci_lo=None, ci_hi=None, method="m")
    strong = weak.model_copy(update={"p": 0.001})
    worst = weak.model_copy(update={"hl": -0.1})
    assert missing_pair_guard(weak, worst, h3_cfg).state == "not_evaluated"
    assert missing_pair_guard(strong, worst, h3_cfg).state == "fail"
    assert missing_pair_guard(strong, weak, h3_cfg).state == "ok"


def test_judged_vs_bo8_needs_the_bo8_value(
    make_prompt: Callable[..., PromptRecord], make_score: Score, h3_settings: Settings
) -> None:
    rules = load_finetune(h3_settings).pairs
    a = arm(make_prompt("a", "eval", 0), make_score, {0: 0.5}, {0: 0.8, 1: 0.7})
    picks = {"a": pick_best(a.base, rules)}
    bo8, ft = a.base[0], a.ft[1]
    judge = {(bo8.model_tag, bo8.candidate_id): 3.0, (ft.model_tag, ft.candidate_id): 4.0}
    values, dropped = judged_vs_bo8([a], picks, judge, min_pairs=2)
    assert (values, dropped) == ({}, 1)
    values, _ = judged_vs_bo8([a], picks, judge, min_pairs=1)
    assert values == {"a": pytest.approx(1.0)}


def test_a_clear_improvement_is_supported_by_the_verifier(
    better: list[PromptArms], h3_settings: Settings, h3_cfg: H3Settings
) -> None:
    result = analyse(inputs(h3_settings, h3_cfg, better))
    assert result.n_prompts == 8
    assert result.primary.n == 8 and result.primary.hl is not None and result.primary.hl > 0
    assert result.primary.p is not None and result.primary.p < 0.05
    assert result.external is None
    assert result.guards["G4"].state == "not_evaluated"
    assert result.guards["G5"].state == "ok"
    assert {g: result.guards[g].state for g in ("G1", "G2", "G3")} == dict.fromkeys(
        ("G1", "G2", "G3"), "ok"
    )
    assert not result.ranking_claim
    assert result.fallbacks == {"none": 8}
    assert set(result.strata) == {"seen", "unseen", "family:f0", "family:f1"}
    assert result.sensitivities["S4_check_cross_fit"] is None
    assert result.arms["bo8"].n == 8 and result.arms["ft"].n == 32
    assert tuple(result.sensitivities) == SECONDARY_TESTS
    s5 = result.sensitivities["S5_template_mean_wilcoxon"]
    assert s5 is not None and s5.n == 4 and s5.hl is not None and s5.hl > 0


def test_template_means_average_the_prompts_of_a_template() -> None:
    values = {"a": 1.0, "b": 3.0, "c": 5.0}
    templates = {"a": "t0", "b": "t0", "c": "t1"}
    assert template_means((values, 2), templates) == ({"t0": 2.0, "t1": 5.0}, 2)


def test_without_a_fine_tuned_model_no_guard_is_evaluated(
    better: list[PromptArms], h3_settings: Settings, h3_cfg: H3Settings
) -> None:
    result = analyse(inputs(h3_settings, h3_cfg, better, has_ft=False))
    assert {g.state for g in result.guards.values()} == {"not_evaluated"}
    assert result.g5_left_out == 0


def test_the_external_judge_enters_when_it_has_values(
    better: list[PromptArms], h3_settings: Settings, h3_cfg: H3Settings
) -> None:
    pc = {(s.model_tag, s.candidate_id): 4.0 for a in better for s in a.ft.values()}
    pc |= {(s.model_tag, s.candidate_id): 3.0 for a in better for s in a.base.values()}
    result = analyse(inputs(h3_settings, h3_cfg, better, pc=pc, sa=pc, external_status="ok"))
    assert result.external is not None
    assert result.guards["G4"].state in {"ok", "not_evaluated"}
    assert result.holm["pc_ft_vs_base"].test.n == 8
    # Base and fine-tuned candidates of one seed share an id; the judge keeps them apart.
    assert result.arms["ft"].mean_pc == 4.0 and result.arms["base"].mean_pc == 3.0
