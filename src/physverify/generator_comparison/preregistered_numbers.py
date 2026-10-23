"""Every pre-registered generator-comparison number from the arms. Pure: no file is read here.

A verifier difference is always computed on the checks decided in both candidates
(``pairs.on_common_checks``), as the preference pairs and the checkpoint selection are judged.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from physverify.core.models import FrozenModel
from physverify.finetune.pairs import Rescore, drop_reason, on_common_checks
from physverify.finetune.records import CandidateScore
from physverify.finetune.scorers._base import AggregateScore
from physverify.generator_comparison.analysis_inputs import (
    AnalysisInputs,
    ArmSummary,
    JudgeValues,
    Values,
    as_diffs,
)
from physverify.generator_comparison.arms import PromptArms
from physverify.generator_comparison.best_of_n import Key, Pick, pick_best
from physverify.generator_comparison.guards import (
    GuardResult,
    missing_pair_guard,
    sa_guard,
    signal_guards,
    strata,
)
from physverify.generator_comparison.settings import ComparisonSettings
from physverify.generator_comparison.stats import (
    TestResult,
    holm,
    mean_interval,
    paired_test,
)

ABSENT = AggregateScore(None, None, True, "no_checks", 0, 0, 0.0)
GUARDS = ("G1", "G2", "G3", "G4", "G5")
HOLM_TESTS = ("verifier_ft_vs_base", "pc_ft_vs_base", "pc_bo8_vs_random")
SECONDARY_TESTS = (
    "S1_ft_drop_reason",
    "S2_template_clusters",
    "S3_cross_pairs",
    "S4_check_cross_fit",
    "S5_template_mean_wilcoxon",
)
"""The pre-registered sensitivities, in order; S5 is preregistered rule 1.7 (template means,
Wilcoxon)."""


class HolmEntry(FrozenModel):
    test: TestResult
    p_adjusted: float | None


class ComparisonAnalysis(FrozenModel):
    n_prompts: int
    arms: dict[str, ArmSummary]
    fallbacks: dict[str, int]
    primary: TestResult
    external: TestResult | None
    holm: dict[str, HolmEntry]
    ranking_claim: bool
    bo8_vs_base_verifier: TestResult
    guards: dict[str, GuardResult]
    g5_left_out: int
    sensitivities: dict[str, TestResult | None]
    strata: dict[str, TestResult]


def judged(values: JudgeValues, score: CandidateScore) -> float | None:
    return values.get((score.model_tag, score.candidate_id))


def common_diff(a: CandidateScore, b: CandidateScore, rescore: Rescore) -> float | None:
    both = on_common_checks(a, b, rescore)
    return None if both is None else both.score_a - both.score_b


def without(rescore: Rescore, excluded: frozenset[str]) -> Rescore:
    """``rescore`` on every check but ``excluded`` (the judging half of the cross-fit)."""

    def inner(score: CandidateScore, only: frozenset[str]) -> AggregateScore:
        kept = only - excluded
        return rescore(score, kept) if kept else ABSENT

    return inner


def restricted_key(rescore: Rescore, checks: frozenset[str]) -> Key:
    """The score on ``checks`` only (the selecting half of the cross-fit)."""

    def key(score: CandidateScore) -> float | None:
        kept = checks & frozenset(score.decided)
        return rescore(score, kept).score if kept else None

    return key


def _mean(values: Iterable[float | None], minimum: int = 1) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if len(kept) >= minimum else None


def _ft_pairs(
    arm: PromptArms,
    pick: Pick,
    k_ft: int,
    rescore: Rescore,
    eligible: Callable[[CandidateScore], bool],
) -> list[float | None]:
    found = (arm.ft.get(k) for k in range(k_ft))
    return [
        common_diff(ft, pick.score, rescore) if ft is not None and eligible(ft) else None
        for ft in found
    ]


def ft_vs_bo8(
    arms: Sequence[PromptArms],
    picks: Mapping[str, Pick | None],
    k_ft: int,
    rescore: Rescore,
    min_pairs: int,
    eligible: Callable[[CandidateScore], bool] = lambda _: True,
) -> tuple[Values, int]:
    """Per prompt, the mean over valid k of S(ft_k) - S(bo8); prompts with fewer pairs drop."""
    values: Values = {}
    for arm in arms:
        pick = picks.get(arm.prompt_id)
        mean = (
            None
            if pick is None
            else _mean(_ft_pairs(arm, pick, k_ft, rescore, eligible), min_pairs)
        )
        if mean is not None:
            values[arm.prompt_id] = mean
    return values, len(arms) - len(values)


def worst_case(
    arms: Sequence[PromptArms], picks: Mapping[str, Pick | None], k_ft: int, rescore: Rescore
) -> tuple[Values, int]:
    """G5: every missing FT pair imputed with S = 0, so d = -S(bo8); a bo8 without a score
    cannot be imputed and its prompt is left out (counted)."""
    values: Values = {}
    left_out = 0
    for arm in arms:
        pick = picks.get(arm.prompt_id)
        if pick is None or pick.score.score is None:
            left_out += 1
            continue
        pairs = _ft_pairs(arm, pick, k_ft, rescore, lambda _: True)
        imputed = [-pick.score.score if v is None else v for v in pairs]
        values[arm.prompt_id] = float(np.mean(imputed))
    return values, left_out


def judged_vs_bo8(
    arms: Sequence[PromptArms],
    picks: Mapping[str, Pick | None],
    judge: JudgeValues,
    min_pairs: int,
) -> tuple[Values, int]:
    """External d: mean judge(ft_k) - judge(bo8); drops a bo8 without a value or too few FT."""
    values: Values = {}
    for arm in arms:
        pick = picks.get(arm.prompt_id)
        bo8 = None if pick is None else judged(judge, pick.score)
        ft = _mean((judged(judge, s) for s in arm.ft.values()), min_pairs)
        if bo8 is not None and ft is not None:
            values[arm.prompt_id] = ft - bo8
    return values, len(arms) - len(values)


def crn_ft_vs_base(
    arms: Sequence[PromptArms], k_ft: int, rescore: Rescore, min_pairs: int
) -> tuple[Values, int]:
    values: Values = {}
    for arm in arms:
        pairs = [
            common_diff(arm.ft[k], arm.base[k], rescore)
            for k in range(k_ft)
            if k in arm.ft and k in arm.base
        ]
        mean = _mean(pairs, min_pairs)
        if mean is not None:
            values[arm.prompt_id] = mean
    return values, len(arms) - len(values)


def cross_ft_vs_base(
    arms: Sequence[PromptArms], rescore: Rescore, min_pairs: int
) -> tuple[Values, int]:
    """S3: every FT candidate against every base candidate of the prompt."""
    values: Values = {}
    for arm in arms:
        pairs = [common_diff(f, b, rescore) for f in arm.ft.values() for b in arm.base.values()]
        mean = _mean(pairs, min_pairs)
        if mean is not None:
            values[arm.prompt_id] = mean
    return values, len(arms) - len(values)


def judged_against_base(
    arms: Sequence[PromptArms],
    judge: JudgeValues,
    first: Callable[[PromptArms], float | None],
    min_pairs: int,
) -> tuple[Values, int]:
    """``first(arm)`` minus the mean judge value of the prompt's base candidates."""
    values: Values = {}
    for arm in arms:
        base = _mean((judged(judge, s) for s in arm.base.values()), min_pairs)
        head = first(arm)
        if base is not None and head is not None:
            values[arm.prompt_id] = head - base
    return values, len(arms) - len(values)


def _test(found: tuple[Values, int], cfg: ComparisonSettings) -> TestResult:
    values, dropped = found
    return paired_test(as_diffs(values), cfg, dropped)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def summarize(scores: Sequence[CandidateScore], pc: JudgeValues, sa: JudgeValues) -> ArmSummary:
    motions = [s.motion for s in scores if s.motion is not None]
    return ArmSummary(
        n=len(scores),
        n_scored=sum(s.score is not None for s in scores),
        abstention=_rate([s.score is None for s in scores]),
        mean_score=_mean(s.score for s in scores),
        median_motion=float(statistics.median(motions)) if motions else None,
        objects_ok_rate=_rate([s.objects_ok for s in scores if s.objects_ok is not None]),
        mean_pc=_mean(judged(pc, s) for s in scores),
        mean_sa=_mean(judged(sa, s) for s in scores),
    )


def _holm(inputs: AnalysisInputs, picks: Mapping[str, Pick | None]) -> dict[str, HolmEntry]:
    arms, cfg, pc, n = inputs.arms, inputs.cfg, inputs.pc, inputs.cfg.min_ft_pairs_per_prompt

    def ft_pc(arm: PromptArms) -> float | None:
        return _mean((judged(pc, s) for s in arm.ft.values()), n)

    def bo8_pc(arm: PromptArms) -> float | None:
        pick = picks.get(arm.prompt_id)
        return None if pick is None else judged(pc, pick.score)

    tests = {
        HOLM_TESTS[0]: _test(crn_ft_vs_base(arms, inputs.k_ft, inputs.rescore, n), cfg),
        HOLM_TESTS[1]: _test(judged_against_base(arms, pc, ft_pc, n), cfg),
        HOLM_TESTS[2]: _test(judged_against_base(arms, pc, bo8_pc, n), cfg),
    }
    adjusted = holm({name: t.p for name, t in tests.items()})
    return {name: HolmEntry(test=t, p_adjusted=adjusted[name]) for name, t in tests.items()}


def _cross_fit(inputs: AnalysisInputs) -> TestResult | None:
    checks = frozenset(inputs.cfg.cross_fit_checks_a)
    if not checks:
        return None
    key = restricted_key(inputs.rescore, checks)
    picks = {a.prompt_id: pick_best(a.base, inputs.rules, key) for a in inputs.arms}
    judged = without(inputs.rescore, checks)
    found = ft_vs_bo8(inputs.arms, picks, inputs.k_ft, judged, inputs.cfg.min_ft_pairs_per_prompt)
    return _test(found, inputs.cfg)


def _bo8_verifier(arms: Sequence[PromptArms], picks: Mapping[str, Pick | None]) -> Values:
    values: Values = {}
    for arm in arms:
        pick = picks.get(arm.prompt_id)
        base = _mean(s.score for s in arm.base.values())
        if pick is not None and pick.value is not None and base is not None:
            values[arm.prompt_id] = pick.value - base
    return values


def analyse(inputs: AnalysisInputs) -> ComparisonAnalysis:
    arms, cfg, n = inputs.arms, inputs.cfg, inputs.cfg.min_ft_pairs_per_prompt
    picks = {a.prompt_id: pick_best(a.base, inputs.rules, lambda s: s.score) for a in arms}
    chosen = [p for p in picks.values() if p is not None]
    primary_found = ft_vs_bo8(arms, picks, inputs.k_ft, inputs.rescore, n)
    primary = _test(primary_found, cfg)
    external = _test(judged_vs_bo8(arms, picks, inputs.pc, n), cfg) if inputs.pc else None
    summaries = _summaries(inputs, chosen)
    worst_values, left_out = worst_case(arms, picks, inputs.k_ft, inputs.rescore)
    holm_entries = _holm(inputs, picks)
    return ComparisonAnalysis(
        n_prompts=len(arms),
        arms=summaries,
        fallbacks=dict(Counter(p.fallback or "none" for p in chosen)),
        primary=primary,
        external=external,
        holm=holm_entries,
        ranking_claim=_ranking_claim(inputs, holm_entries[HOLM_TESTS[2]]),
        bo8_vs_base_verifier=paired_test(as_diffs(_bo8_verifier(arms, picks)), cfg),
        guards=_guards(inputs, picks, summaries, primary, worst_values),
        g5_left_out=left_out if inputs.has_ft else 0,
        sensitivities=_sensitivities(inputs, picks, primary_found),
        strata=strata(primary_found[0], arms, cfg),
    )


def _summaries(inputs: AnalysisInputs, chosen: Sequence[Pick]) -> dict[str, ArmSummary]:
    arms = inputs.arms
    return {
        "base": summarize([s for a in arms for s in a.base.values()], inputs.pc, inputs.sa),
        "bo8": summarize([p.score for p in chosen], inputs.pc, inputs.sa),
        "ft": summarize([s for a in arms for s in a.ft.values()], inputs.pc, inputs.sa),
    }


def _guards(
    inputs: AnalysisInputs,
    picks: Mapping[str, Pick | None],
    summaries: Mapping[str, ArmSummary],
    primary: TestResult,
    worst_values: Values,
) -> dict[str, GuardResult]:
    if not inputs.has_ft:
        return {g: GuardResult(state="not_evaluated", detail="no FT model") for g in GUARDS}
    cfg, n = inputs.cfg, inputs.cfg.min_ft_pairs_per_prompt
    return {
        **signal_guards(summaries["ft"], summaries["base"], inputs),
        "G4": sa_guard(judged_vs_bo8(inputs.arms, picks, inputs.sa, n), inputs),
        "G5": missing_pair_guard(primary, paired_test(as_diffs(worst_values), cfg), cfg),
    }


def _ranking_claim(inputs: AnalysisInputs, ranking: HolmEntry) -> bool:
    return (
        inputs.external_status == "ok"
        and ranking.p_adjusted is not None
        and ranking.p_adjusted < inputs.cfg.alpha
        and (ranking.test.hl or 0) > 0
    )


def _sensitivities(
    inputs: AnalysisInputs, picks: Mapping[str, Pick | None], primary_found: tuple[Values, int]
) -> dict[str, TestResult | None]:
    arms, cfg, n = inputs.arms, inputs.cfg, inputs.cfg.min_ft_pairs_per_prompt
    templates = {a.prompt_id: a.prompt.template_id for a in arms}
    s1 = ft_vs_bo8(
        arms, picks, inputs.k_ft, inputs.rescore, n, lambda s: drop_reason(s, inputs.rules) is None
    )
    return dict(
        zip(
            SECONDARY_TESTS,
            (
                _test(s1, cfg),
                _clustered(primary_found, templates, cfg),
                _test(cross_ft_vs_base(arms, inputs.rescore, n), cfg),
                _cross_fit(inputs),
                _test(template_means(primary_found, templates), cfg),
            ),
            strict=True,
        )
    )


def template_means(found: tuple[Values, int], templates: Mapping[str, str]) -> tuple[Values, int]:
    """One value per template, the mean of its prompts' differences (the unit of S5)."""
    values, dropped = found
    groups: dict[str, list[float]] = {}
    for prompt_id, value in values.items():
        groups.setdefault(templates[prompt_id], []).append(value)
    return {t: mean for t, v in groups.items() if (mean := _mean(v)) is not None}, dropped


def _clustered(
    found: tuple[Values, int], templates: Mapping[str, str], cfg: ComparisonSettings
) -> TestResult:
    values, dropped = found
    base = paired_test(as_diffs(values), cfg, dropped)
    lo, hi = mean_interval(as_diffs(values, templates), cfg)
    return base.model_copy(update={"ci_lo": lo, "ci_hi": hi})
