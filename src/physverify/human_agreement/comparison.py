"""The comparison itself: decisions, items, every policy and group, secondary cuts, breakdowns.
Pure: inputs are injected, outputs are plain data for ``report``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from physverify.datasets._base import LabelledVideo
from physverify.human_agreement.breakdown import AGGREGATE, by_version, per_check, per_group
from physverify.human_agreement.clip_decision import CALIBRATED_NAME, ClipDecisionRule
from physverify.human_agreement.comparison_items import (
    ComparisonItem,
    JoinCounts,
    build_items,
    decide_all,
    implausible,
)
from physverify.human_agreement.coverage import InputGaps, generator_key
from physverify.human_agreement.error_sample import sample_errors
from physverify.human_agreement.metrics import (
    CalibrationState,
    ComparisonMetrics,
    Coverage,
    PrimaryVerdict,
    RunInfo,
    Sensitivity,
    ViewBlock,
)
from physverify.human_agreement.policy_metrics import (
    BootstrapPlan,
    best_cut,
    cut_metrics,
    evaluate_view,
    h2_verdict,
    view_design_effect,
)
from physverify.human_agreement.prereg import FreezeAudit
from physverify.human_agreement.settings import AgreementSettings
from physverify.human_agreement.undecided_policy import (
    POLICIES,
    abstention_counts,
    decided,
    policy_view,
    risk_coverage,
)
from physverify.human_agreement.verify_run_reader import VerifyRun


@dataclass(frozen=True)
class RuleScope:
    """``rule`` already leaves out ``[check_scope].exclude_checks``; ``primary_excluded`` are the
    time-dependent checks when the primary also leaves them out, and ``time_dependent`` the ones
    the sensitivity block moves in or out."""

    time_dependent: frozenset[str] = frozenset()
    primary_excluded: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ComparisonInputs:
    rows: Sequence[LabelledVideo]
    run: VerifyRun
    rule: ClipDecisionRule
    agreement: AgreementSettings
    cut: float | None
    dev_tuned_cut: float | None
    dev_majority: int | None
    preregistered: bool
    audit: FreezeAudit | None = None
    scope: RuleScope = RuleScope()
    gate: str | None = None
    """Why the calibrated stratum was not usable (the provisional rule ran instead)."""
    calibration: dict[str, Any] | None = None
    dev_design_effect: float | None = None
    gaps: InputGaps | None = None
    """Rows without an object query or a video, reported apart from the abstentions."""


@dataclass(frozen=True)
class Comparison:
    items: tuple[ComparisonItem, ...]
    counts: JoinCounts
    metrics: ComparisonMetrics
    per_check: list[dict[str, object]]
    per_generator: list[dict[str, object]]
    errors: list[dict[str, object]]
    audit: FreezeAudit | None = None


def _items(
    inputs: ComparisonInputs, excluded: frozenset[str]
) -> tuple[list[ComparisonItem], JoinCounts]:
    rule = inputs.rule.excluding(excluded)
    return build_items(inputs.rows, decide_all(rule, inputs.run), inputs.run, inputs.cut)


def _groups(
    inputs: ComparisonInputs, items: Sequence[ComparisonItem], plan: BootstrapPlan
) -> dict[str, dict[str, ViewBlock]]:
    test = (inputs.agreement.alpha, inputs.agreement.power)
    return {
        group: {
            policy: evaluate_view(
                members,
                policy_view(members, policy),
                inputs.dev_majority,
                plan,
                test,
                _frozen_design_effect(inputs, group, policy),
            )
            for policy in POLICIES
        }
        for group, members in by_version(items).items()
    }


def _frozen_design_effect(inputs: ComparisonInputs, group: str, policy: str) -> float | None:
    """The dev design effect was estimated on the aggregate primary view only; every other
    view estimates its own on the compared data."""
    primary = group == AGGREGATE and policy == inputs.agreement.primary_policy
    return inputs.dev_design_effect if primary else None


def _secondary(
    rule: ClipDecisionRule, inputs: ComparisonInputs, items: Sequence[ComparisonItem]
) -> dict[str, dict[str, dict[str, float]]]:
    cuts = {"dev_tuned": inputs.dev_tuned_cut, "kubric_threshold": rule.kubric_threshold}
    return {
        name: {g: cut_metrics(m, range(len(m)), cut) for g, m in by_version(items).items()}
        for name, cut in cuts.items()
        if cut is not None
    }


def _plan(agreement: AgreementSettings, replicas: int) -> BootstrapPlan:
    return BootstrapPlan(
        replicas, agreement.bootstrap_seed, agreement.ci, agreement.permutation_exact_max
    )


def _sensitivity(inputs: ComparisonInputs) -> Sensitivity:
    """The primary policy with the time-dependent checks moved: out of the primary by default,
    back in when the primary already leaves them out."""
    scope = inputs.scope
    in_primary = not scope.primary_excluded
    moved, _ = _items(inputs, scope.time_dependent if in_primary else frozenset())
    plan = _plan(inputs.agreement, 0)
    test = (inputs.agreement.alpha, inputs.agreement.power)
    policy = inputs.agreement.primary_policy
    return Sensitivity(
        time_dependent=sorted(scope.time_dependent),
        time_dependent_in_primary=in_primary,
        excluded=sorted(scope.time_dependent) if in_primary else [],
        policy=policy,
        groups={
            g: evaluate_view(m, policy_view(m, policy), inputs.dev_majority, plan, test)
            for g, m in by_version(moved).items()
        },
    )


def _coverage(items: Sequence[ComparisonItem]) -> Coverage:
    """Fraction of compared clips the verifier decided (plausible or implausible)."""
    n_decided = len(decided(items, range(len(items))))
    return Coverage(
        n=float(len(items)),
        decided=float(n_decided),
        coverage=n_decided / len(items) if items else math.nan,
    )


def _verdict(inputs: ComparisonInputs, primary: ViewBlock) -> PrimaryVerdict:
    audit = inputs.audit
    return h2_verdict(
        primary,
        inputs.rule.name == CALIBRATED_NAME and inputs.preregistered,
        inputs.agreement.alpha,
        post_hoc=audit is not None and audit.post_hoc,
        gate=inputs.gate,
    )


@dataclass(frozen=True)
class _Compared:
    items: list[ComparisonItem]
    counts: JoinCounts
    groups: dict[str, dict[str, ViewBlock]]
    per_check: list[dict[str, object]]
    per_generator: list[dict[str, object]]


def _compared(inputs: ComparisonInputs) -> _Compared:
    """The primary items, every policy view of every group, and the two breakdowns."""
    primary_excluded = inputs.scope.primary_excluded
    policy = inputs.agreement.primary_policy
    items, counts = _items(inputs, primary_excluded)

    def redecide(excluded: frozenset[str]) -> list[ComparisonItem]:
        return _items(inputs, primary_excluded | excluded)[0]

    return _Compared(
        items=items,
        counts=counts,
        groups=_groups(inputs, items, _plan(inputs.agreement, inputs.agreement.bootstrap_n)),
        per_check=per_check(items, redecide, inputs.run.check_ids, policy),
        per_generator=per_group(items, lambda it: generator_key(it.row), policy),
    )


def _metrics(inputs: ComparisonInputs, compared: _Compared) -> ComparisonMetrics:
    items, groups = compared.items, compared.groups
    run = inputs.run
    return ComparisonMetrics(
        rule=inputs.rule.label,
        calibration=CalibrationState(gate=inputs.gate, stratum=inputs.calibration),
        primary_excluded=sorted(inputs.scope.primary_excluded),
        preregistered=inputs.preregistered,
        preregistration=inputs.audit,
        run=RunInfo(path=str(run.path), profile=run.profile, device=run.device),
        primary_policy=inputs.agreement.primary_policy,
        counts=compared.counts,
        abstention=abstention_counts(items),
        input_gaps=inputs.gaps,
        coverage={g: _coverage(m) for g, m in by_version(items).items()},
        h2_group=AGGREGATE,
        descriptive_groups=[g for g in groups if g != AGGREGATE],
        groups=groups,
        per_check=compared.per_check,
        per_generator=compared.per_generator,
        secondary_cuts=_secondary(inputs.rule, inputs, items),
        sensitivity_time_dependent=_sensitivity(inputs),
        risk_coverage_vlm={g: risk_coverage(m) for g, m in by_version(items).items()},
        h2_verdict=_verdict(inputs, groups[AGGREGATE][inputs.agreement.primary_policy]),
    )


def compare(inputs: ComparisonInputs) -> Comparison:
    compared = _compared(inputs)
    agreement = inputs.agreement
    return Comparison(
        items=tuple(compared.items),
        counts=compared.counts,
        metrics=_metrics(inputs, compared),
        per_check=compared.per_check,
        per_generator=compared.per_generator,
        errors=sample_errors(
            compared.items,
            agreement.error_sample_per_class,
            agreement.error_seed,
            inputs.rule.tau,
            inputs.run.results,
            agreement.primary_policy,
        ),
        audit=inputs.audit,
    )


@dataclass(frozen=True)
class DevFreeze:
    n: int
    majority: int | None
    cut: float | None
    tuned_cut: float | None
    design: dict[str, float | int] | None = None
    """Estimated design effect of the primary policy's discordance on dev (frozen, item 1.3)."""


def freeze_dev(
    rows: Sequence[LabelledVideo], rule: ClipDecisionRule, run: VerifyRun, policy: str
) -> DevFreeze:
    """On h2_dev: the majority class, the cut of maximum balanced accuracy on the score
    (primary for the calibrated rule, secondary ``dev_tuned`` for the provisional one) and the
    design effect of ``policy`` under that cut."""
    decisions = decide_all(rule, run)
    labelled = [
        (label, decisions[r.clip_id].score)
        for r in rows
        if r.clip_id in decisions and (label := implausible(r.human_raw, r.human_cut)) is not None
    ]
    if not labelled:
        return DevFreeze(0, None, None, None)
    y = np.asarray([int(label) for label, _ in labelled], dtype=np.int_)
    score = np.asarray([s for _, s in labelled], dtype=np.float64)
    majority = int(np.mean(y) > 0.5)
    cut = best_cut(y, score)
    calibrated = rule.name == CALIBRATED_NAME
    primary_cut = cut if calibrated else None
    design = None
    if primary_cut is not None or not calibrated:
        items, _ = build_items(rows, decisions, run, primary_cut)
        design = view_design_effect(items, policy_view(items, policy))
    return DevFreeze(len(labelled), majority, primary_cut, None if calibrated else cut, design)
