"""The exclusion decision of rules 1.1, 1.2 and 1.5 from the consumed stratum and the R2 runs.

Pure: the CLI loads the artifact, the evaluation and the two R2 runs, this module decides."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from physverify.calibration.artifact import DatasetRef, EvaluationStratum, RealFpr
from physverify.calibration.exclusion_detector_arms import r2_rule
from physverify.calibration.exclusion_inputs import (
    ConsumedStratum,
    DecisionInputs,
    rule_evidence,
)
from physverify.calibration.exclusion_rules import (
    KEPT,
    R2_RULE_ID,
    R5A_RULE_ID,
    REAL_FPR_RULE_ID,
    ExclusionRules,
    R5aRule,
    RealRule,
)
from physverify.stats.intervals import clopper_pearson
from physverify.verification.exclusion_record import (
    EvidenceValue,
    ExclusionInputError,
    ExclusionRecord,
    InputHashes,
    RuleEvidence,
)

# --- 1.1 ----------------------------------------------------------------------------------------


def r5a_rule(inputs: DecisionInputs, rules: ExclusionRules) -> tuple[RuleEvidence, ...]:
    return tuple(_r5a(inputs.consumed, c, rules.r5a) for c in sorted(rules.r5a.check_ids))


def _r5a(consumed: ConsumedStratum, check_id: str, cfg: R5aRule) -> RuleEvidence:
    check = consumed.stratum.check(check_id)
    weight = consumed.stratum.weight(check_id)
    inputs: dict[str, EvidenceValue] = {"weight": weight, "stratum_problem": consumed.problem}
    tau_b_low = None
    if check is not None:
        tau_b_low = check.monotonicity.ci.low
        inputs |= {
            "status": check.status,
            "informedness": check.weight.informedness,
            "informedness_low": check.weight.informedness_ci.low,
            "kendall_tau_b": check.monotonicity.kendall_tau_b,
            "kendall_tau_b_low": tau_b_low,
            "n_pairs": check.monotonicity.n_pairs,
            "n_scenes": check.monotonicity.n_scenes,
        }
    if consumed.problem is not None:
        reason = "stratum_unusable"
    elif check is None:
        reason = "not_in_stratum"
    elif not weight > cfg.min_weight:
        reason = "weight_not_positive"
    elif tau_b_low is None or not tau_b_low > cfg.min_tau_b_low:
        reason = "kendall_tau_b_low_not_positive"
    else:
        reason = KEPT
    return rule_evidence(R5A_RULE_ID, check_id, reason, inputs, decided=True)


# --- 1.5 ----------------------------------------------------------------------------------------


def real_fpr_rule(inputs: DecisionInputs, rules: ExclusionRules) -> tuple[RuleEvidence, ...]:
    stratum = _evaluated_stratum(inputs, rules)
    by_check: dict[str, dict[str, RealFpr]] = {}
    for entry in stratum.real:
        by_check.setdefault(entry.check_id, {})[entry.dataset] = entry
    sets = {ref.name: ref for ref in inputs.evaluation.real_sets}
    return tuple(
        _real(check_id, by_check[check_id], sets, rules.real) for check_id in sorted(by_check)
    )


def _evaluated_stratum(inputs: DecisionInputs, rules: ExclusionRules) -> EvaluationStratum:
    """The evaluation's consumed stratum, after checking it scored this artifact on an allowed
    real clip set."""
    report = inputs.evaluation
    if report.artifact_sha256 != inputs.calibration.sha256:
        raise ExclusionInputError(
            f"the evaluation scored artifact {report.artifact_sha256[:12]}, not the given "
            f"{inputs.calibration.sha256[:12]}: run `calibrate evaluate` on it"
        )
    sizes = {ref.name: ref.n_clips for ref in report.real_sets}
    if len(sizes) != len(report.real_sets) or sizes not in list(rules.real.allowed_sets):
        raise ExclusionInputError(
            f"the evaluation's real clip sets {sizes or 'are not recorded'} are none of the "
            f"preregistered {list(rules.real.allowed_sets)}"
        )
    for stratum in report.strata:
        if stratum.level is rules.consumed_level and stratum.profile == rules.consumed_profile:
            return _require_real_scored(stratum, rules.real)
    raise ExclusionInputError(
        f"the evaluation has no {rules.consumed_level.value}/{rules.consumed_profile} stratum"
    )


def _require_real_scored(stratum: EvaluationStratum, cfg: RealRule) -> EvaluationStratum:
    """1.5 takes the worst of every preregistered real dataset: one left unscored is refused, not
    silently dropped from the worst case."""
    scored = {entry.dataset for entry in stratum.real}
    missing = sorted(set(cfg.datasets) - scored)
    if missing:
        raise ExclusionInputError(
            f"the evaluation scored no real FPR on {missing}: rerun `calibrate evaluate` with them"
        )
    return stratum


def _real(
    check_id: str,
    entries: Mapping[str, RealFpr],
    sets: Mapping[str, DatasetRef],
    cfg: RealRule,
) -> RuleEvidence:
    evidence: dict[str, EvidenceValue] = {"ci": cfg.ci, "sided": cfg.sided}
    lows: list[float] = []
    for name in cfg.datasets:
        ref = sets[name]
        evidence |= {f"{name}.n_clips": ref.n_clips, f"{name}.clip_ids_hash": ref.clip_ids_hash}
        entry = entries.get(name)
        if entry is None or entry.fpr.n <= 0:
            evidence[f"{name}.status"] = "no_rows"
            continue
        exact = clopper_pearson(entry.fpr.k, entry.fpr.n, cfg.ci)
        evidence |= {
            f"{name}.status": "ok",
            f"{name}.k": entry.fpr.k,
            f"{name}.n": entry.fpr.n,
            f"{name}.cp_low": exact.low,
            f"{name}.cp_high": exact.high,
            f"{name}.clustered_low": entry.fpr.ci.low,
            f"{name}.ci_method": entry.fpr.ci_method,
            f"{name}.n_scenes": entry.fpr.n_scenes,
        }
        if exact.low is not None:
            lows.append(exact.low)
    if not lows:
        return rule_evidence(REAL_FPR_RULE_ID, check_id, "no_rows", evidence, decided=False)
    worst = max(lows)
    evidence["worst_cp_low"] = worst
    reason = "real_fpr_high" if worst > cfg.max_fpr_lower else KEPT
    return rule_evidence(REAL_FPR_RULE_ID, check_id, reason, evidence, decided=True)


# --- decision -----------------------------------------------------------------------------------

Apply = Callable[[DecisionInputs, ExclusionRules], tuple[RuleEvidence, ...]]
RULES: Mapping[str, Apply] = {
    R5A_RULE_ID: r5a_rule,
    R2_RULE_ID: r2_rule,
    REAL_FPR_RULE_ID: real_fpr_rule,
}
"""The preregistered rules by id; a new rule is one function with this signature."""


def input_hashes(inputs: DecisionInputs, rules: ExclusionRules) -> InputHashes:
    return InputHashes(
        artifact_sha256=inputs.calibration.sha256,
        artifact_run_id=inputs.calibration.run_id,
        evaluation_sha256=inputs.evaluation_sha256,
        evaluation_run_id=inputs.evaluation.run_id,
        r2_base_run=inputs.arms.base.run_id,
        r2_base_report_sha256=inputs.arms.base.report_sha256,
        r2_tiny_run=inputs.arms.tiny.run_id,
        r2_tiny_report_sha256=inputs.arms.tiny.report_sha256,
        rules_sha256=rules.sha256(),
    )


def decide(
    inputs: DecisionInputs,
    rules: ExclusionRules,
    families_without_target: Callable[[frozenset[str]], Iterable[str]],
    *,
    replaces_sha256: str | None = None,
) -> ExclusionRecord:
    """Applies ``rules.rules`` in order; ``families_without_target`` maps the excluded ids to
    the H3 families left without a target check (reported, it does not change the decision)."""
    evidence = tuple(item for name in rules.rules for item in RULES[name](inputs, rules))
    excluded = frozenset(item.check_id for item in evidence if item.excluded)
    record = ExclusionRecord(
        rules_version=rules.version,
        consumed_level=rules.consumed_level.value,
        consumed_profile=rules.consumed_profile,
        checks_hash=inputs.checks_hash,
        inputs=input_hashes(inputs, rules),
        evidence=evidence,
        exclude_checks=tuple(sorted(excluded)),
        h3_families_without_target=tuple(sorted(families_without_target(excluded))),
        replaces_sha256=replaces_sha256,
    )
    return record.sealed()


def record_markdown(record: ExclusionRecord) -> str:
    """The human view of a record; the JSON is the authority."""
    lines = [
        "# Exclusion decision (ADR 0015 rules 1.1, 1.2, 1.5)",
        "",
        f"- record sha256: `{record.sha256}`",
        f"- consumed stratum: {record.consumed_level}/{record.consumed_profile}",
        f"- artifact sha256: `{record.inputs.artifact_sha256}`",
        f"- evaluation sha256: `{record.inputs.evaluation_sha256}`",
        f"- R2 runs: base `{record.inputs.r2_base_run}`, tiny `{record.inputs.r2_tiny_run}`",
        f"- exclude_checks: {', '.join(record.exclude_checks) or 'none'}",
        "- H3 families without a target check: "
        + (", ".join(record.h3_families_without_target) or "none"),
        "",
        "| rule | check | excluded | decided | reason | inputs |",
        "|---|---|---|---|---|---|",
    ]
    for item in record.evidence:
        shown = "; ".join(f"{k}={_shown(v)}" for k, v in sorted(item.inputs.items()))
        lines.append(
            f"| {item.rule} | {item.check_id} | {item.excluded} | {item.decided} | "
            f"{item.reason} | {shown} |"
        )
    return "\n".join(lines) + "\n"


def _shown(value: EvidenceValue) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "n/a" if value is None else str(value)
