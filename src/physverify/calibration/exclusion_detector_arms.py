"""Rule 1.2: the R2 false-fail rate of the tiny detector against the base one, on the same clips."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

from physverify.calibration.apply import policy_for, rule_for
from physverify.calibration.artifact_io import (
    Stratum,
)
from physverify.calibration.exclusion_inputs import (
    ArmRun,
    DecisionInputs,
    R2Arms,
    rule_evidence,
)
from physverify.calibration.exclusion_rules import (
    ARMS,
    DELTA_TPR,
    KEPT,
    R2_RULE_ID,
    ExclusionRules,
    R2Rule,
)
from physverify.calibration.table import (
    CalibrationDataError,
    CalibrationRow,
    Role,
    in_family,
    require_one_clean_per_scene,
    unique_row_step,
)
from physverify.calibration.thresholds import fails
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.stats.intervals import Ci
from physverify.verification.exclusion_record import (
    EvidenceValue,
    ExclusionInputError,
    RuleEvidence,
)

# --- 1.2 ----------------------------------------------------------------------------------------


def r2_rule(inputs: DecisionInputs, rules: ExclusionRules) -> tuple[RuleEvidence, ...]:
    return tuple(_r2(inputs, c, rules) for c in sorted(rules.r2.check_ids))


def arm_rows(arm: ArmRun, check_id: str, cfg: R2Rule) -> tuple[CalibrationRow, ...]:
    """The clean and ``perturbation`` positive rows of ``check_id`` on the calib ``family``. An
    R2 clip whose overhang is <= 0 is off-target (still supported), not a positive."""
    return tuple(
        row
        for row in arm.rows
        if row.check_id == check_id
        and row.split is cfg.split
        and in_family(row.scene_id, cfg.family)
        and (
            row.role is Role.CLEAN
            or (row.role is Role.POSITIVE and row.perturbation == cfg.perturbation)
        )
    )


def require_matching_arms(
    arms: R2Arms,
    rows: Mapping[str, Sequence[CalibrationRow]],
    rules: ExclusionRules,
    checks_hash: str,
) -> None:
    """Refuses swapped or mismatched arms: each must use its configured detector on the consumed
    profile and the decision's check parameters, both from one commit and on the same clips, one
    clean clip per scene."""
    cfg = rules.r2
    detectors = {"base": cfg.base_detector, "tiny": cfg.tiny_detector}
    for label, arm in arms.by_label().items():
        if arm.checks_hash != checks_hash:
            raise ExclusionInputError(
                f"the R2 {label} run {arm.run_id} ran checks_hash {arm.checks_hash[:12] or '?'}, "
                f"not the decision's {checks_hash[:12]}: rerun it without check overrides"
            )
        if arm.detectors != {detectors[label]}:
            raise ExclusionInputError(
                f"the R2 {label} run {arm.run_id} used detectors {sorted(arm.detectors)}, "
                f"not [{detectors[label]!r}]"
            )
        if arm.profile != rules.consumed_profile:
            raise ExclusionInputError(
                f"the R2 {label} run {arm.run_id} has profile {arm.profile!r}, "
                f"not {rules.consumed_profile!r}"
            )
        try:
            require_one_clean_per_scene(rows[label])
        except CalibrationDataError as exc:
            raise ExclusionInputError(f"R2 {label} run: {exc}") from exc
    if arms.base.commit != arms.tiny.commit:
        raise ExclusionInputError(
            f"the R2 runs come from different commits ({arms.base.commit} != {arms.tiny.commit})"
        )
    base, tiny = ({r.clip_id for r in rows[label]} for label in ARMS)
    if not base or base != tiny:
        raise ExclusionInputError(
            f"the R2 runs must hold the same clips of {cfg.family} {cfg.split.value}: "
            f"{len(base)} base, {len(tiny)} tiny, {len(base ^ tiny)} not shared"
        )


def _require_time_base(stratum: Stratum, profile: str, rows: Sequence[CalibrationRow]) -> None:
    try:
        step = unique_row_step(rows)
    except CalibrationDataError as exc:
        raise ExclusionInputError(str(exc)) from exc
    problem = stratum.time_base_problem(profile, step)
    if problem is not None:
        raise ExclusionInputError(f"R2 run: {problem}")


def _r2(inputs: DecisionInputs, check_id: str, rules: ExclusionRules) -> RuleEvidence:
    rows = {
        label: arm_rows(arm, check_id, rules.r2) for label, arm in inputs.arms.by_label().items()
    }
    require_matching_arms(inputs.arms, rows, rules, inputs.checks_hash)
    require_scored(rows)
    stratum = inputs.consumed.stratum
    for arm_rows_ in rows.values():
        _require_time_base(stratum, rules.consumed_profile, arm_rows_)
    tau = stratum.tau(check_id)
    evidence: dict[str, EvidenceValue] = {**_counts(rows), "tau": tau}
    if tau is None or inputs.consumed.problem is not None:
        evidence["stratum_problem"] = inputs.consumed.problem
        return rule_evidence(R2_RULE_ID, check_id, "no_usable_tau", evidence, decided=False)
    rule = rule_for(stratum, check_id, policy_for(stratum))
    evidence |= {"bias": rule.bias, "sigma_factor": rule.factor}
    flags = {label: {r.clip_id: fails(r, rule, tau) for r in arm} for label, arm in rows.items()}
    return _r2_decision(check_id, flags, rows["base"], rules.r2, evidence)


FailFlags = Mapping[str, bool]


def _counts(rows: Mapping[str, Sequence[CalibrationRow]]) -> dict[str, EvidenceValue]:
    found: dict[str, EvidenceValue] = {}
    for label, arm in rows.items():
        found[f"{label}.n_clean"] = sum(r.role is Role.CLEAN for r in arm)
        found[f"{label}.n_positive"] = sum(r.role is Role.POSITIVE for r in arm)
        found[f"{label}.n_scored"] = sum(r.scored for r in arm)
    found["n_scenes"] = len({r.scene_id for r in rows["base"]})
    return found


def require_scored(rows: Mapping[str, Sequence[CalibrationRow]]) -> None:
    """1.2 counts FAIL over scored clips, as the evaluation does: an unscored row in either arm
    is refused rather than counted as no FAIL (an arm failing to extract would lower its FPR)."""
    for label, arm in rows.items():
        unscored = sorted(r.clip_id for r in arm if not r.scored)
        if unscored:
            raise ExclusionInputError(
                f"the R2 {label} run has {len(unscored)} unscored clips ({unscored[:3]}): "
                "rerun it once they extract"
            )


def _rate(flags: FailFlags, clips: Iterable[str]) -> Fraction | None:
    chosen = [flags[c] for c in clips]
    return Fraction(sum(chosen), len(chosen)) if chosen else None


def _float(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


def _r2_decision(
    check_id: str,
    flags: Mapping[str, FailFlags],
    rows: Sequence[CalibrationRow],
    cfg: R2Rule,
    evidence: dict[str, EvidenceValue],
) -> RuleEvidence:
    """Either test may exclude alone; ``decided`` is False when neither fires and one of them
    is undefined (no clean or positive clip, or a TPR interval that is not finite). Rates are
    exact fractions so that a gap of exactly ``max_fpr_increase`` is not excluded by rounding."""
    clean = sorted(r.clip_id for r in rows if r.role is Role.CLEAN)
    positive = sorted(r.clip_id for r in rows if r.role is Role.POSITIVE)
    rates = {
        f"{arm}.{name}": _rate(flags[arm], clips)
        for arm in ARMS
        for name, clips in (("fpr", clean), ("tpr", positive))
    }
    base_fpr, tiny_fpr = rates["base.fpr"], rates["tiny.fpr"]
    delta_fpr = None if base_fpr is None or tiny_fpr is None else base_fpr - tiny_fpr
    max_increase = Fraction(repr(cfg.max_fpr_increase))
    ci = _delta_tpr_ci(flags, rows, cfg) if positive else Ci()
    high = ci.high if ci.high is not None and math.isfinite(ci.high) else None
    if delta_fpr is not None and delta_fpr > max_increase:
        reason, decided = "fpr_increase", True
    elif high is not None and high < 0.0:
        reason, decided = "tpr_decrease", True
    elif not positive:
        reason, decided = "no_positive", False
    elif high is None:
        reason, decided = "tpr_ci_undefined", False
    elif delta_fpr is None:
        reason, decided = "no_clean", False
    else:
        reason, decided = KEPT, True
    evidence |= {
        **{key: _float(value) for key, value in rates.items()},
        "delta_fpr": _float(delta_fpr),
        "delta_tpr_low": ci.low,
        "delta_tpr_high": ci.high,
        "delta_tpr_non_finite": ci.non_finite,
    }
    return rule_evidence(R2_RULE_ID, check_id, reason, evidence, decided=decided)


def _delta_tpr_ci(
    flags: Mapping[str, FailFlags], rows: Sequence[CalibrationRow], cfg: R2Rule
) -> Ci:
    """Scene bootstrap of TPR(base) - TPR(tiny), paired clip by clip inside a drawn scene."""
    blocks = [
        ClusterBlock(clip_id=r.scene_id, item=(flags["base"][r.clip_id], flags["tiny"][r.clip_id]))
        for r in rows
        if r.role is Role.POSITIVE
    ]
    found = scene_bootstrap(blocks, _delta_tpr, cfg.bootstrap_n, cfg.bootstrap_seed, cfg.tpr_ci)
    return found.get(DELTA_TPR, Ci())


def _delta_tpr(pairs: Sequence[tuple[bool, bool]]) -> dict[str, float]:
    if not pairs:
        return {DELTA_TPR: float("nan")}
    return {DELTA_TPR: sum(int(b) - int(t) for b, t in pairs) / len(pairs)}
