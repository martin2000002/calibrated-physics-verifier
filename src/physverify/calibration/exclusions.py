"""The preregistered exclusion rules 1.1, 1.2 and 1.5 of ADR 0015 (with its Enmienda 1).

Pure: the CLI loads the artifact, the evaluation and the two R2 runs, this module decides. Every
threshold comes from ``[exclusion_rules]`` (``configs/exclusion_rules.toml``): the model fields
have no defaults, so no number of a rule lives in code. It imports no consumer (H2, H3, finetune).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, model_validator

from physverify.calibration.apply import policy_for, rule_for
from physverify.calibration.exclusion_record import (
    EvidenceValue,
    ExclusionInputError,
    ExclusionRecord,
    InputHashes,
    RuleEvidence,
)
from physverify.calibration.loader import (
    Calibration,
    CalibrationError,
    Stratum,
    UnusableStratumError,
    open_stratum,
)
from physverify.calibration.schema import (
    Ci,
    DatasetRef,
    EvaluationReport,
    EvaluationStratum,
    RealFpr,
)
from physverify.calibration.stats import SceneBlock, clopper_pearson, scene_bootstrap
from physverify.calibration.table import (
    CalibrationDataError,
    CalibrationRow,
    Role,
    require_one_clean_per_scene,
    unique_row_step,
)
from physverify.calibration.thresholds import fails
from physverify.config.models import FrozenModel
from physverify.core.errors import ConfigError
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.io import stable_hash
from physverify.synth.plan import family_of_scene
from physverify.verification.runner import UNKNOWN_ROW_STEP, RowStatus

SECTION = "exclusion_rules"
DELTA_TPR = "delta_tpr"
KEPT = "kept"
ARMS = ("base", "tiny")


class R5aRule(FrozenModel):
    """1.1: kept only when the consumed stratum is usable, its weight > ``min_weight`` (status
    ok and Youden J low > 0) and the tau_b CI low > ``min_tau_b_low``."""

    check_ids: tuple[str, ...]
    min_weight: float
    min_tau_b_low: float


class R2Rule(FrozenModel):
    """1.2 (Enmienda 1): both detector arms judged at the artifact's tau of the consumed stratum
    on the calib clips of ``family``. ``rates = "total"``: an abstention or no event is no FAIL."""

    check_ids: tuple[str, ...]
    dataset: str
    subset: str
    family: str
    perturbation: str
    split: Split
    base_detector: str
    tiny_detector: str
    rates: Literal["total"]
    max_fpr_increase: float
    tpr_ci: float
    bootstrap_n: int
    bootstrap_seed: int


class RealRule(FrozenModel):
    """1.5: excluded when the worst exact Clopper-Pearson low of the real FPR exceeds the max."""

    datasets: tuple[str, ...]
    max_fpr_lower: float
    ci: float
    sided: Literal["two_sided"]
    allowed_sets: tuple[dict[str, int], ...]

    @model_validator(mode="after")
    def _sets_name_the_datasets(self) -> RealRule:
        for allowed in self.allowed_sets:
            if set(allowed) != set(self.datasets):
                raise ValueError(f"allowed set {allowed} must name exactly {list(self.datasets)}")
        return self


class ExclusionRules(FrozenModel):
    version: str
    consumed_level: InputLevel
    consumed_profile: str
    rules: tuple[str, ...]
    r5a: R5aRule
    r2: R2Rule
    real: RealRule

    @model_validator(mode="after")
    def _known_rules(self) -> ExclusionRules:
        unknown = sorted(set(self.rules) - set(RULES))
        if unknown:
            raise ValueError(f"unknown rules {unknown}; known: {sorted(RULES)}")
        return self

    def checked(self, known: Collection[str]) -> ExclusionRules:
        unknown = sorted(set(self.r5a.check_ids + self.r2.check_ids) - set(known))
        if unknown:
            raise ConfigError(f"[{SECTION}] names unknown checks {unknown}; known: {sorted(known)}")
        return self

    def sha256(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def exclusion_rules(raw: Mapping[str, object], known: Collection[str]) -> ExclusionRules:
    """``[exclusion_rules]`` validated, its check ids against the registry ``known``."""
    try:
        rules = ExclusionRules.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"[{SECTION}] is invalid:\n{exc}") from exc
    return rules.checked(known)


# --- inputs -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumedStratum:
    """The stratum H2/H3 consume; ``problem`` says why it cannot score (None: usable)."""

    stratum: Stratum
    problem: str | None


def consumed_stratum(
    path: Path,
    calibration: Calibration,
    rules: ExclusionRules,
    *,
    checks_hash: str,
    require_calibrated: bool,
) -> ConsumedStratum:
    """Through the consumers' gate (``open_stratum``); a missing stratum is an input error."""
    level, profile = rules.consumed_level, rules.consumed_profile
    try:
        stratum = open_stratum(
            path,
            level,
            profile,
            checks_hash=checks_hash,
            row_step=UNKNOWN_ROW_STEP,
            require_calibrated=require_calibrated,
        )
    except UnusableStratumError as exc:
        return ConsumedStratum(calibration.stratum(level, profile), str(exc))
    except CalibrationError as exc:
        raise ExclusionInputError(str(exc)) from exc
    return ConsumedStratum(stratum, None)


@dataclass(frozen=True)
class ArmRun:
    """One ``verify --policy raw`` run of the R2 clips; ``detectors`` are the DETECT providers
    its results name, ``report_sha256`` the hash of its checks.jsonl and table.csv."""

    run_id: str
    report_sha256: str
    detectors: frozenset[str]
    profile: str
    commit: str
    rows: tuple[CalibrationRow, ...]


@dataclass(frozen=True)
class R2Arms:
    base: ArmRun
    tiny: ArmRun

    def by_label(self) -> dict[str, ArmRun]:
        return {"base": self.base, "tiny": self.tiny}


@dataclass(frozen=True)
class DecisionInputs:
    calibration: Calibration
    consumed: ConsumedStratum
    evaluation: EvaluationReport
    evaluation_sha256: str
    arms: R2Arms
    checks_hash: str


def _evidence(
    rule: str, check_id: str, reason: str, inputs: Mapping[str, EvidenceValue], *, decided: bool
) -> RuleEvidence:
    return RuleEvidence(
        rule=rule,
        check_id=check_id,
        excluded=decided and reason != KEPT,
        decided=decided,
        reason=reason,
        inputs=dict(inputs),
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
    return _evidence("1.1", check_id, reason, inputs, decided=True)


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
        and family_of_scene(row.scene_id) == cfg.family
        and (
            row.role is Role.CLEAN
            or (row.role is Role.POSITIVE and row.perturbation == cfg.perturbation)
        )
    )


def require_matching_arms(
    arms: R2Arms, rows: Mapping[str, Sequence[CalibrationRow]], rules: ExclusionRules
) -> None:
    """Refuses swapped or mismatched arms: each must use its configured detector on the consumed
    profile, both from one commit and on the same clips, one clean clip per scene."""
    cfg = rules.r2
    detectors = {"base": cfg.base_detector, "tiny": cfg.tiny_detector}
    for label, arm in arms.by_label().items():
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
    require_matching_arms(inputs.arms, rows, rules)
    stratum = inputs.consumed.stratum
    for arm_rows_ in rows.values():
        _require_time_base(stratum, rules.consumed_profile, arm_rows_)
    tau = stratum.tau(check_id)
    evidence: dict[str, EvidenceValue] = {**_counts(rows), "tau": tau}
    if tau is None or inputs.consumed.problem is not None:
        evidence["stratum_problem"] = inputs.consumed.problem
        return _evidence("1.2", check_id, "no_usable_tau", evidence, decided=False)
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
        found[f"{label}.n_scored"] = sum(r.status is RowStatus.SCORED for r in arm)
    found["n_scenes"] = len({r.scene_id for r in rows["base"]})
    return found


def _rate(flags: FailFlags, clips: Iterable[str]) -> float | None:
    chosen = [flags[c] for c in clips]
    return sum(chosen) / len(chosen) if chosen else None


def _r2_decision(
    check_id: str,
    flags: Mapping[str, FailFlags],
    rows: Sequence[CalibrationRow],
    cfg: R2Rule,
    evidence: dict[str, EvidenceValue],
) -> RuleEvidence:
    clean = sorted(r.clip_id for r in rows if r.role is Role.CLEAN)
    positive = sorted(r.clip_id for r in rows if r.role is Role.POSITIVE)
    rates = {
        f"{arm}.{name}": _rate(flags[arm], clips)
        for arm in ARMS
        for name, clips in (("fpr", clean), ("tpr", positive))
    }
    base_fpr, tiny_fpr = rates["base.fpr"], rates["tiny.fpr"]
    delta_fpr = None if base_fpr is None or tiny_fpr is None else base_fpr - tiny_fpr
    ci = _delta_tpr_ci(flags, rows, cfg)
    if delta_fpr is not None and delta_fpr > cfg.max_fpr_increase:
        reason = "fpr_increase"
    elif ci.high is not None and ci.high < 0.0:
        reason = "tpr_decrease"
    else:
        reason = KEPT
    evidence |= {
        **rates,
        "delta_fpr": delta_fpr,
        "delta_tpr_low": ci.low,
        "delta_tpr_high": ci.high,
        "delta_tpr_non_finite": ci.non_finite,
    }
    return _evidence("1.2", check_id, reason, evidence, decided=True)


def _delta_tpr_ci(
    flags: Mapping[str, FailFlags], rows: Sequence[CalibrationRow], cfg: R2Rule
) -> Ci:
    """Scene bootstrap of TPR(base) - TPR(tiny), paired clip by clip inside a drawn scene."""
    blocks = [
        SceneBlock(clip_id=r.scene_id, item=(flags["base"][r.clip_id], flags["tiny"][r.clip_id]))
        for r in rows
        if r.role is Role.POSITIVE
    ]
    found = scene_bootstrap(blocks, _delta_tpr, cfg.bootstrap_n, cfg.bootstrap_seed, cfg.tpr_ci)
    return found.get(DELTA_TPR, Ci())


def _delta_tpr(pairs: Sequence[tuple[bool, bool]]) -> dict[str, float]:
    if not pairs:
        return {DELTA_TPR: float("nan")}
    return {DELTA_TPR: sum(int(b) - int(t) for b, t in pairs) / len(pairs)}


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
            return stratum
    raise ExclusionInputError(
        f"the evaluation has no {rules.consumed_level.value}/{rules.consumed_profile} stratum"
    )


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
        return _evidence("1.5", check_id, "no_rows", evidence, decided=False)
    worst = max(lows)
    evidence["worst_cp_low"] = worst
    reason = "real_fpr_high" if worst > cfg.max_fpr_lower else KEPT
    return _evidence("1.5", check_id, reason, evidence, decided=True)


# --- decision -----------------------------------------------------------------------------------

Apply = Callable[[DecisionInputs, ExclusionRules], tuple[RuleEvidence, ...]]
RULES: Mapping[str, Apply] = {"1.1": r5a_rule, "1.2": r2_rule, "1.5": real_fpr_rule}
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
