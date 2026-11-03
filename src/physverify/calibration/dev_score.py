"""The campaign's development metric on calib.

Per check: the monotonicity of the full calib fit (rule 1.1) and the informedness J of decisions
made out of fold, everything that decides a verdict (tau, the per-check sigma correction) refitted
without the fold. Per stratum: the AUROC of the clip score out of fold (tau, sigma, p_c and the
weights refitted). One scene partition serves every check, both levels and the score. Pure, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import Field

from physverify.calibration.artifact import (
    Auroc,
    CalibrationArtifact,
    CheckCalibration,
    DatasetRef,
    Monotonicity,
    Reading,
    SettingsSnapshot,
    Stratified,
    Stratum,
    Weight,
)
from physverify.calibration.check_weights import Flag, raw_weight
from physverify.calibration.clause import clause
from physverify.calibration.dev_folds import FoldPart, fold_parts, scene_folds
from physverify.calibration.fit import (
    FitMeta,
    LevelData,
    assemble_stratum,
    auroc_of,
    build_artifact,
    fit_checks,
    settings_snapshot,
)
from physverify.calibration.oof import (
    DevScoreOof,
    LevelOof,
    check_valued,
    clip_valued,
    flagger,
    in_informedness,
    level_oof,
    oof_report,
)
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.calibration.stratified import (
    FamilyOf,
    Stratification,
    UnitScores,
    Valued,
    family_map,
    stratification,
    unit_scores,
)
from physverify.calibration.table import CalibrationRow, Role
from physverify.core.errors import UsageError
from physverify.core.models import SCHEMA_KEY, ArtifactModel
from physverify.schema.checks import InputLevel
from physverify.schema.io import stable_hash
from physverify.verification.sweep import CheckUnit
from physverify.verification.thresholding import ThresholdRule

DEVSCORE_SCHEMA = "physverify.devscore"
DEVSCORE_VERSION = "1.2.0"
DEVSCORE_FILE = "devscore.json"
SELECTION_LEVEL = InputLevel.PERCEIVED
SCOPE_NOTE = (
    "Development metric on calib (seen data, unlimited looks), not a held-out result. Refitted "
    "without each scene fold: tau, the per-check sigma correction (bias, factor), p_c, the "
    "weights and the clip score. Fitted on all of calib (in-sample): the per-stage error profile "
    "(diagnose stages), the per-stage sigma factors (calibrate stage-sigma, D10) and H1 (Kendall "
    "tau_b, rule 1.1). These folds are not those of fpr.cv: the artifact's FPR and J_cv differ. "
    "AUROC_cv pools the held-out scores of the fold models."
)
STRATIFIED_NOTE = (
    " Every scene carries a family: AUROC_cv and J_cv are also stratified (positives against the "
    "negatives of their own family, weighted by positives) in two readings, scored rows and "
    "every labelled row with an unscored one read as 0.0; the family floor is the in-sample "
    "AUROC of the family positive rate alone. n_met counts a check only when it is monotone, "
    "its J_cv^lo > 0, its stratified J_cv^lo > 0 and its stratified AUROC_cv^lo > 0.5 in both "
    "readings; the tie-break AUROC_cv is the stratified one (scored reading)."
)


class FoldInfo(ArtifactModel):
    index: int
    n_scenes: int
    scenes_hash: str


class CheckDevScore(ArtifactModel):
    h1: Monotonicity
    tau: float | None
    """tau of the full calib fit (the artifact's)."""
    tau_by_fold: tuple[float | None, ...]
    j_cv: Weight
    """J of out-of-fold decisions; ``value`` is max(0, J^lo), not normalised."""
    j_in_sample: Weight
    """The artifact's J before normalisation (TPR in-sample, FPR fpr.cv)."""
    folds_without_sigma_factor: int = 0
    """Folds whose sigma factor is unknown (too few pairs): every held-out subject of the check
    abstains there (``on_unknown_sigma = "abstain"``)."""
    meets_h1: bool
    meets_j: bool
    meets_both: bool
    auroc_cv_readings: dict[Reading, Auroc] = Field(default_factory=dict)
    """Out-of-fold AUROC of the corrected magnitude, global."""
    stratified: Stratified | None = None
    """None where the set has no families (or before 1.2.0)."""
    meets_stratified: bool | None = None
    meets_clause: bool | None = None
    """``meets_both`` and, where stratified, ``meets_stratified``; None before 1.2.0."""


class LevelDevScore(ArtifactModel):
    level: InputLevel
    profile: str
    checks: dict[str, CheckDevScore]
    auroc_cv: Auroc
    n_met: int
    met: tuple[str, ...]
    """The checks meeting the clause (``meets_clause``) from 1.2.0 on."""
    auroc_cv_operational: Auroc | None = None
    """``auroc_cv`` over every labelled clip, an unscored one read as 0.0."""
    stratification: Stratification | None = None
    stratified: Stratified | None = None
    n_met_global: int | None = None
    met_global: tuple[str, ...] | None = None
    """The checks meeting monotonicity and J_cv^lo > 0 (``meets_both``), the 1.1.0 count."""


class Selection(ArtifactModel):
    """The preregistered final-version criterion (most checks met, AUROC_cv tiebreak, early freeze
    when every oracle-evaluable check meets) on ``level``; ``oracle_evaluable`` is None without an
    oracle stratum."""

    level: InputLevel
    n_met: int
    met: tuple[str, ...]
    oracle_evaluable: tuple[str, ...] | None
    all_oracle_evaluable_met: bool | None
    tiebreak_auroc_cv: float | None
    """The stratified AUROC_cv (scored reading) where it applies, else the global one."""
    tiebreak_auroc_cv_global: float | None = None


class DevScoreReport(ArtifactModel):
    schema_: str = Field(default=DEVSCORE_SCHEMA, alias=SCHEMA_KEY)
    version: str = DEVSCORE_VERSION
    created_utc: str
    run_id: str
    commit: str
    config_hash: str
    checks_hash: str
    error_profile_hash: str
    dataset: DatasetRef
    settings: SettingsSnapshot
    folds: tuple[FoldInfo, ...]
    levels: tuple[LevelDevScore, ...]
    selection: Selection | None
    scope_note: str = SCOPE_NOTE


@dataclass(frozen=True)
class DevScore:
    """The report and the full calib fit it took the monotonicity from (``calibrate fit``'s
    artifact)."""

    report: DevScoreReport
    artifact: CalibrationArtifact
    oof: DevScoreOof


def require_bootstrap(settings: CalibrationSettings) -> None:
    """The weight is max(0, J^lo): without bootstrap J^lo is undefined and every check fails."""
    if settings.bootstrap < 1:
        raise UsageError("devscore needs calibration.bootstrap >= 1 (J^lo is a bootstrap bound)")


def oof_flags(
    held: Sequence[CalibrationRow], check: CheckCalibration, policy: ThresholdRule
) -> list[Flag]:
    """The populations of ``fit_check``'s J judged with the fold's tau and sigma correction; a
    fold without tau flags nothing."""
    flagged = flagger(check, policy)
    judged = [r for r in held if in_informedness(r)]
    flags = [Flag(r.scene_id, True, flagged(r)) for r in judged if r.role is Role.POSITIVE]
    return flags + [Flag(r.scene_id, False, flagged(r)) for r in judged if r.role is Role.CLEAN]


@dataclass(frozen=True)
class Scored:
    score: LevelDevScore
    full: Stratum
    """``fit_stratum`` of all the level's rows."""
    oof: LevelOof


def check_score(
    check_id: str,
    full: CheckCalibration,
    parts: Sequence[FoldPart],
    flags: Sequence[Flag],
    settings: CalibrationSettings,
    unit: UnitScores,
) -> CheckDevScore:
    """``full`` is the check of ``fit_checks`` on all of calib (weight not normalised); ``flags``
    its out-of-fold decisions; ``unit`` the out-of-fold AUROC_cv of its corrected magnitude."""
    j_cv = raw_weight(flags, settings)
    met = clause(full.monotonicity, j_cv, unit.stratified)
    return CheckDevScore(
        h1=full.monotonicity,
        tau=full.tau,
        tau_by_fold=tuple(part.stratum.checks[check_id].tau for part in parts),
        j_cv=j_cv,
        j_in_sample=full.weight,
        folds_without_sigma_factor=sum(
            part.stratum.checks[check_id].sigma.factor is None for part in parts
        ),
        meets_h1=met.meets_h1,
        meets_j=met.meets_j,
        meets_both=met.meets_both,
        auroc_cv_readings=unit.auroc_cv,
        stratified=unit.stratified,
        meets_stratified=met.meets_stratified,
        meets_clause=met.meets_clause,
    )


def check_flags(check_id: str, parts: Sequence[FoldPart], policy: ThresholdRule) -> list[Flag]:
    return [
        flag
        for part in parts
        for flag in oof_flags(
            [r for r in part.held.rows if r.check_id == check_id],
            part.stratum.checks[check_id],
            policy,
        )
    ]


def level_score(
    data: LevelData,
    checks: Sequence[CheckUnit],
    profile: str,
    config: CalibrationConfig,
    folds: Sequence[frozenset[str]],
    family_of: FamilyOf,
) -> Scored:
    fitted = fit_checks(data, checks, config)
    parts = fold_parts(data, checks, profile, config, folds)
    policy, settings = ThresholdRule(config.policy), config.calibration
    held = [(part.held, part.stratum) for part in parts]
    oof = level_oof(data.level, held, [c.check_id for c in checks], policy)
    scenes = [row.scene_id for row in data.rows]
    families = family_map(scenes, family_of)

    def unit(values: Mapping[str, Valued], flags: Sequence[Flag]) -> UnitScores:
        return unit_scores(list(values.values()), flags, families, settings)

    flags = {c.check_id: check_flags(c.check_id, parts, policy) for c in checks}
    scores = {
        c.check_id: check_score(
            c.check_id,
            fitted[c.check_id],
            parts,
            flags[c.check_id],
            settings,
            unit(check_valued(oof.checks[c.check_id]), flags[c.check_id]),
        )
        for c in checks
    }
    aggregate = unit(clip_valued(oof.aggregate), ())
    scored = [(c.scene_id, c.value, c.perturbed) for c in oof.aggregate if c.value is not None]
    met = tuple(sorted(c for c, s in scores.items() if s.meets_clause))
    met_global = tuple(sorted(c for c, s in scores.items() if s.meets_both))
    level = LevelDevScore(
        level=data.level,
        profile=profile,
        checks=scores,
        auroc_cv=auroc_of(scored, settings).model_copy(update={"in_sample": False}),
        n_met=len(met),
        met=met,
        auroc_cv_operational=aggregate.auroc_cv[Reading.OPERATIONAL],
        stratification=stratification(scenes, family_of),
        stratified=aggregate.stratified,
        n_met_global=len(met_global),
        met_global=met_global,
    )
    return Scored(level, assemble_stratum(fitted, data, profile, config), oof)


def selection(levels: Sequence[LevelDevScore]) -> Selection | None:
    """Freeze early when every check the oracle leaves evaluable is met (none evaluable: no)."""
    by_level = {level.level: level for level in levels}
    chosen = by_level.get(SELECTION_LEVEL)
    if chosen is None:
        return None
    oracle = by_level.get(InputLevel.ORACLE)
    evaluable = None if oracle is None else oracle.met
    return Selection(
        level=SELECTION_LEVEL,
        n_met=chosen.n_met,
        met=chosen.met,
        oracle_evaluable=evaluable,
        all_oracle_evaluable_met=None
        if evaluable is None
        else bool(evaluable) and set(evaluable) <= set(chosen.met),
        tiebreak_auroc_cv=chosen.auroc_cv.value
        if chosen.stratified is None
        else chosen.stratified.auroc_cv[Reading.SCORED].value,
        tiebreak_auroc_cv_global=chosen.auroc_cv.value,
    )


def dev_score(
    levels: Sequence[LevelData],
    checks: Sequence[CheckUnit],
    meta: FitMeta,
    config: CalibrationConfig,
    family_of: FamilyOf,
) -> DevScore:
    """``family_of`` names the family of a scene (None: none), for the stratified block."""
    require_bootstrap(config.calibration)
    folds = scene_folds(levels, config.calibration)
    scored = [
        level_score(level, checks, meta.profile, config, folds, family_of) for level in levels
    ]
    stratified = any(s.score.stratified is not None for s in scored)
    report = DevScoreReport(
        created_utc=meta.created_utc,
        run_id=meta.run_id,
        commit=meta.commit,
        config_hash=meta.config_hash,
        checks_hash=meta.checks_hash,
        error_profile_hash=meta.error_profile_hash,
        dataset=meta.dataset,
        settings=settings_snapshot(config),
        folds=tuple(
            FoldInfo(index=i, n_scenes=len(fold), scenes_hash=stable_hash(sorted(fold)))
            for i, fold in enumerate(folds)
        ),
        levels=tuple(s.score for s in scored),
        selection=selection([s.score for s in scored]),
        scope_note=SCOPE_NOTE + STRATIFIED_NOTE if stratified else SCOPE_NOTE,
    )
    artifact = build_artifact(tuple(s.full for s in scored), meta, config)
    oof = oof_report(meta.run_id, [s.oof for s in scored], config.calibration)
    return DevScore(report, artifact, oof)
