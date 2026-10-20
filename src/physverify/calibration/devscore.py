"""The campaign's development metric on calib (ADR 0015 amendment 2 section 4, WP-O).

Per check: H1 of the full calib fit (rule 1.1) and the informedness J of decisions made out of
fold, everything that decides a verdict (tau, the per-check sigma correction) refitted without
the fold. Per stratum: the AUROC of the clip score out of fold (tau, sigma, p_c and the weights
refitted). One scene partition serves every check, both levels and the score. Pure, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import Field

from physverify.calibration.fit import (
    FitMeta,
    LevelData,
    assemble_stratum,
    auroc_of,
    build_artifact,
    clip_scores,
    fit_checks,
    fit_stratum,
    settings_snapshot,
)
from physverify.calibration.schema import (
    SCHEMA_KEY,
    ArtifactModel,
    Auroc,
    CalibrationArtifact,
    CheckCalibration,
    DatasetRef,
    Monotonicity,
    SettingsSnapshot,
    Stratum,
    Weight,
)
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.calibration.stats import grouped_folds
from physverify.calibration.table import CalibrationRow, Role
from physverify.calibration.thresholds import fails
from physverify.calibration.verdict import Rule
from physverify.calibration.weights import Flag, raw_weight
from physverify.core.errors import UsageError
from physverify.schema.checks import InputLevel
from physverify.schema.io import stable_hash
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.sweep import CheckUnit

DEVSCORE_SCHEMA = "physverify.devscore"
DEVSCORE_VERSION = "1.0.0"
DEVSCORE_FILE = "devscore.json"
SELECTION_LEVEL = InputLevel.PERCEIVED
SCOPE_NOTE = (
    "Development metric on calib (seen data, unlimited looks), not a held-out result. Refitted "
    "without each scene fold: tau, the per-check sigma correction (bias, factor), p_c, the "
    "weights and the clip score. Fitted on all of calib (in-sample): the per-stage error profile "
    "(diagnose stages), the per-stage sigma factors (calibrate stage-sigma, D10) and H1 (Kendall "
    "tau_b, rule 1.1). These folds are not those of fpr.cv: the artifact's FPR and J_cv differ."
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
    meets_h1: bool
    meets_j: bool
    meets_both: bool


class LevelDevScore(ArtifactModel):
    level: InputLevel
    profile: str
    checks: dict[str, CheckDevScore]
    auroc_cv: Auroc
    n_met: int
    met: tuple[str, ...]


class Selection(ArtifactModel):
    """The criterion of amendment 2 section 4 on ``level``; ``oracle_evaluable`` is None without
    an oracle stratum."""

    level: InputLevel
    n_met: int
    met: tuple[str, ...]
    oracle_evaluable: tuple[str, ...] | None
    all_oracle_evaluable_met: bool | None
    tiebreak_auroc_cv: float | None


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
    """The report and the full calib fit it took H1 from (``calibrate fit``'s artifact)."""

    report: DevScoreReport
    artifact: CalibrationArtifact


@dataclass(frozen=True)
class FoldPart:
    train: LevelData
    held: LevelData
    stratum: Stratum
    """Fitted on ``train`` only."""


def require_bootstrap(settings: CalibrationSettings) -> None:
    """The weight is max(0, J^lo): without bootstrap J^lo is undefined and every check fails."""
    if settings.bootstrap < 1:
        raise UsageError("devscore needs calibration.bootstrap >= 1 (J^lo is a bootstrap bound)")


def scene_folds(
    levels: Sequence[LevelData], settings: CalibrationSettings
) -> tuple[frozenset[str], ...]:
    scenes = sorted({row.scene_id for level in levels for row in level.rows})
    return grouped_folds(scenes, settings.folds, settings.seed)


def split_data(data: LevelData, fold: frozenset[str]) -> tuple[LevelData, LevelData]:
    """(train, held out): rows, results and sigma pairs follow the scene of their row; a result
    of a clip without a row is dropped, as ``clip_scores`` drops it."""
    scene = {row.clip_id: row.scene_id for row in data.rows}

    def part(held: bool) -> LevelData:
        pairs = data.sigma_pairs
        return LevelData(
            level=data.level,
            rows=tuple(r for r in data.rows if (r.scene_id in fold) is held),
            results=tuple(
                r for r in data.results if r.clip_id in scene and (scene[r.clip_id] in fold) is held
            ),
            sigma_pairs=None
            if pairs is None
            else tuple(p for p in pairs if (p.scene_id in fold) is held),
        )

    return part(False), part(True)


def fold_parts(
    data: LevelData,
    checks: Sequence[CheckUnit],
    profile: str,
    config: CalibrationConfig,
    folds: Sequence[frozenset[str]],
) -> tuple[FoldPart, ...]:
    found: list[FoldPart] = []
    for fold in folds:
        train, held = split_data(data, fold)
        found.append(FoldPart(train, held, fit_stratum(train, checks, profile, config)))
    return tuple(found)


def oof_flags(
    held: Sequence[CalibrationRow], check: CheckCalibration, policy: ThresholdPolicy
) -> list[Flag]:
    """The populations of ``fit_check``'s J judged with the fold's tau and sigma correction; a
    fold without tau flags nothing."""
    rule, tau = Rule(policy, check.sigma.bias, check.sigma.factor), check.tau

    def flagged(row: CalibrationRow) -> bool:
        return tau is not None and fails(row, rule, tau)

    flags = [
        Flag(r.scene_id, True, flagged(r)) for r in held if r.role is Role.POSITIVE and r.has_event
    ]
    flags += [
        Flag(r.scene_id, False, flagged(r)) for r in held if r.role is Role.CLEAN and r.scored
    ]
    return flags


def check_score(
    check_id: str,
    full: CheckCalibration,
    parts: Sequence[FoldPart],
    policy: ThresholdPolicy,
    settings: CalibrationSettings,
) -> CheckDevScore:
    """``full`` is the check of ``fit_checks`` on all of calib (weight not normalised)."""
    flags = [
        flag
        for part in parts
        for flag in oof_flags(
            [r for r in part.held.rows if r.check_id == check_id],
            part.stratum.checks[check_id],
            policy,
        )
    ]
    j_cv = raw_weight(flags, settings)
    low = j_cv.informedness_ci.low
    meets_h1, meets_j = full.monotonicity.h1_pass, low is not None and low > 0.0
    return CheckDevScore(
        h1=full.monotonicity,
        tau=full.tau,
        tau_by_fold=tuple(part.stratum.checks[check_id].tau for part in parts),
        j_cv=j_cv,
        j_in_sample=full.weight,
        meets_h1=meets_h1,
        meets_j=meets_j,
        meets_both=meets_h1 and meets_j,
    )


def level_score(
    data: LevelData,
    checks: Sequence[CheckUnit],
    profile: str,
    config: CalibrationConfig,
    folds: Sequence[frozenset[str]],
) -> tuple[LevelDevScore, Stratum]:
    """The level's score and its full calib stratum (``fit_stratum`` of all its rows)."""
    fitted = fit_checks(data, checks, config)
    parts = fold_parts(data, checks, profile, config, folds)
    policy, settings = ThresholdPolicy(config.policy), config.calibration
    scores = {
        c.check_id: check_score(c.check_id, fitted[c.check_id], parts, policy, settings)
        for c in checks
    }
    scored = [s for part in parts for s in clip_scores(part.stratum, part.held)]
    met = tuple(sorted(c for c, s in scores.items() if s.meets_both))
    level = LevelDevScore(
        level=data.level,
        profile=profile,
        checks=scores,
        auroc_cv=auroc_of(scored, settings).model_copy(update={"in_sample": False}),
        n_met=len(met),
        met=met,
    )
    return level, assemble_stratum(fitted, data, profile, config)


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
        tiebreak_auroc_cv=chosen.auroc_cv.value,
    )


def dev_score(
    levels: Sequence[LevelData],
    checks: Sequence[CheckUnit],
    meta: FitMeta,
    config: CalibrationConfig,
) -> DevScore:
    require_bootstrap(config.calibration)
    folds = scene_folds(levels, config.calibration)
    scored = [level_score(level, checks, meta.profile, config, folds) for level in levels]
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
        levels=tuple(level for level, _ in scored),
        selection=selection([level for level, _ in scored]),
    )
    return DevScore(report, build_artifact(tuple(full for _, full in scored), meta, config))
