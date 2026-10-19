"""``fit_calibration``: calib rows in, the artifact out (orchestration only, no I/O)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from physverify.calibration.loader import Stratum as StratumView
from physverify.calibration.monotonicity import deltas, monotonicity
from physverify.calibration.schema import (
    OK,
    UNRANKED_NEGATIVES,
    Aggregate,
    ArtifactStatus,
    Auroc,
    CalibrationArtifact,
    CheckCalibration,
    CheckStatus,
    Ci,
    DatasetRef,
    Rate,
    SettingsSnapshot,
    SigmaCorrection,
    StageSigma,
    Stratum,
    Weight,
)
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.calibration.sigma import SigmaPair, fit_sigma, oracle_correction
from physverify.calibration.stats import (
    SceneBlock,
    auroc,
    conformal_quantile,
    rate,
    scene_bootstrap,
)
from physverify.calibration.table import (
    CalibrationRow,
    Role,
    corrected,
    require_subjects,
    unique_row_step,
)
from physverify.calibration.thresholds import (
    Tau,
    cv_verdicts,
    detection_limit,
    detections,
    fails,
    fit_tau,
    fpr_summary,
    tpr_by_level,
    verdict_of,
)
from physverify.calibration.verdict import DECIDED, Rule, clip_verdict
from physverify.calibration.weights import Flag, Labelled, fit_probability, normalised, raw_weight
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.sweep import CheckUnit

AUROC_KEY = "auroc"
NOT_RECALIBRATED = "not recalibrated (--no-sigma): the propagated sigma is used as is"


@dataclass(frozen=True)
class LevelData:
    """Raw-policy rows and results of one input level; ``sigma_pairs`` None = not measured."""

    level: InputLevel
    rows: tuple[CalibrationRow, ...]
    results: tuple[CheckResult, ...]
    sigma_pairs: tuple[SigmaPair, ...] | None = ()


@dataclass(frozen=True)
class FitMeta:
    created_utc: str
    run_id: str
    commit: str
    config_hash: str
    checks_hash: str
    error_profile_hash: str
    profile: str
    dataset: DatasetRef
    stage_sigma: StageSigma


@dataclass(frozen=True)
class FitInputs:
    levels: tuple[LevelData, ...]
    checks: tuple[CheckUnit, ...]
    meta: FitMeta


def fit_calibration(inputs: FitInputs, config: CalibrationConfig) -> CalibrationArtifact:
    strata = tuple(_stratum(level, inputs, config) for level in inputs.levels)
    settings = config.calibration
    return CalibrationArtifact(
        status=_status(strata),
        created_utc=inputs.meta.created_utc,
        run_id=inputs.meta.run_id,
        commit=inputs.meta.commit,
        config_hash=inputs.meta.config_hash,
        checks_hash=inputs.meta.checks_hash,
        error_profile_hash=inputs.meta.error_profile_hash,
        dataset=inputs.meta.dataset,
        settings=SettingsSnapshot(
            alpha=settings.alpha,
            beta=settings.beta,
            k_sigma=config.policy.k_sigma,
            folds=settings.folds,
            bootstrap=settings.bootstrap,
            seed=settings.seed,
            ci=settings.ci,
            min_sigma_pairs=settings.min_sigma_pairs,
            platt_l2=settings.platt_l2,
        ),
        strata=strata,
        stage_sigma=inputs.meta.stage_sigma,
    )


def _status(strata: Sequence[Stratum]) -> ArtifactStatus:
    checks_ok = all(c.status == OK for s in strata for c in s.checks.values())
    aggregate_ok = all(s.aggregate.sufficient for s in strata)
    return "calibrated" if strata and checks_ok and aggregate_ok else "insufficient"


def _stratum(data: LevelData, inputs: FitInputs, config: CalibrationConfig) -> Stratum:
    policy = ThresholdPolicy(config.policy)
    fitted = {
        check.check_id: fit_check(
            [r for r in data.rows if r.check_id == check.check_id],
            check,
            _sigma(data, check.check_id, config.calibration),
            policy,
            config.calibration,
        )
        for check in inputs.checks
    }
    weights = normalised({c: f.weight for c, f in fitted.items() if f.status == OK})
    checks = {
        c: f.model_copy(update={"weight": weights.get(c, _zero(f.weight))})
        for c, f in fitted.items()
    }
    model = Stratum(
        level=data.level,
        profile=inputs.meta.profile,
        row_step=unique_row_step(data.rows),
        k_sigma=config.policy.k_sigma,
        on_unknown_sigma=config.policy.on_unknown_sigma,
        checks=checks,
        aggregate=_empty_aggregate(),
    )
    aggregate = fit_aggregate(model, data, config.calibration)
    return model.model_copy(update={"aggregate": aggregate})


def _zero(weight: Weight) -> Weight:
    return weight.model_copy(update={"value": 0.0})


def _sigma(data: LevelData, check_id: str, settings: CalibrationSettings) -> SigmaCorrection:
    if data.level is InputLevel.ORACLE:
        return oracle_correction()
    if data.sigma_pairs is None:
        return oracle_correction().model_copy(
            update={"reason": NOT_RECALIBRATED, "estimated": True}
        )
    return fit_sigma([p for p in data.sigma_pairs if p.check_id == check_id], settings)


def check_status(tau: Tau, positives: Sequence[CalibrationRow]) -> CheckStatus:
    """No clean scene at all (or only NaN ones), then too few for the conformal rank (AC2), then
    no positives."""
    if tau.n == 0:
        return UNRANKED_NEGATIVES if tau.n_dropped else "no_negatives"
    if not tau.sufficient:
        return "insufficient"
    return OK if any(r.has_event for r in positives) else "no_positives"


def fit_check(
    rows: Sequence[CalibrationRow],
    check: CheckUnit,
    sigma: SigmaCorrection,
    policy: ThresholdPolicy,
    settings: CalibrationSettings,
) -> CheckCalibration:
    rule = Rule(policy, sigma.bias, sigma.factor)
    tolerance = settings.tau_tolerance.get(check.check_id, 0.0)
    clean = [r for r in rows if r.role is Role.CLEAN]
    positives = [r for r in rows if r.role is Role.POSITIVE]
    require_subjects(rows)
    tau = fit_tau(clean, sigma.bias, settings.alpha, tolerance)
    cv = cv_verdicts(clean, rule, settings, tolerance)
    flags = [Flag(r.scene_id, True, fails(r, rule, tau.value)) for r in positives if r.has_event]
    flags += [
        Flag(r.scene_id, False, cv[r.clip_id] is Verdict.FAIL) for r in clean if r.clip_id in cv
    ]
    mu_unit = next((r.mu_unit for r in positives), check.magnitude_unit)
    return CheckCalibration(
        unit=check.magnitude_unit,
        status=check_status(tau, positives),
        tau=tau.value,
        tau_tolerance=tolerance,
        tau_conformal=tau.conformal,
        n_clean_scenes=tau.n,
        n_clean_dropped=tau.n_dropped,
        n_positive_clips=sum(r.has_event for r in positives),
        n_off_target_clips=sum(r.role is Role.OFF_TARGET and r.scored for r in rows),
        fpr=fpr_summary(rows, rule, tau.value, cv, settings),
        tpr_by_level=tpr_by_level(positives, rule, tau.value, settings.ci),
        auroc=check_auroc(rows, sigma.bias, settings),
        abstention_rate=_abstention(rows, rule, tau.value, settings.ci),
        detection_limit=detection_limit(detections(positives, rule, tau.value), mu_unit, settings),
        monotonicity=monotonicity(deltas(rows, sigma.bias), settings),
        weight=raw_weight(flags, settings),
        probability=fit_probability(platt_data(rows, rule, tau.value), tau.value, settings),
        sigma=sigma,
    )


LABELLED_ROLES = (Role.POSITIVE, Role.CLEAN, Role.OFF_TARGET)


def _labelled(rows: Sequence[CalibrationRow], bias: float) -> list[Labelled]:
    """The clip's max corrected magnitude: the threshold-free ranking the AUROC measures."""
    return [
        Labelled(r.scene_id, corrected(r, bias), r.role is Role.POSITIVE)
        for r in rows
        if r.scored and r.role in LABELLED_ROLES
    ]


def platt_data(rows: Sequence[CalibrationRow], rule: Rule, tau: float | None) -> list[Labelled]:
    """The magnitude of the decided clip outcomes, the ones ``score_clip`` passes to p_c, so that
    p_c is fitted on what it is applied to. Undecided clips are left out."""
    found: list[Labelled] = []
    for r in rows:
        if not r.scored or r.role not in LABELLED_ROLES:
            continue
        outcome = clip_verdict(r.subjects, rule, tau)
        if outcome.verdict in DECIDED and outcome.magnitude is not None:
            found.append(Labelled(r.scene_id, outcome.magnitude, r.role is Role.POSITIVE))
    return found


def _abstention(rows: Sequence[CalibrationRow], rule: Rule, tau: float | None, ci: float) -> Rate:
    """ABSTAIN over the clips with an event. Without tau no clip is judged, so the rate is
    undefined (value None, n 0) rather than 0 or 1."""
    if tau is None:
        return rate(0, 0, ci)
    judged = [r for r in rows if r.has_event and r.role is not Role.COUPLED]
    abstained = sum(verdict_of(r, rule, tau) is Verdict.ABSTAIN for r in judged)
    return rate(abstained, len(judged), ci)


def auroc_of(scored: Sequence[tuple[str, float, bool]], settings: CalibrationSettings) -> Auroc:
    """In-sample AUROC (marked so) of (scene, value, positive) with a scene-bootstrap interval."""
    pos = [v for _, v, positive in scored if positive]
    neg = [v for _, v, positive in scored if not positive]
    blocks = [SceneBlock(scene, (value, positive)) for scene, value, positive in scored]
    ci = scene_bootstrap(
        blocks,
        lambda s: {AUROC_KEY: auroc([v for v, p in s if p], [v for v, p in s if not p])},
        settings.bootstrap,
        settings.seed,
        settings.ci,
    ).get(AUROC_KEY, Ci())
    value = auroc(pos, neg)
    return Auroc(
        value=value if value == value else None, ci=ci, n_positive=len(pos), n_negative=len(neg)
    )


def check_auroc(
    rows: Sequence[CalibrationRow], bias: float, settings: CalibrationSettings
) -> Auroc:
    return auroc_of(
        [(d.scene_id, d.magnitude, d.positive) for d in _labelled(rows, bias)], settings
    )


def _empty_aggregate() -> Aggregate:
    empty = Auroc(value=None, ci=Ci(), n_positive=0, n_negative=0)
    return Aggregate(
        score_threshold=None, sufficient=False, n_clean_scenes=0, n_perturbed_clips=0, auroc=empty
    )


def fit_aggregate(model: Stratum, data: LevelData, settings: CalibrationSettings) -> Aggregate:
    """s* = conformal quantile of 1 - S over the clean clips (one per scene). In-sample: tau, p_c
    and the weights behind S were fitted on the same clips, so its coverage is optimistic (the
    held-out flag rate is measured by ``evaluate``)."""
    from physverify.calibration.apply import score_clip

    view = StratumView(model=model, source="fit", artifact=None)
    by_clip: dict[str, list[CheckResult]] = {}
    for result in data.results:
        by_clip.setdefault(result.clip_id, []).append(result)
    info = _clip_info(data.rows)
    scored: list[tuple[str, float, bool]] = []
    for clip_id in sorted(by_clip):
        if clip_id not in info:
            continue
        score = score_clip(by_clip[clip_id], view).score
        scene, perturbed = info[clip_id]
        if score is not None and perturbed is not None:
            scored.append((scene, 1.0 - score, perturbed))
    threshold = conformal_quantile([v for _, v, p in scored if not p], settings.alpha)
    return Aggregate(
        score_threshold=threshold.value,
        sufficient=threshold.sufficient,
        score_threshold_in_sample=True,
        n_clean_scenes=sum(not p for _, _, p in scored),
        n_perturbed_clips=sum(p for _, _, p in scored),
        auroc=auroc_of(scored, settings),
    )


def _clip_info(rows: Sequence[CalibrationRow]) -> dict[str, tuple[str, bool | None]]:
    """clip -> (scene, perturbed?); None for a clip whose perturbation targets no check."""
    info: dict[str, tuple[str, bool | None]] = {}
    for row in rows:
        scene, previous = info.get(row.clip_id, (row.scene_id, None))
        if row.role is Role.CLEAN:
            info[row.clip_id] = (scene, False)
        elif row.role is Role.POSITIVE or previous:
            info[row.clip_id] = (scene, True)
        else:
            info[row.clip_id] = (scene, previous)
    return info
