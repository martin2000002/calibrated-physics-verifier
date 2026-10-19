"""``evaluate``: the frozen artifact on held-out clips and real video (D12). No fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from physverify.calibration.apply import policy_for, rule_for, score_clip
from physverify.calibration.fit import LevelData
from physverify.calibration.loader import Calibration, CalibrationError, Stratum
from physverify.calibration.monotonicity import scene_kendall
from physverify.calibration.scenes import SceneScope, in_scope
from physverify.calibration.schema import (
    DatasetRef,
    EvaluationReport,
    EvaluationStratum,
    H1PrimaryView,
    HeldoutCheck,
    Rate,
    RealFpr,
    SceneKendall,
)
from physverify.calibration.settings import CalibrationConfig
from physverify.calibration.stats import clustered_rate
from physverify.calibration.table import (
    CalibrationRow,
    Role,
    require_one_clean_per_scene,
    unique_row_step,
)
from physverify.calibration.thresholds import (
    fails,
    false_positive_rates,
    off_target_rate,
    tpr_by_level,
    verdict_of,
)
from physverify.calibration.verdict import Rule
from physverify.schema.checks import CheckResult
from physverify.schema.clip import Split

SCORE_CHECK = "score"


@dataclass(frozen=True)
class EvaluationInputs:
    """``synthetic`` holds every selected clip of a level (calib ones only feed the per-scene
    Kendall of ``levels``); ``real`` maps a real dataset to its rows at the same level."""

    synthetic: LevelData
    real: Mapping[str, LevelData]


@dataclass(frozen=True)
class PrimaryScope:
    """The H1 primary scenes (ADR 0015 1.4) and the reference of the clips they select."""

    scope: SceneScope
    dataset: DatasetRef


@dataclass(frozen=True)
class EvaluationMeta:
    """``profile`` is the sampling profile the evaluated clips were run with; ``primary`` adds
    the H1 primary view next to the all-scenes strata."""

    created_utc: str
    run_id: str
    artifact_path: str
    dataset: DatasetRef
    profile: str
    real_sets: tuple[DatasetRef, ...] = ()
    primary: PrimaryScope | None = None


def evaluate(
    calibration: Calibration,
    inputs: Sequence[EvaluationInputs],
    meta: EvaluationMeta,
    config: CalibrationConfig,
) -> EvaluationReport:
    """Applies the strata of the run's level and profile; raises CalibrationError when there is
    none or when the data's row_step differs from the stratum's."""
    matched = [
        (_matching(calibration.stratum(level, profile), item, meta.profile), item)
        for item in inputs
        for level, profile in calibration.strata()
        if level is item.synthetic.level and profile == meta.profile
    ]
    if not matched:
        raise CalibrationError(
            f"the artifact has no stratum of profile {meta.profile!r} at the evaluated levels"
        )
    return EvaluationReport(
        created_utc=meta.created_utc,
        run_id=meta.run_id,
        artifact_path=meta.artifact_path,
        artifact_sha256=calibration.sha256,
        artifact_run_id=calibration.run_id,
        dataset=meta.dataset,
        real_datasets=tuple(sorted({name for item in inputs for name in item.real})),
        real_sets=meta.real_sets,
        strata=tuple(_stratum(stratum, item, config) for stratum, item in matched),
        h1_primary=_primary(matched, meta.primary, config),
    )


def _primary(
    matched: Sequence[tuple[Stratum, EvaluationInputs]],
    primary: PrimaryScope | None,
    config: CalibrationConfig,
) -> H1PrimaryView | None:
    if primary is None or not primary.scope.primary_scenes:
        return None
    keep = primary.scope.indices
    return H1PrimaryView(
        scenes=primary.scope.primary_scenes,
        dataset=primary.dataset,
        strata=tuple(_stratum(stratum, item, config, keep=keep) for stratum, item in matched),
    )


def _matching(stratum: Stratum, item: EvaluationInputs, profile: str) -> Stratum:
    named = {"synthetic": item.synthetic, **item.real}
    for name, data in sorted(named.items()):
        problem = stratum.time_base_problem(profile, unique_row_step(data.rows))
        if problem is not None:
            raise CalibrationError(f"{name} clips: {problem}")
    return stratum


def _rule(stratum: Stratum, check_id: str) -> Rule:
    return rule_for(stratum, check_id, policy_for(stratum))


def _stratum(
    stratum: Stratum,
    item: EvaluationInputs,
    config: CalibrationConfig,
    keep: frozenset[int] | None = None,
) -> EvaluationStratum:
    """``keep``: only the scenes of these indices, and no real video (the primary view)."""
    rows = [r for r in item.synthetic.rows if keep is None or in_scope(r.scene_id, keep)]
    real: Mapping[str, LevelData] = item.real if keep is None else {}
    heldout = [r for r in rows if r.split is not Split.CALIB]
    check_ids = sorted(stratum.model.checks)
    return EvaluationStratum(
        level=stratum.level,
        profile=stratum.profile,
        checks=tuple(heldout_check(stratum, c, heldout, config) for c in check_ids),
        scene_kendall=_kendall(stratum, rows, check_ids),
        real=tuple(
            RealFpr(dataset=name, check_id=c, fpr=_real_fpr(stratum, c, data.rows, config))
            for name, data in sorted(real.items())
            for c in check_ids
        ),
        real_clip_flags=tuple(
            RealFpr(dataset=name, check_id=SCORE_CHECK, fpr=clip_flags(stratum, data, config))
            for name, data in sorted(real.items())
        ),
    )


def heldout_check(
    stratum: Stratum, check_id: str, rows: Sequence[CalibrationRow], config: CalibrationConfig
) -> HeldoutCheck:
    check = stratum.model.checks[check_id]
    mine = [r for r in rows if r.check_id == check_id]
    rule = _rule(stratum, check_id)
    off = [r for r in mine if r.role is Role.OFF_TARGET and r.scored]
    clean = [r for r in mine if r.role is Role.CLEAN]
    require_one_clean_per_scene(clean)
    total, decided = false_positive_rates(
        [verdict_of(r, rule, check.tau) for r in clean if r.scored], config.calibration.ci
    )
    return HeldoutCheck(
        check_id=check_id,
        status=check.status,
        tau=check.tau,
        fpr_clean=total,
        fpr_clean_decided=decided,
        fpr_off_target=off_target_rate(off, rule, check.tau, config.calibration),
        tpr_by_level=tpr_by_level(
            [r for r in mine if r.role is Role.POSITIVE], rule, check.tau, config.calibration.ci
        ),
    )


def _real_fpr(
    stratum: Stratum, check_id: str, rows: Sequence[CalibrationRow], config: CalibrationConfig
) -> Rate:
    """Real clips are all negatives; takes of one scene are clustered."""
    check = stratum.model.checks[check_id]
    rule = _rule(stratum, check_id)
    mine = [r for r in rows if r.check_id == check_id and r.scored]
    flags = [(r.scene_id, fails(r, rule, check.tau)) for r in mine]
    cal = config.calibration
    return clustered_rate(flags, cal.bootstrap, cal.seed, cal.ci)


def _kendall(
    stratum: Stratum, rows: Sequence[CalibrationRow], check_ids: Sequence[str]
) -> tuple[SceneKendall, ...]:
    found: list[SceneKendall] = []
    for check_id in check_ids:
        mine = [r for r in rows if r.check_id == check_id]
        if mine:
            found += scene_kendall(mine, stratum.sigma_correction(check_id)[0])
    return tuple(found)


def clip_flags(stratum: Stratum, data: LevelData, config: CalibrationConfig) -> Rate:
    """Clips whose 1 - S exceeds s*, over the clips with a score (clustered by scene)."""
    threshold = stratum.score_threshold()
    scene = {r.clip_id: r.scene_id for r in data.rows}
    by_clip: dict[str, list[CheckResult]] = {}
    for result in data.results:
        by_clip.setdefault(result.clip_id, []).append(result)
    flags: list[tuple[str, bool]] = []
    for clip_id, results in sorted(by_clip.items()):
        score = score_clip(results, stratum).score
        if score is not None and threshold is not None:
            flags.append((scene.get(clip_id, clip_id), 1.0 - score > threshold))
    cal = config.calibration
    return clustered_rate(flags, cal.bootstrap, cal.seed, cal.ci)
