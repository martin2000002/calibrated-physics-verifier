"""The steps of the ``calibrate`` leaves over explicitly injected dependencies."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.calibration.artifact import DatasetRef, StageFactor, StageSigma
from physverify.calibration.artifact_io import (
    Calibration,
    CalibrationError,
    Stratum,
    UnusableStratumError,
    load_calibration,
    open_stratum,
    projection_name,
)
from physverify.calibration.collect import (
    SubjectLookup,
    level_data,
    read_run,
    run_level_of,
    sigma_pairs,
)
from physverify.calibration.evaluate import EvaluationInputs, PrimaryScope
from physverify.calibration.fit import FitMeta, LevelData
from physverify.calibration.report import (
    CALIBRATION_MD,
    CURVES_CSV,
    calibration_markdown,
    curves_csv,
)
from physverify.calibration.settings import CalibrationConfig
from physverify.calibration.sigma_recalibration import STAGE_SIGMA_FILE
from physverify.calibration.summary import factor_lines, show_lines
from physverify.config.models import Settings
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import PhysverifyError, UsageError
from physverify.datasets._base import DatasetSource, select_clips
from physverify.diagnosis.error_profile_writer import write_profile
from physverify.diagnosis.report import dataset_versions, profile_hash
from physverify.diagnosis.settings import diagnosis_settings
from physverify.extraction.error_profile import load_error_profile
from physverify.runtime.runs import RunContext
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.io import stable_hash
from physverify.synth.scene_scope import SceneScope, h1_scope, in_scope, scene_of
from physverify.verification.inputs import InputSource, OracleSource
from physverify.verification.runner import (
    UNKNOWN_ROW_STEP,
    VerificationReport,
    VerificationRunner,
)
from physverify.verification.sweep import CheckUnit

BOTH = "both"

Echo = Callable[[str], None]
RealSources = dict[str, tuple[DatasetSource, tuple[ClipManifest, ...]]]


@dataclass(frozen=True)
class Session:
    """One ``calibrate`` leaf: its run and what it reads. The factories are called when needed,
    so a leaf that never runs a check never builds a runner nor hashes the checks."""

    settings: Settings
    config: CalibrationConfig
    run: RunContext
    raw_runner: Callable[[], VerificationRunner]
    check_units: Callable[[], tuple[CheckUnit, ...]]
    input_source: Callable[[DatasetSource, InputLevel], InputSource]
    subject_lookup: Callable[[DatasetSource], SubjectLookup]
    dataset: Callable[[str], DatasetSource]
    checks_hash: Callable[[], str]
    echo: Echo
    err: Echo


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)


def levels_of(level: str | None, session: Session) -> tuple[InputLevel, ...]:
    if level is None:
        return session.config.calibration.levels
    if level == BOTH:
        return (InputLevel.ORACLE, InputLevel.PERCEIVED)
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError(f"--level must be oracle, perceived or {BOTH}")
    return (InputLevel(level),)


def dataset_ref(
    dataset: DatasetSource, subset: str, split: str, clips: Sequence[ClipManifest]
) -> DatasetRef:
    ids = sorted(clip.clip_id for clip in clips)
    return DatasetRef(
        name=dataset.name,
        version=dataset_versions(dataset, clips).get(dataset.name, ""),
        subset=subset,
        split=split,
        n_clips=len(ids),
        n_scenes=len({scene_of(clip) for clip in clips}),
        clip_ids_hash=stable_hash(ids),
    )


def artifact_at(session: Session, path: Path, *, stale_check: bool = True) -> Calibration:
    if not path.is_file():
        raise UsageError(f"no calibration artifact at {path}: run `physverify calibrate fit`")
    return load_calibration(path, checks_hash=session.checks_hash() if stale_check else None)


# --- fit and devscore ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reports:
    by_level: dict[InputLevel, VerificationReport]
    clips: tuple[ClipManifest, ...]


def live_reports(
    session: Session,
    dataset: DatasetSource,
    clips: Sequence[ClipManifest],
    levels: Sequence[InputLevel],
) -> Reports:
    runner = session.raw_runner()
    wanted = set(levels) | ({InputLevel.ORACLE} if InputLevel.PERCEIVED in levels else set())
    reports = {lv: runner.run(session.input_source(dataset, lv), clips) for lv in sorted(wanted)}
    return Reports(reports, tuple(clips))


def run_reports(session: Session, dataset: DatasetSource, run_ids: Sequence[str]) -> Reports:
    """The runs read back, one per level; ``with_oracle`` completes a perceived-only set."""
    runs_dir = session.settings.paths.runs_dir
    every = dataset.clips(None)
    reports = {}
    seen: dict[InputLevel, str] = {}
    for run_id in run_ids:
        report = read_run(runs_dir / run_id, every)
        level = run_level_of(report)
        if level in seen:
            raise UsageError(
                f"--from-run {seen[level]} and {run_id} are both {level.value} runs: "
                "pass one per level"
            )
        seen[level] = run_id
        reports[level] = report
    return Reports(reports, next(iter(reports.values())).clips)


def with_oracle(session: Session, dataset: DatasetSource, reports: Reports) -> Reports:
    """Run after the sealed guard: the oracle a perceived-only set lacks."""
    found = dict(reports.by_level)
    if InputLevel.PERCEIVED in found and InputLevel.ORACLE not in found:
        found[InputLevel.ORACLE] = session.raw_runner().run(OracleSource(dataset), reports.clips)
    return Reports(found, reports.clips)


def fit_level_data(
    session: Session,
    dataset: DatasetSource,
    reports: Reports,
    levels: Sequence[InputLevel],
    no_sigma: bool,
) -> tuple[LevelData, ...]:
    units, config = session.check_units(), session.config
    found: list[LevelData] = []
    for level in levels:
        pairs = None
        if level is InputLevel.PERCEIVED and not no_sigma:
            measured = sigma_pairs(
                reports.by_level[InputLevel.ORACLE],
                reports.by_level[level],
                session.subject_lookup(dataset),
                config.min_span_iou,
            )
            session.echo(
                f"sigma pairs: {len(measured.pairs)} "
                f"({len(measured.unpaired_clips)} clips without a subject map)"
            )
            pairs = measured.pairs
        found.append(
            level_data(
                level, reports.by_level[level], units, config, split=Split.CALIB, pairs=pairs
            )
        )
    return tuple(found)


def _stage_sigma(session: Session) -> StageSigma:
    path = session.settings.paths.calibration_dir / STAGE_SIGMA_FILE
    if not path.is_file():
        return StageSigma(path=None, hash=None)
    return StageSigma.model_validate_json(path.read_text(encoding="utf-8"))


def fit_meta(
    session: Session, dataset: DatasetSource, subset: str, clips: Sequence[ClipManifest]
) -> FitMeta:
    settings = session.settings
    return FitMeta(
        created_utc=session.run.created_utc,
        run_id=session.run.run_id,
        commit=session.run.commit,
        config_hash=settings.config_hash(),
        checks_hash=session.checks_hash(),
        error_profile_hash=profile_hash(load_error_profile(settings.paths.error_profile)),
        profile=settings.profile,
        dataset=dataset_ref(dataset, subset, Split.CALIB.value, clips),
        stage_sigma=_stage_sigma(session),
    )


def fit_clips(source: DatasetSource, subset: str, wanted: Sequence[str]) -> list[ClipManifest]:
    """The subset's calib clips; explicit ``--clips`` are taken as given (a non-calib one fails)."""
    if wanted:
        return select_clips(source.clips(None), wanted)
    chosen = [c for c in source.clips(subset) if c.split is Split.CALIB]
    if not chosen:
        raise UsageError(f"subset {subset!r} of {source.name} has no split=calib clip")
    return chosen


def require_calib(clips: Sequence[ClipManifest]) -> None:
    """Checked before any check runs: devscore is a calib-only metric."""
    wrong = sorted(c.clip_id for c in clips if c.split is not Split.CALIB)
    if wrong:
        raise UsageError(f"devscore runs on split=calib only; not calib: {wrong[:5]}")


def publish_fit(session: Session, path: Path, text: str) -> None:
    write(path, text)
    calibration = load_calibration(path)
    root = session.settings.paths.calibration_dir
    for lv, profile in calibration.strata():
        stratum = calibration.stratum(lv, profile)
        projection = path.parent / projection_name(lv, profile)
        write(projection, projection_text(stratum, root))
    run_dir = session.run.run_dir
    write(run_dir / path.name, text)
    write(run_dir / CALIBRATION_MD, calibration_markdown(calibration.artifact))
    write(run_dir / CURVES_CSV, curves_csv(calibration.artifact))
    session.echo(f"artifact: {path} ({calibration.status}, sha256 {calibration.sha256[:12]})")
    for line in show_lines(calibration):
        session.echo(line)


# --- stage-sigma --------------------------------------------------------------------------------


def write_stage_sigma(
    session: Session, profile: ErrorProfile, factors: Sequence[StageFactor]
) -> None:
    directory = session.settings.paths.calibration_dir
    path = directory / session.config.calibration.stage_profile
    directory.mkdir(parents=True, exist_ok=True)
    write_profile(path, profile)
    summary = StageSigma(path=str(path), hash=profile_hash(profile), entries=tuple(factors))
    write(directory / STAGE_SIGMA_FILE, summary.to_json())
    write(session.run.run_dir / STAGE_SIGMA_FILE, summary.to_json())
    session.echo(f"profile: {path}")
    for line in factor_lines(factors):
        session.echo(line)


# --- evaluate and publish -----------------------------------------------------------------------


def primary_scope(
    session: Session, source: DatasetSource, subset: str, clips: Sequence[ClipManifest]
) -> PrimaryScope | None:
    scope = SceneScope(primary_scenes=h1_scope(session.settings).scenes_for(source.name))
    if not scope.primary_scenes:
        return None
    kept = [c for c in clips if in_scope(scene_of(c), scope.indices)]
    return PrimaryScope(scope=scope, dataset=dataset_ref(source, subset, "all", kept))


def evaluation_inputs(
    session: Session,
    source: DatasetSource,
    clips: Sequence[ClipManifest],
    levels: Sequence[InputLevel],
    sources: RealSources,
) -> tuple[EvaluationInputs, ...]:
    return tuple(
        EvaluationInputs(_eval_data(session, source, clips, lv), _real_data(session, lv, sources))
        for lv in levels
    )


def _eval_data(
    session: Session, source: DatasetSource, clips: Sequence[ClipManifest], level: InputLevel
) -> LevelData:
    report = session.raw_runner().run(session.input_source(source, level), clips)
    return level_data(level, report, session.check_units(), session.config, split=None, pairs=None)


def real_sources(session: Session, named: Sequence[str] | None = None) -> RealSources:
    """The real datasets with their clips of the ``diagnosis.subset``: the ``named`` ones, which
    must all resolve, or else those of ``calibration.real_datasets`` that resolve."""
    subset = diagnosis_settings(session.settings).subset
    found: RealSources = {}
    for name in dict.fromkeys(named or session.config.calibration.real_datasets):
        try:
            source = session.dataset(name)
            found[name] = (source, tuple(source.clips(subset)))
        except PhysverifyError as exc:
            if named:
                raise
            session.echo(f"real dataset {name} skipped: {exc}")
    return found


def _real_data(session: Session, level: InputLevel, sources: RealSources) -> dict[str, LevelData]:
    """Real video has no ground truth: only the perceived stratum is scored on it."""
    if level is InputLevel.ORACLE:
        return {}
    found: dict[str, LevelData] = {}
    for name, (source, clips) in sources.items():
        try:
            found[name] = _eval_data(session, source, clips, level)
        except PhysverifyError as exc:
            session.echo(f"real dataset {name} skipped: {exc}")
    return found


def projection_text(stratum: Stratum, root: Path) -> str:
    return json.dumps(stratum.thresholds_projection(root), indent=2, sort_keys=True) + "\n"


def publishable(
    session: Session, path: Path, level: InputLevel, profile: str | None, *, force: bool
) -> Stratum:
    """The stratum through the consumers' gate; ``force`` publishes one that is not usable."""
    name = profile or session.settings.profile
    try:
        return open_stratum(
            path,
            level,
            name,
            checks_hash=session.checks_hash(),
            row_step=UNKNOWN_ROW_STEP,
            require_calibrated=session.config.calibration.require_calibrated_artifact,
        )
    except UnusableStratumError as exc:
        if not force:
            raise UsageError(f"{exc}: refit on more scenes or pass --force") from exc
        session.err(f"warning: publishing a stratum that is not usable ({exc})")
    except CalibrationError as exc:
        raise UsageError(str(exc)) from exc
    return load_calibration(path).stratum(level, name)
