"""``physverify calibrate``: fit, evaluate and publish the H1 calibration (ADR 0009 D13).

Not heavy: the checks read cached stage outputs and refuse the lock. ``stage-sigma`` takes the
heavy lock lazily, only when a stage has to be computed.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from physverify.calibration.collect import (
    SubjectLookup,
    Subjects,
    level_data,
    read_run,
    run_level_of,
    sigma_pairs,
)
from physverify.calibration.evaluate import EvaluationInputs, EvaluationMeta, evaluate
from physverify.calibration.fit import FitInputs, FitMeta, LevelData, fit_calibration
from physverify.calibration.loader import (
    Calibration,
    Stratum,
    default_artifact_path,
    load_calibration,
    projection_name,
)
from physverify.calibration.report import (
    CALIBRATION_MD,
    CURVES_CSV,
    EVALUATION_MD,
    calibration_markdown,
    curves_csv,
    evaluation_markdown,
)
from physverify.calibration.schema import DatasetRef, StageFactor, StageSigma
from physverify.calibration.settings import (
    CalibrationConfig,
    calibration_config,
    checks_hash,
)
from physverify.calibration.sigma import factor_map, recalibrate_profile, stage_factors
from physverify.calibration.table import scene_of
from physverify.cli.commands.extract import (
    ConsoleObserver,
    EventRecorder,
    load_error_profile,
    pipeline_deps,
    resolve_dataset,
    select_clips,
)
from physverify.cli.context import AppContext, app_context
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.atomic_io import write_text_atomic
from physverify.diagnosis.chain import ChainRunner, warm_variants
from physverify.diagnosis.end_to_end import track_of
from physverify.diagnosis.error_profile_writer import write_profile
from physverify.diagnosis.pairing import subject_map
from physverify.diagnosis.plans import oracle_variant, perceived_variant
from physverify.diagnosis.report import dataset_versions, profile_hash
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import measure, pair_variants, stage_pairs
from physverify.extraction.plan import resolve_plan
from physverify.runtime.runs import RunContext, current_commit
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.io import stable_hash
from physverify.verification.inputs import (
    InputSource,
    OracleSource,
    PerceivedSource,
    PipelineOutputLoader,
    refuse_heavy,
)
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import (
    CALIBRATED,
    CHECKS_SECTION,
    ChecksSettings,
    VerificationReport,
    VerificationRunner,
    build_checks,
)
from physverify.verification.sweep import CheckUnit

NAME = "calibrate"
OWNER = "A8"
IMPLEMENTED = True
BOTH = "both"
STAGE_SIGMA_FILE = "stage_sigma.json"
EVALUATION_FILE = "evaluation.json"

app = typer.Typer(
    help="H1 calibration: thresholds, rates, detection limits and the clip score.",
    no_args_is_help=True,
)

DatasetOption = Annotated[str | None, typer.Option(help="Dataset (default: calibration.dataset).")]
SubsetOption = Annotated[str | None, typer.Option(help="Subset (default: calibration.subset).")]
ClipsOption = Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")]
ArtifactOption = Annotated[
    Path | None, typer.Option(help="Artifact (default: calibration_dir/artifact).")
]
ProfileOption = Annotated[str | None, typer.Option(help="Sampling profile (default: active).")]


# --- shared -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    context: AppContext
    config: CalibrationConfig
    run: RunContext


def _session() -> Session:
    context = app_context()
    settings = context.settings
    run = RunContext.create(
        settings.paths.runs_dir,
        argv=context.invocation.argv,
        device=settings.runtime.device,
        profile=settings.profile,
        config_hash=settings.config_hash(),
        commit=current_commit(settings.repo_root),
        now=dt.datetime.now(dt.UTC),
    )
    typer.echo(f"run: {run.run_dir}")
    return Session(context, calibration_config(settings), run)


def _overrides(session: Session) -> ChecksSettings:
    return session.context.settings.section(CHECKS_SECTION, ChecksSettings)


def _raw_runner(session: Session) -> VerificationRunner:
    registry = session.context.registry("checks")
    return VerificationRunner(build_checks(registry, _overrides(session)), RawPolicy(), {})


def _check_units(session: Session) -> tuple[CheckUnit, ...]:
    registry = session.context.registry("checks")
    return tuple(CheckUnit(c, registry[c].magnitude_unit) for c in sorted(registry))


def _checks_hash(session: Session) -> str:
    return checks_hash(session.context.registry("checks"), _overrides(session))


def _source(session: Session, dataset: DatasetSource, level: InputLevel) -> InputSource:
    if level is InputLevel.ORACLE:
        return OracleSource(dataset)
    context = session.context
    plan = resolve_plan(context.settings, context.registry("providers"), dataset=dataset.name)
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset.name))
    return PerceivedSource(dataset, PipelineOutputLoader(plan, deps), level)


def _subject_lookup(session: Session, dataset: DatasetSource) -> SubjectLookup:
    context = session.context
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset.name))
    chain = ChainRunner(context.settings, context.registry("providers"), deps, dataset)

    def lookup(clip: ClipManifest) -> Subjects | None:
        oracle, perceived = (
            chain.outputs(clip, oracle_variant()),
            chain.outputs(clip, perceived_variant()),
        )
        if oracle is None or perceived is None:
            return None
        try:
            return Subjects(subject_map(track_of(perceived), track_of(oracle)))
        except MissingInputError:
            return None

    return lookup


def _levels(level: str | None, session: Session) -> tuple[InputLevel, ...]:
    if level is None:
        return session.config.calibration.levels
    if level == BOTH:
        return (InputLevel.ORACLE, InputLevel.PERCEIVED)
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError(f"--level must be oracle, perceived or {BOTH}")
    return (InputLevel(level),)


def _dataset_ref(
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


def _artifact(session: Session, artifact: Path | None, *, stale_check: bool = True) -> Calibration:
    path = artifact or default_artifact_path(session.context.settings)
    if not path.is_file():
        raise UsageError(f"no calibration artifact at {path}: run `physverify calibrate fit`")
    return load_calibration(path, checks_hash=_checks_hash(session) if stale_check else None)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)


# --- fit ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reports:
    by_level: dict[InputLevel, VerificationReport]
    clips: tuple[ClipManifest, ...]


def _live_reports(
    session: Session,
    dataset: DatasetSource,
    clips: Sequence[ClipManifest],
    levels: Sequence[InputLevel],
) -> Reports:
    runner = _raw_runner(session)
    wanted = set(levels) | ({InputLevel.ORACLE} if InputLevel.PERCEIVED in levels else set())
    reports = {lv: runner.run(_source(session, dataset, lv), clips) for lv in sorted(wanted)}
    return Reports(reports, tuple(clips))


def _run_reports(session: Session, dataset: DatasetSource, run_ids: Sequence[str]) -> Reports:
    runs_dir = session.context.settings.paths.runs_dir
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
    clips = next(iter(reports.values())).clips
    if InputLevel.PERCEIVED in reports and InputLevel.ORACLE not in reports:
        reports[InputLevel.ORACLE] = _raw_runner(session).run(OracleSource(dataset), clips)
    return Reports(reports, clips)


def _level_data(
    session: Session,
    dataset: DatasetSource,
    reports: Reports,
    levels: Sequence[InputLevel],
    no_sigma: bool,
) -> tuple[LevelData, ...]:
    units, config = _check_units(session), session.config
    found: list[LevelData] = []
    for level in levels:
        pairs = None
        if level is InputLevel.PERCEIVED and not no_sigma:
            measured = sigma_pairs(
                reports.by_level[InputLevel.ORACLE],
                reports.by_level[level],
                _subject_lookup(session, dataset),
                config.min_span_iou,
            )
            typer.echo(
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
    path = session.context.settings.paths.calibration_dir / STAGE_SIGMA_FILE
    if not path.is_file():
        return StageSigma(path=None, hash=None)
    return StageSigma.model_validate_json(path.read_text(encoding="utf-8"))


def _fit_meta(
    session: Session, dataset: DatasetSource, subset: str, clips: Sequence[ClipManifest]
) -> FitMeta:
    settings = session.context.settings
    return FitMeta(
        created_utc=session.run.created_utc,
        run_id=session.run.run_id,
        commit=session.run.commit,
        config_hash=settings.config_hash(),
        checks_hash=_checks_hash(session),
        error_profile_hash=profile_hash(load_error_profile(settings.paths.error_profile)),
        profile=settings.profile,
        dataset=_dataset_ref(dataset, subset, Split.CALIB.value, clips),
        stage_sigma=_stage_sigma(session),
    )


@app.command("fit", help="Fit thresholds, rates and weights on split=calib (D5-D11).")
def fit(
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    level: Annotated[str | None, typer.Option(help="oracle, perceived or both.")] = None,
    clips: ClipsOption = None,
    from_run: Annotated[list[str] | None, typer.Option(help="verify --policy raw run ids.")] = None,
    out: Annotated[Path | None, typer.Option(help="Artifact path.")] = None,
    no_sigma: Annotated[bool, typer.Option(help="Do not recalibrate the perceived sigma.")] = False,
) -> None:
    session = _session()
    cal = session.config.calibration
    source = resolve_dataset(session.context, dataset or cal.dataset)
    name = subset or cal.subset
    levels: tuple[InputLevel, ...]
    if from_run:
        reports = _run_reports(session, source, from_run)
        levels = tuple(
            lv for lv in (InputLevel.ORACLE, InputLevel.PERCEIVED) if lv in reports.by_level
        )
    else:
        levels = _levels(level, session)
        chosen = _fit_clips(source, name, clips or ())
        reports = _live_reports(session, source, chosen, levels)
    data = _level_data(session, source, reports, levels, no_sigma)
    meta = _fit_meta(session, source, name, reports.clips)
    artifact = fit_calibration(FitInputs(data, _check_units(session), meta), session.config)
    path = out or default_artifact_path(session.context.settings)
    _publish_fit(session, path, artifact.to_json())


def _fit_clips(source: DatasetSource, subset: str, wanted: Sequence[str]) -> list[ClipManifest]:
    """The subset's calib clips; explicit ``--clips`` are taken as given (a non-calib one fails)."""
    if wanted:
        return select_clips(source.clips(None), wanted)
    chosen = [c for c in source.clips(subset) if c.split is Split.CALIB]
    if not chosen:
        raise UsageError(f"subset {subset!r} of {source.name} has no split=calib clip")
    return chosen


def _publish_fit(session: Session, path: Path, text: str) -> None:
    _write(path, text)
    calibration = load_calibration(path)
    root = session.context.settings.paths.calibration_dir
    for lv, profile in calibration.strata():
        stratum = calibration.stratum(lv, profile)
        projection = path.parent / projection_name(lv, profile)
        _write(projection, _projection_text(stratum, root))
    run_dir = session.run.run_dir
    _write(run_dir / path.name, text)
    _write(run_dir / CALIBRATION_MD, calibration_markdown(calibration.artifact))
    _write(run_dir / CURVES_CSV, curves_csv(calibration.artifact))
    typer.echo(f"artifact: {path} ({calibration.status}, sha256 {calibration.sha256[:12]})")
    for line in show_lines(calibration):
        typer.echo(line)


# --- stage-sigma --------------------------------------------------------------------------------


@app.command("stage-sigma", help="D10: per-stage sigma factors (heavy, lazy lock; H200).")
def stage_sigma(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    from physverify.cli.commands.diagnose import diagnosis_context

    session = _session()
    cal = session.config.calibration
    recorder = EventRecorder(ConsoleObserver(typer.echo))
    ctx = diagnosis_context(
        session.context, recorder, None, dataset=dataset or cal.dataset, subset=subset or cal.subset
    )
    source = ctx.dataset()
    calib = [
        c
        for c in select_clips(source.clips(ctx.diag.subset), clips or ())
        if c.split is Split.CALIB
    ]
    if not calib:
        raise UsageError(
            f"no split=calib clip in subset {ctx.diag.subset!r}: stage-sigma fits on calib only"
        )
    profile = load_error_profile(ctx.settings.paths.error_profile)
    if profile is None:
        raise UsageError(
            f"no error profile at {ctx.settings.paths.error_profile}: run `diagnose stages`"
        )
    runner = ctx.chain(source).with_error_profile(None)
    pairs = stage_pairs(ctx.diag)
    warm_variants(runner, calib, pair_variants(pairs), skip_failed=False)
    samples = measure(runner, calib, ctx.metrics, pairs, ctx.diag.metrics)
    factors = stage_factors(samples, {c.clip_id: scene_of(c) for c in calib}, cal)
    _write_stage_sigma(session, recalibrate_profile(profile, factor_map(factors)), factors)


def _write_stage_sigma(
    session: Session, profile: ErrorProfile, factors: Sequence[StageFactor]
) -> None:
    directory = session.context.settings.paths.calibration_dir
    path = directory / session.config.calibration.stage_profile
    directory.mkdir(parents=True, exist_ok=True)
    write_profile(path, profile)
    summary = StageSigma(path=str(path), hash=profile_hash(profile), entries=tuple(factors))
    _write(directory / STAGE_SIGMA_FILE, summary.to_json())
    _write(session.run.run_dir / STAGE_SIGMA_FILE, summary.to_json())
    typer.echo(f"profile: {path}")
    for f in factors:
        before = next((c.observed for c in f.coverage_before if c.k == 1.0), None)
        after = next((c.observed for c in f.coverage_after if c.k == 1.0), None)
        typer.echo(
            f"{f.stage}/{f.provider} {f.quantity}: factor {_num(f.factor)} n {f.n} "
            f"coverage@1 before {_num(before)} after (out of fold) {_num(after)}"
        )


# --- evaluate / publish / show ------------------------------------------------------------------


@app.command("evaluate", help="Score held-out and real clips with the frozen artifact (D12).")
def evaluate_command(
    artifact: ArtifactOption = None,
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
    real: Annotated[bool, typer.Option(help="Also score the real datasets.")] = True,
) -> None:
    session = _session()
    calibration = _artifact(session, artifact)
    cal = session.config.calibration
    source = resolve_dataset(session.context, dataset or cal.dataset)
    name = subset or cal.subset
    chosen = select_clips(source.clips(name), clips or ())
    levels = sorted({lv for lv, _ in calibration.strata()})
    inputs = tuple(
        EvaluationInputs(
            _eval_data(session, source, chosen, lv), _real_data(session, lv) if real else {}
        )
        for lv in levels
    )
    meta = EvaluationMeta(
        created_utc=session.run.created_utc,
        run_id=session.run.run_id,
        artifact_path=str(calibration.path),
        dataset=_dataset_ref(source, name, "all", chosen),
        profile=session.context.settings.profile,
    )
    report = evaluate(calibration, inputs, meta, session.config)
    _write(session.run.run_dir / EVALUATION_FILE, report.to_json())
    _write(session.run.run_dir / EVALUATION_MD, evaluation_markdown(report))
    typer.echo(f"evaluation: {session.run.run_dir / EVALUATION_FILE}")


def _eval_data(
    session: Session, source: DatasetSource, clips: Sequence[ClipManifest], level: InputLevel
) -> LevelData:
    report = _raw_runner(session).run(_source(session, source, level), clips)
    return level_data(level, report, _check_units(session), session.config, split=None, pairs=None)


def _real_data(session: Session, level: InputLevel) -> dict[str, LevelData]:
    """Real video has no ground truth: only the perceived stratum is scored on it."""
    if level is InputLevel.ORACLE:
        return {}
    subset = diagnosis_settings(session.context.settings).subset
    found: dict[str, LevelData] = {}
    for name in session.config.calibration.real_datasets:
        try:
            source = resolve_dataset(session.context, name)
            found[name] = _eval_data(session, source, source.clips(subset), level)
        except PhysverifyError as exc:
            typer.echo(f"real dataset {name} skipped: {exc}")
    return found


def _projection_text(stratum: Stratum, root: Path) -> str:
    return json.dumps(stratum.thresholds_projection(root), indent=2, sort_keys=True) + "\n"


@app.command("publish", help="Copy a stratum's projection to paths.thresholds (verify uses it).")
def publish(
    level: Annotated[str, typer.Option(help="oracle or perceived.")],
    profile: ProfileOption = None,
    artifact: ArtifactOption = None,
    force: Annotated[
        bool, typer.Option(help="Publish an artifact that is not calibrated (warns).")
    ] = False,
) -> None:
    session = _session()
    calibration = _artifact(session, artifact)
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError("--level must be oracle or perceived")
    if calibration.status != CALIBRATED:
        if not force:
            raise UsageError(
                f"{calibration.path} is {calibration.status}, not {CALIBRATED}: refit on more "
                "scenes or pass --force"
            )
        typer.echo(
            f"warning: publishing an artifact that is {calibration.status}; "
            "only its ok checks decide",
            err=True,
        )
    stratum = calibration.stratum(InputLevel(level), profile or session.context.settings.profile)
    settings = session.context.settings
    target = settings.paths.thresholds
    _write(target, _projection_text(stratum, settings.paths.calibration_dir))
    typer.echo(f"thresholds: {target} ({stratum.source})")


@app.command("show", help="Summary of the artifact per stratum and check.")
def show(
    artifact: ArtifactOption = None,
    level: Annotated[str | None, typer.Option()] = None,
    profile: ProfileOption = None,
) -> None:
    context = app_context()
    path = artifact or default_artifact_path(context.settings)
    if not path.is_file():
        raise UsageError(f"no calibration artifact at {path}: run `physverify calibrate fit`")
    for line in show_lines(load_calibration(path), level, profile):
        typer.echo(line)


def show_lines(
    calibration: Calibration, level: str | None = None, profile: str | None = None
) -> list[str]:
    lines = [f"calibration {calibration.run_id} v{calibration.version} {calibration.status}"]
    for lv, prof in calibration.strata():
        if (level and lv.value != level) or (profile and prof != profile):
            continue
        stratum = calibration.stratum(lv, prof)
        lines.append(f"{lv.value}/{prof}: s* {_num(stratum.score_threshold())}")
        for check_id, c in sorted(stratum.model.checks.items()):
            lines.append(
                f"  {check_id} {c.status} tau {_num(c.tau)} {c.unit} clean {c.n_clean_scenes} "
                f"FPR cv {_num(c.fpr.cv.value)} "
                f"[{_num(c.fpr.cv.ci.low)}, {_num(c.fpr.cv.ci.high)}] "
                f"w {_num(c.weight.value)}"
            )
    return lines


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"
