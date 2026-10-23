"""``physverify calibrate``: fit, evaluate and publish the H1 calibration (ADR 0009 D13).

Not heavy: the checks read cached stage outputs and refuse the lock. ``stage-sigma`` takes the
heavy lock lazily, only when a stage has to be computed.
"""

from __future__ import annotations

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
from physverify.calibration.devscore import (
    DEVSCORE_FILE,
    DevScoreReport,
    LevelDevScore,
    dev_score,
    require_bootstrap,
)
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
)
from physverify.calibration.fit import FitInputs, FitMeta, LevelData, fit_calibration
from physverify.calibration.loader import (
    Calibration,
    CalibrationError,
    Stratum,
    UnusableStratumError,
    default_artifact_path,
    load_calibration,
    open_stratum,
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
from physverify.calibration.scenes import SceneScope, h1_scope, in_scope
from physverify.calibration.schema import DatasetRef, StageFactor, StageSigma
from physverify.calibration.settings import (
    CalibrationConfig,
    active_checks_hash,
    calibration_config,
)
from physverify.calibration.sigma import factor_map, recalibrate_profile, stage_factors
from physverify.calibration.table import scene_of
from physverify.cli.commands._sealed import Selection, guard_sealed
from physverify.cli.commands.extract import (
    ConsoleObserver,
    EventRecorder,
    load_error_profile,
    pipeline_deps,
    resolve_dataset,
    select_clips,
)
from physverify.cli.context import AppContext, app_context
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.core.markdown import number
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain import ChainRunner, warm_variants
from physverify.diagnosis.end_to_end import track_of
from physverify.diagnosis.error_profile_writer import write_profile
from physverify.diagnosis.pairing import subject_map
from physverify.diagnosis.plans import oracle_variant, perceived_variant
from physverify.diagnosis.report import dataset_versions, profile_hash
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import measure, pair_variants, stage_pairs
from physverify.extraction.plan import resolve_plan
from physverify.runtime.runs import RunContext, new_run
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
    CHECKS_SECTION,
    UNKNOWN_ROW_STEP,
    ChecksSettings,
    VerificationReport,
    VerificationRunner,
    build_checks,
)
from physverify.verification.sweep import CheckUnit

NAME = "calibrate"
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
    run = new_run(settings, context.invocation.argv)
    typer.echo(f"run: {run.run_dir}")
    return Session(context, calibration_config(settings), run)


def _overrides(session: Session) -> ChecksSettings:
    return session.context.settings.section(CHECKS_SECTION, ChecksSettings)


def _raw_runner(session: Session) -> VerificationRunner:
    registry = session.context.registry("checks")
    return VerificationRunner(build_checks(registry, _overrides(session)), RawPolicy(), {})


def _check_units(session: Session) -> tuple[CheckUnit, ...]:
    registry = session.context.registry("checks")
    return tuple(CheckUnit.of(registry[c]) for c in sorted(registry))


def _checks_hash(session: Session) -> str:
    return active_checks_hash(session.context.settings, session.context.registry("checks"))


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


def _guard(
    session: Session,
    leaf: str,
    selections: Sequence[Selection],
    artifacts: dict[str, str] | None = None,
) -> None:
    guard_sealed(
        session.context,
        f"{NAME} {leaf}",
        selections,
        artifacts=artifacts,
        run_dir=session.run.run_dir,
    )


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
    """The runs read back; a perceived-only set gets its oracle run after the sealed guard."""
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
    _guard(session, "fit", [Selection(dataset.name, None, clips)])
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
        _guard(session, "fit", [Selection(source.name, name, tuple(chosen))])
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


# --- devscore -----------------------------------------------------------------------------------


@app.command("devscore", help="Development metric on calib: H1 and out-of-fold J (WP-O).")
def devscore(
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    level: Annotated[str | None, typer.Option(help="oracle, perceived or both.")] = None,
    clips: ClipsOption = None,
    out: Annotated[Path | None, typer.Option(help="Also write devscore.json here.")] = None,
    no_sigma: Annotated[bool, typer.Option(help="Do not recalibrate the perceived sigma.")] = False,
) -> None:
    """Writes only its run directory (and ``--out``): never calibration_dir or the thresholds."""
    session = _session()
    cal = session.config.calibration
    require_bootstrap(cal)
    source = resolve_dataset(session.context, dataset or cal.dataset)
    name = subset or cal.subset
    levels = _levels(level, session)
    chosen = _fit_clips(source, name, clips or ())
    _require_calib(chosen)
    _guard(session, "devscore", [Selection(source.name, name, tuple(chosen))])
    reports = _live_reports(session, source, chosen, levels)
    data = _level_data(session, source, reports, levels, no_sigma)
    meta = _fit_meta(session, source, name, reports.clips)
    result = dev_score(data, _check_units(session), meta, session.config)
    run_dir = session.run.run_dir
    text = result.report.to_json()
    for path in (run_dir / DEVSCORE_FILE, *(() if out is None else (out,))):
        _write(path, text)
        typer.echo(f"devscore: {path}")
    _write(run_dir / cal.artifact, result.artifact.to_json())
    typer.echo(f"calibration: {run_dir / cal.artifact} (full calib fit)")
    for line in devscore_lines(result.report):
        typer.echo(line)


def _require_calib(clips: Sequence[ClipManifest]) -> None:
    """Checked before any check runs: devscore is a calib-only metric."""
    wrong = sorted(c.clip_id for c in clips if c.split is not Split.CALIB)
    if wrong:
        raise UsageError(f"devscore runs on split=calib only; not calib: {wrong[:5]}")


def devscore_lines(report: DevScoreReport) -> list[str]:
    lines = [f"note: {report.scope_note}"]
    for level in report.levels:
        lines += _level_lines(level)
    chosen = report.selection
    if chosen is not None:
        evaluable = "-" if chosen.oracle_evaluable is None else ",".join(chosen.oracle_evaluable)
        lines.append(
            f"selection ({chosen.level.value}): n_met {chosen.n_met}; oracle evaluable "
            f"[{evaluable}]; all met {chosen.all_oracle_evaluable_met}; "
            f"tie-break AUROC_cv {_num(chosen.tiebreak_auroc_cv)}"
        )
    return lines


def _level_lines(level: LevelDevScore) -> list[str]:
    auroc = level.auroc_cv
    lines = [
        f"{level.level.value}/{level.profile}: {level.n_met}/{len(level.checks)} meet H1 and "
        f"J_cv^lo > 0 ({','.join(level.met) or 'none'}); AUROC_cv {_num(auroc.value)} "
        f"[{_num(auroc.ci.low)}, {_num(auroc.ci.high)}]"
    ]
    for check_id, c in sorted(level.checks.items()):
        lines.append(
            f"  {check_id}: tau_b {_num(c.h1.kendall_tau_b)} [{_num(c.h1.ci.low)}, "
            f"{_num(c.h1.ci.high)}] H1 {c.meets_h1}; J_cv {_num(c.j_cv.informedness)} "
            f"[{_num(c.j_cv.informedness_ci.low)}, {_num(c.j_cv.informedness_ci.high)}]"
        )
    return lines


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
    _guard(session, "stage-sigma", [Selection(source.name, ctx.diag.subset, tuple(calib))])
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
    real_dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--real-dataset",
            help="Real dataset to score instead of calibration.real_datasets (repeatable).",
        ),
    ] = None,
) -> None:
    if real_dataset and not real:
        raise UsageError("--real-dataset and --no-real exclude each other")
    session = _session()
    calibration = _artifact(session, artifact)
    cal = session.config.calibration
    source = resolve_dataset(session.context, dataset or cal.dataset)
    name = subset or cal.subset
    chosen = select_clips(source.clips(name), clips or ())
    levels = sorted({lv for lv, _ in calibration.strata()})
    sources = _real_sources(session, real_dataset) if real else {}
    real_subset = diagnosis_settings(session.context.settings).subset
    selections = [
        Selection(source.name, name, tuple(chosen)),
        *(Selection(n, real_subset, found) for n, (_, found) in sorted(sources.items())),
    ]
    _guard(session, "evaluate", selections, {"calibration": calibration.sha256})
    inputs = tuple(
        EvaluationInputs(_eval_data(session, source, chosen, lv), _real_data(session, lv, sources))
        for lv in levels
    )
    meta = EvaluationMeta(
        created_utc=session.run.created_utc,
        run_id=session.run.run_id,
        artifact_path=str(calibration.path),
        dataset=_dataset_ref(source, name, "all", chosen),
        profile=session.context.settings.profile,
        real_sets=tuple(
            _dataset_ref(src, real_subset, "all", real_clips)
            for name, (src, real_clips) in sorted(sources.items())
            if any(name in item.real for item in inputs)
        ),
        primary=_primary_scope(session, source, name, chosen),
    )
    report = evaluate(calibration, inputs, meta, session.config)
    _write(session.run.run_dir / EVALUATION_FILE, report.to_json())
    _write(session.run.run_dir / EVALUATION_MD, evaluation_markdown(report))
    typer.echo(f"evaluation: {session.run.run_dir / EVALUATION_FILE}")


def _primary_scope(
    session: Session, source: DatasetSource, subset: str, clips: Sequence[ClipManifest]
) -> PrimaryScope | None:
    scope = SceneScope(primary_scenes=h1_scope(session.context.settings).scenes_for(source.name))
    if not scope.primary_scenes:
        return None
    kept = [c for c in clips if in_scope(scene_of(c), scope.indices)]
    return PrimaryScope(scope=scope, dataset=_dataset_ref(source, subset, "all", kept))


def _eval_data(
    session: Session, source: DatasetSource, clips: Sequence[ClipManifest], level: InputLevel
) -> LevelData:
    report = _raw_runner(session).run(_source(session, source, level), clips)
    return level_data(level, report, _check_units(session), session.config, split=None, pairs=None)


RealSources = dict[str, tuple[DatasetSource, tuple[ClipManifest, ...]]]


def _real_sources(session: Session, named: Sequence[str] | None = None) -> RealSources:
    """The real datasets with their clips of the ``diagnosis.subset``: the ``named`` ones, which
    must all resolve, or else those of ``calibration.real_datasets`` that resolve."""
    subset = diagnosis_settings(session.context.settings).subset
    found: RealSources = {}
    for name in dict.fromkeys(named or session.config.calibration.real_datasets):
        try:
            source = resolve_dataset(session.context, name)
            found[name] = (source, tuple(source.clips(subset)))
        except PhysverifyError as exc:
            if named:
                raise
            typer.echo(f"real dataset {name} skipped: {exc}")
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
        bool, typer.Option(help="Publish a stratum that is not usable yet (warns).")
    ] = False,
) -> None:
    session = _session()
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError("--level must be oracle or perceived")
    path = artifact or default_artifact_path(session.context.settings)
    stratum = _publishable(session, path, InputLevel(level), profile, force=force)
    settings = session.context.settings
    target = settings.paths.thresholds
    _write(target, _projection_text(stratum, settings.paths.calibration_dir))
    typer.echo(f"thresholds: {target} ({stratum.source})")


def _publishable(
    session: Session, path: Path, level: InputLevel, profile: str | None, *, force: bool
) -> Stratum:
    """The stratum through the consumers' gate; ``force`` publishes one that is not usable."""
    name = profile or session.context.settings.profile
    try:
        return open_stratum(
            path,
            level,
            name,
            checks_hash=_checks_hash(session),
            row_step=UNKNOWN_ROW_STEP,
            require_calibrated=session.config.calibration.require_calibrated_artifact,
        )
    except UnusableStratumError as exc:
        if not force:
            raise UsageError(f"{exc}: refit on more scenes or pass --force") from exc
        typer.echo(f"warning: publishing a stratum that is not usable ({exc})", err=True)
    except CalibrationError as exc:
        raise UsageError(str(exc)) from exc
    return load_calibration(path).stratum(level, name)


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
    return number(value, ".4g", "-")
