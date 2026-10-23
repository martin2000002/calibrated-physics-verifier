"""Wiring of the ``calibrate`` leaves from the application context."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.diagnosis import diagnosis_context
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import (
    chain_runner,
    checks_hash_for,
    dataset_for,
    pipeline_deps,
    plan_for,
)
from physverify.calibration.artifact_io import default_artifact_path, load_calibration
from physverify.calibration.collect import SubjectLookup, Subjects
from physverify.calibration.dev_score import DEVSCORE_FILE, dev_score, require_bootstrap
from physverify.calibration.evaluate import EvaluationMeta, evaluate
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.report import EVALUATION_FILE, EVALUATION_MD, evaluation_markdown
from physverify.calibration.session import (
    Session,
    artifact_at,
    dataset_ref,
    evaluation_inputs,
    fit_clips,
    fit_level_data,
    fit_meta,
    levels_of,
    live_reports,
    primary_scope,
    projection_text,
    publish_fit,
    publishable,
    real_sources,
    require_calib,
    run_reports,
    with_oracle,
    write_stage_sigma,
)
from physverify.calibration.settings import calibration_config
from physverify.calibration.sigma_recalibration import (
    factor_map,
    recalibrate_profile,
    stage_factors,
)
from physverify.calibration.summary import devscore_lines, show_lines
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource, select_clips
from physverify.diagnosis.chain_runner import warm_variants
from physverify.diagnosis.chain_variants import oracle_variant, perceived_variant
from physverify.diagnosis.end_to_end import track_of
from physverify.diagnosis.pairing import subject_map
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import measure, pair_variants, stage_pairs
from physverify.extraction.error_profile import load_error_profile
from physverify.extraction.events import ConsoleObserver, EventRecorder
from physverify.runtime.runs import new_run
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.synth.scene_scope import scene_of
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, build_checks
from physverify.verification.inputs import (
    InputSource,
    OracleSource,
    PerceivedSource,
    PipelineOutputLoader,
    refuse_heavy,
)
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner
from physverify.verification.sweep import units_of
from physverify.verification.thresholding import ThresholdTable

Guard = Callable[[Sequence[Selection], Mapping[str, str] | None], None]


@dataclass(frozen=True)
class Picked:
    """The ``--dataset``/``--subset``/``--clips`` a leaf was given (None: the configured one)."""

    dataset: str | None
    subset: str | None
    clips: Sequence[str] | None


def fit_artifact(
    context: AppContext,
    command: str,
    picked: Picked,
    level: str | None,
    from_run: Sequence[str] | None,
    out: Path | None,
    no_sigma: bool,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """``calibrate fit``: the artifact, its projections and the run's copies."""
    session = _session(context, echo, err)
    guard = _guard(context, command, session, err)
    cal = session.config.calibration
    source = dataset_for(context, picked.dataset or cal.dataset)
    name = picked.subset or cal.subset
    levels: tuple[InputLevel, ...]
    if from_run:
        read = run_reports(session, source, from_run)
        guard([Selection(source.name, None, read.clips)], None)
        reports = with_oracle(session, source, read)
        levels = tuple(
            lv for lv in (InputLevel.ORACLE, InputLevel.PERCEIVED) if lv in reports.by_level
        )
    else:
        levels = levels_of(level, session)
        chosen = fit_clips(source, name, picked.clips or ())
        guard([Selection(source.name, name, tuple(chosen))], None)
        reports = live_reports(session, source, chosen, levels)
    data = fit_level_data(session, source, reports, levels, no_sigma)
    meta = fit_meta(session, source, name, reports.clips)
    artifact = fit_calibration(FitInputs(data, session.check_units(), meta), session.config)
    path = out or default_artifact_path(context.settings)
    publish_fit(session, path, artifact.to_json())


def score_devscore(
    context: AppContext,
    command: str,
    picked: Picked,
    level: str | None,
    out: Path | None,
    no_sigma: bool,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """``calibrate dev-score``: writes only its run directory (and ``out``)."""
    session = _session(context, echo, err)
    guard = _guard(context, command, session, err)
    cal = session.config.calibration
    require_bootstrap(cal)
    source = dataset_for(context, picked.dataset or cal.dataset)
    name = picked.subset or cal.subset
    levels = levels_of(level, session)
    chosen = fit_clips(source, name, picked.clips or ())
    require_calib(chosen)
    guard([Selection(source.name, name, tuple(chosen))], None)
    reports = live_reports(session, source, chosen, levels)
    data = fit_level_data(session, source, reports, levels, no_sigma)
    meta = fit_meta(session, source, name, reports.clips)
    result = dev_score(data, session.check_units(), meta, session.config)
    run_dir = session.run.run_dir
    text = result.report.to_json()
    for path in (run_dir / DEVSCORE_FILE, *(() if out is None else (out,))):
        write_text_atomic(path, text)
        echo(f"devscore: {path}")
    write_text_atomic(run_dir / cal.artifact, result.artifact.to_json())
    echo(f"calibration: {run_dir / cal.artifact} (full calib fit)")
    for line in devscore_lines(result.report):
        echo(line)


def fit_stage_sigma(
    context: AppContext, command: str, picked: Picked, *, echo: Echo, err: Echo
) -> None:
    """``calibrate stage-sigma``: the per-stage sigma factors on the calib clips."""
    session = _session(context, echo, err)
    guard = _guard(context, command, session, err)
    cal = session.config.calibration
    recorder = EventRecorder(ConsoleObserver(echo))
    ctx = diagnosis_context(
        context,
        recorder,
        None,
        dataset=picked.dataset or cal.dataset,
        subset=picked.subset or cal.subset,
    )
    source = ctx.dataset()
    calib = [
        c
        for c in select_clips(source.clips(ctx.diag.subset), picked.clips or ())
        if c.split is Split.CALIB
    ]
    if not calib:
        raise UsageError(
            f"no split=calib clip in subset {ctx.diag.subset!r}: stage-sigma fits on calib only"
        )
    guard([Selection(source.name, ctx.diag.subset, tuple(calib))], None)
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
    write_stage_sigma(session, recalibrate_profile(profile, factor_map(factors)), factors)


def evaluate_artifact(
    context: AppContext,
    command: str,
    picked: Picked,
    artifact: Path | None,
    real: bool,
    real_dataset: Sequence[str] | None,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """``calibrate evaluate``: held-out and real clips scored with the frozen artifact."""
    if real_dataset and not real:
        raise UsageError("--real-dataset and --no-real exclude each other")
    session = _session(context, echo, err)
    guard = _guard(context, command, session, err)
    calibration = artifact_at(session, artifact or default_artifact_path(context.settings))
    cal = session.config.calibration
    source = dataset_for(context, picked.dataset or cal.dataset)
    name = picked.subset or cal.subset
    chosen = select_clips(source.clips(name), picked.clips or ())
    levels = sorted({lv for lv, _ in calibration.strata()})
    sources = real_sources(session, real_dataset) if real else {}
    real_subset = diagnosis_settings(context.settings).subset
    selections = [
        Selection(source.name, name, tuple(chosen)),
        *(Selection(n, real_subset, found) for n, (_, found) in sorted(sources.items())),
    ]
    guard(selections, {"calibration": calibration.sha256})
    inputs = evaluation_inputs(session, source, chosen, levels, sources)
    meta = EvaluationMeta(
        created_utc=session.run.created_utc,
        run_id=session.run.run_id,
        artifact_path=str(calibration.path),
        dataset=dataset_ref(source, name, "all", chosen),
        profile=context.settings.profile,
        real_sets=tuple(
            dataset_ref(src, real_subset, "all", real_clips)
            for name, (src, real_clips) in sorted(sources.items())
            if any(name in item.real for item in inputs)
        ),
        primary=primary_scope(session, source, name, chosen),
    )
    report = evaluate(calibration, inputs, meta, session.config)
    write_text_atomic(session.run.run_dir / EVALUATION_FILE, report.to_json())
    write_text_atomic(session.run.run_dir / EVALUATION_MD, evaluation_markdown(report))
    echo(f"evaluation: {session.run.run_dir / EVALUATION_FILE}")


def publish_stratum(
    context: AppContext,
    level: str,
    profile: str | None,
    artifact: Path | None,
    force: bool,
    *,
    echo: Echo,
    err: Echo,
) -> None:
    """``calibrate publish``: a stratum's projection copied to ``paths.thresholds``."""
    session = _session(context, echo, err)
    if level not in (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value):
        raise UsageError("--level must be oracle or perceived")
    path = artifact or default_artifact_path(context.settings)
    stratum = publishable(session, path, InputLevel(level), profile, force=force)
    settings = context.settings
    target = settings.paths.thresholds
    write_text_atomic(target, projection_text(stratum, settings.paths.calibration_dir))
    echo(f"thresholds: {target} ({stratum.source})")


def show_artifact(
    context: AppContext, artifact: Path | None, level: str | None, profile: str | None
) -> list[str]:
    """``calibrate show``: the artifact per stratum and check; no run directory."""
    path = artifact or default_artifact_path(context.settings)
    if not path.is_file():
        raise UsageError(f"no calibration artifact at {path}: run `physverify calibrate fit`")
    return show_lines(load_calibration(path), level, profile)


def _session(context: AppContext, echo: Echo, err: Echo) -> Session:
    settings = context.settings
    run = new_run(settings, context.invocation.argv)
    echo(f"run: {run.run_dir}")
    return Session(
        settings=settings,
        config=calibration_config(settings),
        run=run,
        raw_runner=partial(_raw_runner, context),
        check_units=lambda: units_of(context.registry("checks")),
        input_source=partial(_input_source, context),
        subject_lookup=partial(_subject_lookup, context),
        dataset=partial(dataset_for, context),
        checks_hash=partial(checks_hash_for, context),
        echo=echo,
        err=err,
    )


def _guard(context: AppContext, command: str, session: Session, err: Echo) -> Guard:
    """The sealed-ledger guard of leaf ``command``, recorded in the session's run."""

    def guard(selections: Sequence[Selection], artifacts: Mapping[str, str] | None) -> None:
        guard_sealed(
            context,
            command,
            selections,
            err=err,
            artifacts=artifacts,
            run_dir=session.run.run_dir,
        )

    return guard


def _raw_runner(context: AppContext) -> VerificationRunner:
    registry = context.registry("checks")
    overrides = context.settings.section(CHECKS_SECTION, ChecksSettings)
    return VerificationRunner(
        build_checks(registry, overrides), RawPolicy(), ThresholdTable.provisional({})
    )


def _input_source(context: AppContext, dataset: DatasetSource, level: InputLevel) -> InputSource:
    if level is InputLevel.ORACLE:
        return OracleSource(dataset)
    plan = plan_for(context, dataset=dataset.name)
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset.name))
    return PerceivedSource(dataset, PipelineOutputLoader(plan, deps), level)


def _subject_lookup(context: AppContext, dataset: DatasetSource) -> SubjectLookup:
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset.name))
    chain = chain_runner(context, deps, dataset)

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
