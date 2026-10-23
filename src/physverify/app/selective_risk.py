"""Wiring of ``abstain fit`` and ``abstain curve`` from the application context."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import chain_runner, check_suite, dataset_for, pipeline_deps
from physverify.calibration.artifact_io import default_artifact_path
from physverify.core.atomic import write_text_atomic
from physverify.core.digests import sha256_if_file
from physverify.core.errors import PhysverifyError
from physverify.datasets._base import DatasetSource
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.settings import diagnosis_settings
from physverify.runtime.runs import new_run, write_json
from physverify.schema.checks import InputLevel
from physverify.selective_risk.curve import raw_table, require_taus
from physverify.selective_risk.report import (
    curve_csv,
    curve_parts,
    curve_record,
    fit_lines,
    fit_record,
    fitted_k,
    summary_lines,
    tables_of,
)
from physverify.selective_risk.settings import (
    ABSTENTION_FILE,
    CURVE_FILE,
    REPORT_FILE,
    SECTION,
    AbstentionSettings,
)
from physverify.selective_risk.tables import (
    Session,
    collect_synthetic,
    compare_tables,
    level_tables,
    require,
    sweep,
)
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, checks_hash
from physverify.verification.inputs import refuse_heavy
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    VerificationSettings,
    load_thresholds,
)
from physverify.verification.sweep import CheckUnit

Echo = Callable[[str], None]


def fit_abstention(
    context: AppContext,
    command: str,
    oracle: Path | None,
    perceived: Path | None,
    provisional: bool,
    *,
    err: Echo,
) -> list[str]:
    """``abstain fit``: k* on the calibration split, written to ``abstention.json``."""
    _guard(context, command, real=False, err=err)
    session = _session(context, oracle, perceived, provisional)
    settings = session.settings
    collected = collect_synthetic(session, settings.calib_split)
    require(collected, settings.calib_split)
    result = sweep(collected, session.thresholds, settings)
    record = fit_record(collected, result, settings, tables_of(session))
    target = session.calibration_dir / ABSTENTION_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, record)
    return fit_lines(record, result, settings.risk_target, target)


def abstention_curve(
    context: AppContext,
    command: str,
    oracle: Path | None,
    perceived: Path | None,
    provisional: bool,
    *,
    err: Echo,
) -> list[str]:
    """``abstain curve``: the held-out curve of both tables, written into a new run directory."""
    _guard(context, command, real=True, err=err)
    session = _session(context, oracle, perceived, provisional)
    settings = session.settings
    fitted = fitted_k(session)
    evaluated = collect_synthetic(session, settings.eval_split)
    require(evaluated, settings.eval_split)
    calibrated, raw = session.thresholds, raw_table(session.thresholds)
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    compared = compare_tables(evaluated, calibrated, raw, settings)
    parts = curve_parts(session, evaluated, fitted)
    report = curve_record(evaluated, compared, fitted, settings, tables_of(session), parts)
    write_text_atomic(run_dir / CURVE_FILE, curve_csv(compared))
    write_json(run_dir / REPORT_FILE, report)
    return [*summary_lines(report), f"run: {run_dir}"]


def _session(
    context: AppContext, oracle: Path | None, perceived: Path | None, provisional: bool
) -> Session:
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    registry = context.registry("checks")
    fallback = load_thresholds(settings.paths.thresholds, verification.thresholds)
    overrides = {
        level: path
        for level, path in ((InputLevel.ORACLE, oracle), (InputLevel.PERCEIVED, perceived))
        if path is not None
    }
    tables = level_tables(
        None if provisional else default_artifact_path(settings),
        settings.profile,
        fallback,
        overrides,
        stale_hash=checks_hash(registry, settings.section(CHECKS_SECTION, ChecksSettings)),
    )
    for level, table in tables.items():
        require_taus(table, level)
    return Session(
        chain=partial(_chain, context),
        calibration_dir=settings.paths.calibration_dir,
        settings=settings.section(SECTION, AbstentionSettings),
        suite=check_suite(context),
        reference_table=tables[InputLevel.ORACLE],
        thresholds=tables[InputLevel.PERCEIVED],
        min_span_iou=diagnosis_settings(settings).end_to_end.min_span_iou,
        checks=tuple(CheckUnit.of(cls) for _, cls in sorted(registry.items())),
        couplings=frozenset((p, c) for p, c in verification.specificity.couplings),
    )


def _chain(context: AppContext, dataset: str) -> tuple[ChainRunner, DatasetSource]:
    source = dataset_for(context, dataset)
    deps = pipeline_deps(context, None, partial(refuse_heavy, dataset))
    return chain_runner(context, deps, source), source


def _guard(context: AppContext, command: str, *, real: bool, err: Echo) -> None:
    """The sealed-ledger guard over the clips ``collect_synthetic`` (the eval split with
    ``real``, else calib) and ``collect_real`` read; a dataset that does not resolve is reported
    where it is read."""
    settings = context.settings
    own = settings.section(SECTION, AbstentionSettings)
    wanted: list[tuple[str, str | None]] = [
        (own.dataset, own.eval_split if real else own.calib_split)
    ]
    wanted += [(name, None) for name in own.real_datasets] if real else []
    selections = []
    for name, subset in wanted:
        try:
            source = dataset_for(context, name)
        except PhysverifyError:
            continue
        selections.append(Selection(source.name, subset, tuple(source.clips(subset))))
    abstention = sha256_if_file(settings.paths.calibration_dir / ABSTENTION_FILE)
    shas = {} if abstention is None else {"abstention": abstention}
    guard_sealed(context, command, selections, err=err, artifacts=shas)
