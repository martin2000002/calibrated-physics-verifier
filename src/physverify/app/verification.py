"""The ``verify`` service: run the checks over a dataset and print the summary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import checks_hash_for, dataset_for, pipeline_deps, plan_for
from physverify.checks._base import Check
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource, select_clips
from physverify.runtime.runs import new_run
from physverify.schema.checks import InputLevel
from physverify.schema.stages import Stage, parse_stages
from physverify.verification.check_set import CHECKS_SECTION, ChecksSettings, build_checks
from physverify.verification.inputs import (
    InputSource,
    OracleSource,
    PerceivedSource,
    PipelineOutputLoader,
    refuse_heavy,
)
from physverify.verification.runner import (
    CHECKS_HASH_FILE,
    VERIFICATION_SECTION,
    RowStatus,
    VerificationReport,
    VerificationRunner,
    VerificationSettings,
    build_policy,
    write_report,
)
from physverify.verification.summary import (
    LEVELS as LEVELS,
)
from physverify.verification.summary import (
    matrix_lines,
    parse_level,
    specificity_lines,
    summary_lines,
)
from physverify.verification.sweep import CheckUnit, matrix, write_sweep
from physverify.verification.target_ranking import specificity
from physverify.verification.thresholding import ThresholdTable, load_thresholds


def policy_names(context: AppContext) -> list[str]:
    return sorted(context.registry("policies"))


@dataclass(frozen=True)
class _Verifier:
    verification: VerificationSettings
    checks: tuple[Check, ...]
    thresholds: ThresholdTable
    runner: VerificationRunner


def _verifier(context: AppContext, policy: str | None) -> _Verifier:
    """The configured checks, thresholds and verdict policy (``policy`` overrides the latter)."""
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    if policy is not None:
        verification = verification.model_copy(update={"policy": policy})
    thresholds = load_thresholds(settings.paths.thresholds, verification.thresholds)
    checks = build_checks(
        context.registry("checks"), settings.section(CHECKS_SECTION, ChecksSettings)
    )
    runner = VerificationRunner(
        checks, build_policy(context.registry("policies"), verification), thresholds
    )
    return _Verifier(verification, checks, thresholds, runner)


def _write_run(
    context: AppContext, run_dir: Path, report: VerificationReport, *, echo: Echo
) -> bool:
    """Write the report and the checks hash, print the summary; whether a row errored."""
    write_report(run_dir, report)
    (run_dir / CHECKS_HASH_FILE).write_text(checks_hash_for(context) + "\n", encoding="utf-8")
    for line in summary_lines(report):
        echo(line)
    return any(row.status is RowStatus.ERROR for row in report.table)


def _sweep(run_dir: Path, report: VerificationReport, verifier: _Verifier, *, echo: Echo) -> bool:
    """Write and print the check-by-target matrix; whether the specificity gate passed."""
    identities = [CheckUnit.of(check) for check in verifier.checks]
    cells = matrix(report)
    spec = specificity(report, verifier.verification.specificity, identities)
    write_sweep(run_dir, cells, spec)
    for line in [*matrix_lines(cells), *specificity_lines(spec)]:
        echo(line)
    return spec.passed


def run_verify(
    context: AppContext,
    command: str,
    dataset: str,
    level: str | None,
    gt: Sequence[str],
    *,
    subset: str | None,
    clips: Sequence[str],
    policy: str | None,
    sweep: bool,
    echo: Echo,
    err: Echo,
) -> bool:
    """Run the checks and print the summary; whether a row errored or the sweep failed."""
    input_level = parse_level(level, gt, sweep)
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    verifier = _verifier(context, policy)
    inputs = _inputs(context, source, input_level, parse_stages(gt))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    echo(f"run: {run_dir}")
    guard_sealed(
        context,
        command,
        [Selection(source.name, subset, tuple(selected))],
        err=err,
        run_dir=run_dir,
    )
    echo(f"thresholds: {verifier.thresholds.label}")
    report = verifier.runner.run(inputs, selected)
    failed = _write_run(context, run_dir, report, echo=echo)
    if sweep:
        failed = not _sweep(run_dir, report, verifier, echo=echo) or failed
    return failed


def _inputs(
    context: AppContext, source: DatasetSource, level: InputLevel, gt_stages: Sequence[Stage]
) -> InputSource:
    if level is InputLevel.ORACLE:
        return OracleSource(source)
    plan = plan_for(context, dataset=source.name, gt_stages=tuple(gt_stages))
    return PerceivedSource(
        source,
        PipelineOutputLoader(
            plan, pipeline_deps(context, None, partial(refuse_heavy, source.name))
        ),
        level,
    )
