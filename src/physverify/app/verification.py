"""The ``verify`` service: run the checks over a dataset and print the summary."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import checks_hash_for, dataset_for, pipeline_deps, plan_for
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
    VerificationRunner,
    VerificationSettings,
    build_policy,
    load_thresholds,
    thresholds_label,
    write_report,
)
from physverify.verification.summary import matrix_lines, specificity_lines, summary_lines
from physverify.verification.sweep import CheckUnit, matrix, write_sweep
from physverify.verification.target_ranking import specificity


def run_verify(
    context: AppContext,
    command: str,
    dataset: str,
    level: InputLevel,
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
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    if policy is not None:
        verification = verification.model_copy(update={"policy": policy})
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    thresholds = load_thresholds(settings.paths.thresholds, verification.thresholds)
    checks = build_checks(
        context.registry("checks"), settings.section(CHECKS_SECTION, ChecksSettings)
    )
    runner = VerificationRunner(
        checks, build_policy(context.registry("policies"), verification), thresholds
    )
    inputs = _inputs(context, source, level, parse_stages(gt))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    echo(f"run: {run_dir}")
    guard_sealed(
        context,
        command,
        [Selection(source.name, subset, tuple(selected))],
        err=err,
        run_dir=run_dir,
    )
    echo(f"thresholds: {thresholds_label(thresholds)}")
    report = runner.run(inputs, selected)
    write_report(run_dir, report)
    checks_hash = checks_hash_for(context)
    (run_dir / CHECKS_HASH_FILE).write_text(checks_hash + "\n", encoding="utf-8")
    for line in summary_lines(report):
        echo(line)
    failed = any(row.status is RowStatus.ERROR for row in report.table)
    if sweep:
        identities = [CheckUnit.of(check) for check in checks]
        cells = matrix(report)
        spec = specificity(report, verification.specificity, identities)
        write_sweep(run_dir, cells, spec)
        for line in [*matrix_lines(cells), *specificity_lines(spec)]:
            echo(line)
        failed = failed or not spec.passed
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
