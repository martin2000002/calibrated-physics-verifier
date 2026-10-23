"""``physverify verify``: run the checks R1-R5b over a dataset (ADR 0006 §3.10).

Not a heavy command: the perceived path runs the light stages only and refuses the heavy lock.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Annotated

import typer

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import checks_hash_for, dataset_for, pipeline_deps, plan_for
from physverify.cli.context import app_context, echo_err
from physverify.core.errors import EXIT_ERROR
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
from physverify.verification.summary import (
    matrix_lines,
    parse_level,
    specificity_lines,
    summary_lines,
)
from physverify.verification.sweep import (
    CheckUnit,
    matrix,
    write_sweep,
)
from physverify.verification.target_ranking import specificity

NAME = "verify"

app = typer.Typer(
    invoke_without_command=True,
    help="Run the physics checks R1-R5b over a dataset (oracle or perceived inputs).",
)


@app.callback()
def verify(
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    level: Annotated[str | None, typer.Option(help="oracle or perceived.")] = None,
    gt: Annotated[
        list[str] | None, typer.Option(help="Stages replaced by ground truth (perceived).")
    ] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    policy: Annotated[str | None, typer.Option(help="Verdict policy: threshold or raw.")] = None,
    sweep: Annotated[bool, typer.Option(help="Specificity sweep (oracle only).")] = False,
) -> None:
    input_level = parse_level(level, gt or (), sweep)
    context = app_context()
    settings = context.settings
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    if policy is not None:
        verification = verification.model_copy(update={"policy": policy})
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips or ())
    thresholds = load_thresholds(settings.paths.thresholds, verification.thresholds)
    checks = build_checks(
        context.registry("checks"), settings.section(CHECKS_SECTION, ChecksSettings)
    )
    runner = VerificationRunner(
        checks, build_policy(context.registry("policies"), verification), thresholds
    )
    inputs = _inputs(context, source, input_level, parse_stages(gt or ()))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    typer.echo(f"run: {run_dir}")
    guard_sealed(
        context,
        NAME,
        [Selection(source.name, subset, tuple(selected))],
        err=echo_err,
        run_dir=run_dir,
    )
    typer.echo(f"thresholds: {thresholds_label(thresholds)}")
    report = runner.run(inputs, selected)
    write_report(run_dir, report)
    checks_hash = checks_hash_for(context)
    (run_dir / CHECKS_HASH_FILE).write_text(checks_hash + "\n", encoding="utf-8")
    for line in summary_lines(report):
        typer.echo(line)
    failed = any(row.status is RowStatus.ERROR for row in report.table)
    if sweep:
        identities = [CheckUnit.of(check) for check in checks]
        cells = matrix(report)
        spec = specificity(report, verification.specificity, identities)
        write_sweep(run_dir, cells, spec)
        for line in [*matrix_lines(cells), *specificity_lines(spec)]:
            typer.echo(line)
        failed = failed or not spec.passed
    if failed:
        raise typer.Exit(code=EXIT_ERROR)


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
