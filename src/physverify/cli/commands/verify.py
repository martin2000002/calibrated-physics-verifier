"""``physverify verify``: run the checks R1-R5b over a dataset (ADR 0006 §3.10).

Not a heavy command: the perceived path runs the light stages only and refuses the heavy lock.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Annotated

import typer

from physverify.app.context import AppContext
from physverify.app.services import checks_hash_for, dataset_for, pipeline_deps, plan_for
from physverify.cli.commands._sealed import Selection, guard_sealed
from physverify.cli.context import app_context
from physverify.core.errors import EXIT_ERROR, UsageError
from physverify.core.markdown import number
from physverify.datasets._base import DatasetSource, select_clips
from physverify.runtime.runs import new_run
from physverify.schema.checks import InputLevel, Verdict
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
    load_thresholds,
    thresholds_label,
    write_report,
)
from physverify.verification.sweep import (
    CheckUnit,
    MatrixCell,
    Rule,
    RuleOutcome,
    SpecificityReport,
    matrix,
    specificity,
    write_sweep,
)

NAME = "verify"
LEVELS = (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value)

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
    guard_sealed(context, NAME, [Selection(source.name, subset, tuple(selected))], run_dir=run_dir)
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


def parse_level(level: str | None, gt: Sequence[str], sweep: bool) -> InputLevel:
    """ORACLE, PERCEIVED, or MIXED when a perceived run replaces stages with ground truth."""
    if level not in LEVELS:
        raise UsageError(f"--level is required: one of {list(LEVELS)}")
    parsed = InputLevel(level)
    if parsed is InputLevel.ORACLE and gt:
        raise UsageError("--gt applies to --level perceived; the oracle is ground truth already")
    if sweep and parsed is not InputLevel.ORACLE:
        raise UsageError("--sweep needs --level oracle (A4 labels are exact only on the oracle)")
    return InputLevel.MIXED if gt else parsed


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


def summary_lines(report: VerificationReport) -> list[str]:
    lines: list[str] = []
    check_ids = list(dict.fromkeys(row.check_id for row in report.table))
    for check_id in check_ids:
        rows = [row for row in report.table if row.check_id == check_id]
        counts = {status: sum(r.status is status for r in rows) for status in RowStatus}
        fails = sum(r.verdict is Verdict.FAIL for r in rows)
        tally = " ".join(f"{status.value} {n}" for status, n in counts.items())
        lines.append(f"{check_id}: {tally} FAIL {fails}")
    return lines


def matrix_lines(cells: Sequence[MatrixCell]) -> list[str]:
    lines = ["sweep: perturbation check median max n/clips"]
    for cell in cells:
        lines.append(
            f"{cell.perturbation} {cell.check_id} {_number(cell.median)} {_number(cell.max)} "
            f"{cell.n}/{cell.n_clips}"
        )
    return lines


def specificity_lines(spec: SpecificityReport) -> list[str]:
    verdict = "PASS" if spec.passed else "FAIL"
    lines = [f"specificity: {verdict} ({len(spec.outcomes)} outcomes, {len(spec.failures)} failed)"]
    for outcome in spec.outcomes:
        if outcome.passed and outcome.rule is not Rule.COUPLING:
            continue
        label = _label(outcome)
        lines.append(
            f"{label} {outcome.rule.value} {outcome.perturbation} {outcome.check_id}: "
            f"expected {_number(outcome.expected)} observed {_number(outcome.observed)} "
            f"{outcome.detail}".rstrip()
        )
    return lines


def _label(outcome: RuleOutcome) -> str:
    if outcome.rule is Rule.COUPLING:
        return "expected coupling"
    return "UNDEFINED" if outcome.undefined else "FAILED"


def _number(value: float | None) -> str:
    return number(value, ".4g", "-", "nan")
