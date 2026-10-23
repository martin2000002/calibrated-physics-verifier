"""``physverify compare-generators``: fine-tuned against base against best-of-8 (ADR 0013).

Order: ``gate-autoeval``, ``judge --split dev``, ``freeze``, ``generate``, ``score``,
``judge --split eval``, ``compare``. Every step resumes. The GPU subcommands exit 2 without
``--device cuda`` and a CUDA device, before they write anything; ``score``, ``rescore``,
``judge --split eval`` and ``compare`` refuse when the pre-registered state changed.
"""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app import generator_comparison as comparison
from physverify.cli.context import app_context, echo_err, heavy_command
from physverify.cli.options import ShardOption, StopRuleDeviationOption
from physverify.core.errors import EXIT_ERROR

NAME = "compare-generators"

app = typer.Typer(no_args_is_help=True, help="H3: fine-tuned vs base vs best-of-8 (ADR 0013).")

ArmOption = Annotated[str, typer.Option(help="base, or ft (the selected checkpoint).")]
TaskOption = Annotated[str | None, typer.Option(help="pc or sa (default: both).")]


@app.command("generate", help="Generate the eval candidates of one arm (resumes).")
@heavy_command
def generate(
    arm: ArmOption = "base",
    shard: ShardOption = None,
    limit: int | None = None,
    stop_rule_deviation: StopRuleDeviationOption = None,
) -> None:
    context = app_context()
    comparison.generate_arm(
        context, arm, shard, limit, stop_rule_deviation, echo=typer.echo, err=echo_err
    )


@app.command("freeze", help="Write the pre-registration (before any eval score or judge row).")
def freeze(
    force: Annotated[
        bool, typer.Option(help="Replace a pre-registration whose state changed.")
    ] = False,
    declare_deviation: Annotated[
        str | None,
        typer.Option(
            help="Why a non-recalibration field changed, or any field once eval data exist "
            "(with --force; the latter makes the verdict post hoc)."
        ),
    ] = None,
    stop_rule_deviation: StopRuleDeviationOption = None,
) -> None:
    context = app_context()
    comparison.freeze_prereg(
        context, force, declare_deviation, stop_rule_deviation, echo=typer.echo, err=echo_err
    )


@app.command("score", help="Extract, verify and score the eval candidates of one arm (resumes).")
@heavy_command
def score(arm: ArmOption = "base", shard: ShardOption = None) -> None:
    comparison.score_arm(app_context(), arm, shard, echo=typer.echo)


@app.command("rescore", help="Re-aggregate stored eval check results under the frozen rule (CPU).")
def rescore_command() -> None:
    comparison.rescore_arms(app_context(), echo=typer.echo)


@app.command("judge", help="Run VideoPhy-2 AutoEval on the dev or eval candidates (resumes).")
@heavy_command
def judge(
    split: Annotated[str, typer.Option(help="dev (usability) or eval (after the freeze).")],
    task: TaskOption = None,
) -> None:
    comparison.judge_split(app_context(), split, task, echo=typer.echo)


@app.command("gate-autoeval", help="Regression gate of AutoEval on published VideoPhy-2 videos.")
@heavy_command
def gate_autoeval() -> None:
    if not comparison.run_gate(app_context(), echo=typer.echo):
        raise typer.Exit(code=EXIT_ERROR)


@app.command("compare", help="The H3 analysis and verdict from the frozen eval data (CPU).")
def compare() -> None:
    comparison.compare_arms(app_context(), echo=typer.echo)


@app.command("status", help="Eval candidates, scores, judge rows, gate and freeze of H3.")
def status() -> None:
    for line in comparison.campaign_status_lines(app_context()):
        typer.echo(line)
