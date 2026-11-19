"""``physverify final-table``: the final run's active-set preflight and its closing table.

Read only: ``preflight`` reads a calibration artifact, ``build`` reads the outputs the final run
wrote; neither selects a dataset split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.final_table import (
    ComparisonInputs,
    Inputs,
    check_inputs,
    write_comparison_table,
    write_final_table,
    write_preflight,
)
from physverify.cli.context import app_context

NAME = "final-table"

app = typer.Typer(help="The final run's preflight and closing table.")

Optional = Annotated[Path | None, typer.Option()]


@app.command("preflight", help="Assert that no weight-0 check can make a clip abstain.")
def preflight(
    artifact: Annotated[Path, typer.Option(help="The frozen calibration.json.")],
    out: Annotated[Path, typer.Option(help="Where to write preflight.json.")],
) -> None:
    if not write_preflight(artifact, out, typer.echo):
        raise typer.Exit(code=1)


@app.command("inputs", help="Assert that every active check of an evaluation got its input.")
def inputs(
    evaluation: Annotated[Path, typer.Option(help="evaluation.json to check.")],
    preflight: Annotated[Path, typer.Option(help="preflight.json (the active set).")],
) -> None:
    if not check_inputs(evaluation, preflight, typer.echo):
        raise typer.Exit(code=1)


@app.command("build", help="Write final_table.{md,json}: one row per closing criterion.")
def build(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="Output directory.")],
    freeze: Optional = None,
    kubric: Annotated[Path | None, typer.Option(help="evaluation.json of the new Kubric.")] = None,
    curve: Annotated[Path | None, typer.Option(help="abstention_report.json.")] = None,
    agreement_items: Annotated[
        Path | None, typer.Option(help="items.csv of the h2_test compare.")
    ] = None,
    preflight: Annotated[Path | None, typer.Option(help="preflight.json.")] = None,
    real: Annotated[Path | None, typer.Option(help="evaluation.json of the real sets.")] = None,
    attribution: Annotated[Path | None, typer.Option(help="diagnosis/attribution.json.")] = None,
    r3b: Annotated[Path | None, typer.Option(help="evaluation.json of the R3b extension.")] = None,
    replication: Annotated[
        list[str] | None, typer.Option(help="NAME=evaluation.json of a development replication.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="A rehearsal: the skipped freeze is not a missing output.")
    ] = False,
) -> None:
    pairs = [item.split("=", 1) for item in replication or ()]
    if any(len(p) != 2 for p in pairs):
        raise typer.BadParameter("--replication takes NAME=PATH")
    inputs = Inputs(
        freeze=freeze,
        kubric=kubric,
        curve=curve,
        agreement_items=agreement_items,
        preflight=preflight,
        real=real,
        attribution=attribution,
        r3b=r3b,
        replications={name: Path(path) for name, path in pairs},
        dry_run=dry_run,
    )
    write_final_table(app_context(ctx), inputs, out, typer.echo)


@app.command("comparison", help="Write comparison_table.{md,json}: B1 and the VLM.")
def comparison(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="Output directory.")],
    kubric: Annotated[Path | None, typer.Option(help="kubric.json of baselines compare.")] = None,
    real: Annotated[Path | None, typer.Option(help="real.json of baselines compare.")] = None,
    agreement_items: Annotated[
        Path | None, typer.Option(help="items.csv of the h2_test compare.")
    ] = None,
    final_table: Annotated[
        Path | None, typer.Option(help="final_table.json (its R2.fpr rows).")
    ] = None,
) -> None:
    inputs = ComparisonInputs(kubric, real, agreement_items, final_table)
    write_comparison_table(app_context(ctx), inputs, out, typer.echo)
