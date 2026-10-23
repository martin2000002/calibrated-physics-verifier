"""``physverify results``: the numbers of deliverable 2 from one calibration chain.

An optional human-agreement run adds its sections. Read only and not heavy: it takes no lock, runs
no model and writes nothing outside ``--out``. Every value cites the ``path#pointer`` it was read
from; inputs that do not belong to the same artifact are refused (exit 2). A chain whose device is
not cpu is a rehearsal: not reportable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.results import report_results
from physverify.cli.context import app_context

NAME = "results"

app = typer.Typer(
    invoke_without_command=True,
    help="Collect the reported numbers of a calibration chain into results.md and results.json.",
)


@app.callback()
def results(
    ctx: typer.Context,
    chain_dir: Annotated[Path, typer.Option(help="The chain's output directory (state.env).")],
    calibration_evaluation: Annotated[
        Path | None,
        typer.Option(help="evaluation.json with the pooled tau_b (default: the chain's)."),
    ] = None,
    agreement_run: Annotated[
        Path | None, typer.Option(help="The verify run `agreement compare` wrote into.")
    ] = None,
    selective_risk_run: Annotated[
        Path | None,
        typer.Option(help="The `selective-risk curve` run (default: the one step 7's log names)."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Default: <chain-dir>/results.")] = None,
) -> None:
    report_results(
        app_context(ctx),
        chain_dir,
        calibration_evaluation=calibration_evaluation,
        agreement_run=agreement_run,
        abstention_run=selective_risk_run,
        out=out,
        echo=typer.echo,
    )
