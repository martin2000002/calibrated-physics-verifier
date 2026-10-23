"""``physverify abstain``: risk-coverage of the threshold policy and the choice of k* (ADR 0010 §3).

Not heavy: the cases come from cached chains (``selective_risk.tables``)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.selective_risk import abstention_curve, fit_abstention
from physverify.cli.context import app_context, echo_err

NAME = "abstain"

app = typer.Typer(help="Risk-coverage of the verdict policy and the abstention level k*.")

OracleTableOption = Annotated[
    Path | None,
    typer.Option(help="Oracle-level table for the reference (default: A8's oracle stratum)."),
]
PerceivedTableOption = Annotated[
    Path | None,
    typer.Option(help="Perceived-level table for the decisions (default: A8's perceived stratum)."),
]
ProvisionalOption = Annotated[
    bool,
    typer.Option(
        help="Ignore A8's artifact: both levels use verify's fallback table "
        "(paths.thresholds, else the provisional taus)."
    ),
]


@app.command("fit", help="Choose k* on the calibration split; writes abstention.json.")
def fit(
    oracle_table: OracleTableOption = None,
    perceived_table: PerceivedTableOption = None,
    provisional: ProvisionalOption = False,
) -> None:
    lines = fit_abstention(
        app_context(), f"{NAME} fit", oracle_table, perceived_table, provisional, err=echo_err
    )
    for line in lines:
        typer.echo(line)


@app.command("curve", help="Held-out curve, raw vs calibrated table, per check, real video.")
def curve(
    oracle_table: OracleTableOption = None,
    perceived_table: PerceivedTableOption = None,
    provisional: ProvisionalOption = False,
) -> None:
    lines = abstention_curve(
        app_context(), f"{NAME} curve", oracle_table, perceived_table, provisional, err=echo_err
    )
    for line in lines:
        typer.echo(line)
