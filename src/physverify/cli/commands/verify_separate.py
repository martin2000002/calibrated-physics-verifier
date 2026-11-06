"""``physverify verify-separate``: run the rows reported apart from R1-R5b (R1_dyn), unscored."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated

import typer

from physverify.app.separate_rows import (
    DEFAULT_LAW,
    ROW_LAWS,
    fit_separate_row_sigma,
    run_separate,
)
from physverify.app.verification import LEVELS
from physverify.cli.context import app_context, echo_err
from physverify.cli.options import choices_metavar, require_one_of
from physverify.core.errors import EXIT_ERROR, UsageError

NAME = "verify-separate"

app = typer.Typer(
    invoke_without_command=True,
    help="Run the separate rows (R1_dyn) over a dataset, unscored (oracle or perceived inputs).",
)

LevelOption = Annotated[
    str,
    typer.Option(
        parser=functools.partial(require_one_of, values=LEVELS),
        metavar=choices_metavar(LEVELS),
        help="Input level.",
    ),
]


@app.callback()
def verify_separate(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    level: LevelOption,
    gt: Annotated[
        list[str] | None, typer.Option(help="Stages replaced by ground truth (perceived).")
    ] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    fit_row_sigma: Annotated[
        bool,
        typer.Option(
            "--fit-row-sigma",
            help="Fit R1_dyn's out-of-fold contact-row sigma on clean calib clips instead.",
        ),
    ] = False,
    row_sigma: Annotated[
        Path | None,
        typer.Option(help="Row-sigma artifact of --fit-row-sigma: R1_dyn runs with its sigma."),
    ] = None,
    row_law: Annotated[
        str,
        typer.Option(
            parser=functools.partial(require_one_of, values=ROW_LAWS),
            metavar=choices_metavar(ROW_LAWS),
            help="Band law of --fit-row-sigma.",
        ),
    ] = DEFAULT_LAW,
) -> None:
    if fit_row_sigma and row_sigma is not None:
        raise UsageError("--fit-row-sigma fits a new row sigma; it cannot run with --row-sigma")
    context, stages, chosen = app_context(ctx), gt or (), clips or ()
    if fit_row_sigma:
        fit_separate_row_sigma(
            context,
            NAME,
            dataset,
            level,
            stages,
            subset=subset,
            clips=chosen,
            echo=typer.echo,
            err=echo_err,
            law=row_law,
        )
        return
    failed = run_separate(
        context,
        NAME,
        dataset,
        level,
        stages,
        subset=subset,
        clips=chosen,
        echo=typer.echo,
        err=echo_err,
        row_sigma=row_sigma,
    )
    if failed:
        raise typer.Exit(code=EXIT_ERROR)
