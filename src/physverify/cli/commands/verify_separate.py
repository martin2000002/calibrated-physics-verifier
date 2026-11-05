"""``physverify verify-separate``: run the rows reported apart from R1-R5b (R1_dyn), unscored."""

from __future__ import annotations

import functools
from typing import Annotated

import typer

from physverify.app.separate_rows import run_separate
from physverify.app.verification import LEVELS
from physverify.cli.context import app_context, echo_err
from physverify.cli.options import choices_metavar, require_one_of
from physverify.core.errors import EXIT_ERROR

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
) -> None:
    failed = run_separate(
        app_context(ctx),
        NAME,
        dataset,
        level,
        gt or (),
        subset=subset,
        clips=clips or (),
        echo=typer.echo,
        err=echo_err,
    )
    if failed:
        raise typer.Exit(code=EXIT_ERROR)
