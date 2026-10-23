"""``physverify verify``: run the checks R1-R5b over a dataset.

Not a heavy command: the perceived path runs the light stages only and refuses the heavy lock.
"""

from __future__ import annotations

import functools
from typing import Annotated

import typer

from physverify.app.verification import LEVELS, policy_names, run_verify
from physverify.cli.context import app_context, echo_err
from physverify.cli.options import choices_metavar, require_one_of
from physverify.core.errors import EXIT_ERROR

NAME = "verify"

app = typer.Typer(
    invoke_without_command=True,
    help="Run the physics checks R1-R5b over a dataset (oracle or perceived inputs).",
)

LevelOption = Annotated[
    str,
    typer.Option(
        parser=functools.partial(require_one_of, values=LEVELS),
        metavar=choices_metavar(LEVELS),
        help="Input level.",
    ),
]


def _known_policy(ctx: typer.Context, value: str | None) -> str | None:
    return None if value is None else require_one_of(value, policy_names(app_context(ctx)))


@app.callback()
def verify(
    ctx: typer.Context,
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    level: LevelOption,
    gt: Annotated[
        list[str] | None, typer.Option(help="Stages replaced by ground truth (perceived).")
    ] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    policy: Annotated[
        str | None,
        typer.Option(callback=_known_policy, help="Verdict policy (default: verification.policy)."),
    ] = None,
    sweep: Annotated[bool, typer.Option(help="Specificity sweep (oracle only).")] = False,
) -> None:
    failed = run_verify(
        app_context(ctx),
        NAME,
        dataset,
        level,
        gt or (),
        subset=subset,
        clips=clips or (),
        policy=policy,
        sweep=sweep,
        echo=typer.echo,
        err=echo_err,
    )
    if failed:
        raise typer.Exit(code=EXIT_ERROR)
