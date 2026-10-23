"""``physverify exclusions``: the preregistered exclusion decision (ADR 0015 rules 1.1, 1.2, 1.5).

``r2-clips`` lists the clips both R2 detector arms must verify (``verify --policy raw``);
``decide`` reads the artifact, its evaluation and the two runs and writes the decision record
that ``[check_scope]`` merges; ``show`` prints that record and its sha256. Not heavy: nothing
here runs a model.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from physverify.app.calibration import decide_exclusions, r2_clip_listing, show_exclusions
from physverify.cli.context import app_context

NAME = "exclusions"

app = typer.Typer(
    help="ADR 0015: the preregistered check exclusions H2 and H3 consume.",
    no_args_is_help=True,
)

_out = partial(typer.echo, nl=False)


@app.command("r2-clips", help="The clip ids both R2 detector arms verify (rule 1.2).")
def r2_clips(
    out: Annotated[Path | None, typer.Option(help="Write the ids here, one per line.")] = None,
) -> None:
    _out(r2_clip_listing(app_context(), out))


@app.command("decide", help="Apply rules 1.1, 1.2 and 1.5 and write the decision record.")
def decide_command(
    evaluation: Annotated[Path, typer.Option(help="evaluation.json of `calibrate evaluate`.")],
    r2_base_run: Annotated[str, typer.Option(help="Run id of the base-detector R2 arm.")],
    r2_tiny_run: Annotated[str, typer.Option(help="Run id of the tiny-detector R2 arm.")],
    artifact: Annotated[
        Path | None, typer.Option(help="Artifact (default: the calibration artifact).")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the decision, write nothing.")] = False,
    replace: Annotated[bool, typer.Option(help="Supersede a different record.")] = False,
) -> None:
    runs = (r2_base_run, r2_tiny_run)
    context = app_context()
    decide_exclusions(
        context, evaluation, runs, artifact, dry_run=dry_run, replace=replace, out=_out
    )


@app.command("show", help="Print the decision record and its sha256 (what H2/H3 freeze on).")
def show_command(
    checks_hash: Annotated[
        bool, typer.Option(help="Refuse a record decided with other check parameters.")
    ] = True,
) -> None:
    show_exclusions(app_context(), checks_hash=checks_hash, out=_out)
