"""``physverify agreement``: the verifier against the reference VLM on VideoPhy human labels.

``fetch`` is heavy; ``subset``, ``freeze`` and ``compare`` read files only."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import human_agreement as service
from physverify.cli.context import app_context, echo_err, heavy_command

NAME = "agreement"

app = typer.Typer(
    help="Verifier vs reference VLM on VideoPhy human labels (human-agreement study).",
    no_args_is_help=True,
)
RunOption = Annotated[Path, typer.Option("--verify-run", help="A perceived verify run directory.")]


@app.command("fetch", help="Download the label tables and the subset's videos (default: smoke).")
@heavy_command
def fetch(
    ctx: typer.Context, subset: Annotated[str | None, typer.Option(help="VideoPhy subset.")] = None
) -> None:
    service.fetch(app_context(ctx), subset, echo=typer.echo)


@app.command("subset", help="Write the pending caption review (--audit) or record it (--record).")
def subset(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Option("--subset", help="Default: the test subset.")] = None,
    show: Annotated[bool, typer.Option("--audit", help="Print the captions to review.")] = False,
    freeze_review: Annotated[
        bool, typer.Option("--record", help="Freeze n and sha256 of the reviewed subsets.")
    ] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Fail unless the review is recorded (no download).")
    ] = False,
) -> None:
    service.subset(
        app_context(ctx), name, show=show, record=freeze_review, check=check, echo=typer.echo
    )


@app.command("freeze", help="Pre-register the test comparison from a dev verify run.")
def freeze_command(
    ctx: typer.Context,
    run_dir: RunOption,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing pre-registration (archived).")
    ] = False,
    declare_deviation: Annotated[
        str | None,
        typer.Option(
            help=(
                "Why the freeze is redone after a test comparison (makes the human-agreement"
                " study post hoc)."
            )
        ),
    ] = None,
) -> None:
    service.preregister(
        app_context(ctx),
        run_dir,
        force=force,
        deviation=declare_deviation,
        echo=typer.echo,
        err=echo_err,
    )


@app.command("compare", help="Compare a verify run with the VLM and the human labels.")
def compare_command(
    ctx: typer.Context,
    run_dir: RunOption,
    name: Annotated[str | None, typer.Option("--subset", help="Default: the test subset.")] = None,
    rule: Annotated[str | None, typer.Option(help="provisional_any_fail or calibrated.")] = None,
    preregistration: Annotated[
        Path | None,
        typer.Option(
            help="Pre-registration file. The test-comparison log always goes next to the "
            "default pre-registration path, which `agreement freeze` writes."
        ),
    ] = None,
) -> None:
    request = service.CompareRequest(run_dir, name, rule, preregistration)
    service.compare(app_context(ctx), request, echo=typer.echo)
