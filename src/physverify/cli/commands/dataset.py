"""``physverify dataset`` (stub owned by cycle A4)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "dataset"
OWNER = "A4"
IMPLEMENTED = False

app = typer.Typer(
    help=(
        "Kubric synthetic dataset: image, scenes, simulation, perturbations, rendering, GT. "
        "Not implemented yet (cycle A4)."
    ),
    no_args_is_help=True,
)


@app.command("pull-image", help="Pull the pinned Kubric image (heavy).")
def pull_image(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset pull-image")


@app.command("generate", help="Sample scene specs for every family.")
def generate(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset generate")


@app.command("simulate", help="Simulate scenes in the container (heavy).")
def simulate(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset simulate")


@app.command("perturb", help="Apply scene and trajectory perturbations.")
def perturb(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset perturb")


@app.command("render", help="Render clips in the container (heavy).")
def render(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset render")


@app.command("export-gt", help="Export ground truth in the canonical schema.")
def export_gt(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset export-gt")


@app.command("list", help="List the clips of a dataset version.")
def list_command(
    subset: Annotated[str, typer.Option(help="local or full.")] = "local",
) -> None:
    not_implemented(OWNER, "dataset list")
