"""``physverify extract`` (stub owned by cycle A5)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "extract"
OWNER = "A5"
IMPLEMENTED = False

app = typer.Typer(
    help="Perception pipeline: from video to states and contacts. Not implemented yet (cycle A5).",
    no_args_is_help=True,
)


@app.command("run", help="Run the extraction plan over a dataset.")
def run(
    dataset: Annotated[str | None, typer.Option(help="Dataset name.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    gt: Annotated[list[str] | None, typer.Option(help="Stages replaced by ground truth.")] = None,
    fps: Annotated[float | None, typer.Option(help="Target sampling fps.")] = None,
    max_side: Annotated[int | None, typer.Option(help="Longest frame side in pixels.")] = None,
) -> None:
    not_implemented(OWNER, "extract run")


@app.command("fetch-weights", help="Download the weights of the active profile's providers.")
def fetch_weights() -> None:
    not_implemented(OWNER, "extract fetch-weights")
