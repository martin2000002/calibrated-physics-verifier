"""``physverify extract``: run the stage plan over a dataset, fetch its weights (ADR 0005, 3.12)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import extraction
from physverify.cli.context import app_context, heavy_command
from physverify.core.errors import EXIT_ERROR

NAME = "extract"

app = typer.Typer(
    help="Perception pipeline: from video to states and contacts.",
    no_args_is_help=True,
)


@app.command("run", help="Run the extraction plan over a dataset.")
def run(
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    gt: Annotated[list[str] | None, typer.Option(help="Stages replaced by ground truth.")] = None,
    fps: Annotated[float | None, typer.Option(help="Target sampling fps.")] = None,
    max_side: Annotated[int | None, typer.Option(help="Longest frame side in pixels.")] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
) -> None:
    report = extraction.run_extraction(
        app_context(),
        dataset,
        clips or (),
        gt or (),
        fps=fps,
        max_side=max_side,
        subset=subset,
        echo=typer.echo,
    )
    if report.failures:
        raise typer.Exit(code=EXIT_ERROR)


@app.command("fetch-weights", help="Download the weights of the active profile's providers.")
@heavy_command
def fetch_weights() -> None:
    extraction.fetch_weights(app_context(), echo=typer.echo)


@app.command(
    "purge",
    help="Remove every cached stage of the clips a dataset migration re-rendered (all providers).",
)
def purge(
    clips_from: Annotated[
        Path, typer.Option(help="Migration manifest (<dataset>/migrations/<name>.json).")
    ],
    cache_root: Annotated[
        list[Path] | None,
        typer.Option(help="Cache root (repeatable; default: paths.cache_dir)."),
    ] = None,
) -> None:
    extraction.purge_rerendered(
        clips_from,
        cache_root or (),
        default_root=lambda: app_context().settings.paths.cache_dir,
        echo=typer.echo,
    )
