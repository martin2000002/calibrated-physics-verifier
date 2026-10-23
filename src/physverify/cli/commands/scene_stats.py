"""``physverify scene-stats``: per-clip stage errors, per-scene tables and scene-bootstrap deltas
between two configurations. ``per-clip`` reads the cache ``diagnose stages`` filled."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.scene_stats import collect_per_clip, delta_line, delta_query, print_tables
from physverify.cli.commands.diagnose import ClipsOption, DatasetOption, SubsetOption
from physverify.cli.context import app_context, echo_err, heavy_command

NAME = "scene-stats"

app = typer.Typer(
    help="Per-clip DETECT/DEPTH errors, per-scene tables and scene-bootstrap deltas.",
    no_args_is_help=True,
)


@app.command("per-clip", help="Per-clip sufficient statistics of the DETECT and DEPTH metrics.")
@heavy_command
def per_clip(
    ctx: typer.Context,
    label: Annotated[str, typer.Option(help="Configuration label stored in the file.")],
    out: Annotated[Path, typer.Option(help="Output JSON.")],
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
) -> None:
    rows = collect_per_clip(
        app_context(ctx),
        f"{NAME} per-clip",
        label,
        out,
        dataset=dataset,
        subset=subset,
        clips=clips or (),
        err=echo_err,
    )
    typer.echo(f"collected: {out} ({rows} rows)")


@app.command("table", help="Pooled and per-scene values of every collected file.")
def table(files: Annotated[list[Path], typer.Argument(help="per-clip JSON files.")]) -> None:
    print_tables(files, echo=typer.echo)


@app.command("delta", help="candidate - base with a scene-bootstrap interval.")
def delta(
    base: Annotated[Path, typer.Argument(help="Baseline per-clip JSON.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate per-clip JSON.")],
    metric: Annotated[str, typer.Option(help="Metric key, e.g. depth/object_rel.")],
    quantity: Annotated[str, typer.Option(help="Sample or rate name, e.g. depth_rel.")],
    stat: Annotated[str, typer.Option(help="rms, mean or rate.")] = "rms",
    split: Annotated[str, typer.Option(help="calib or eval.")] = "eval",
    n: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 2000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
    ci: Annotated[float, typer.Option(help="Interval level.")] = 0.95,
) -> None:
    try:
        query = delta_query(split, metric, quantity, stat)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(delta_line(base, candidate, query, n, seed, ci))
