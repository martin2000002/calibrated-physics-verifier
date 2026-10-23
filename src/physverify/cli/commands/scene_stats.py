"""``physverify scene-stats``: per-clip stage errors, per-scene tables and scene-bootstrap deltas
between two configurations (ADR 0014). ``per-clip`` reads the cache ``diagnose stages`` filled."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.scene_stats import collect_per_clip
from physverify.cli.commands.diagnose import ClipsOption, DatasetOption, SubsetOption
from physverify.cli.context import app_context, echo_err, heavy_command
from physverify.diagnosis import scene_stats

NAME = "scene-stats"

app = typer.Typer(
    help="Per-clip DETECT/DEPTH errors, per-scene tables and scene-bootstrap deltas.",
    no_args_is_help=True,
)


@app.command("per-clip", help="Per-clip sufficient statistics of the DETECT and DEPTH metrics.")
@heavy_command
def per_clip(
    label: Annotated[str, typer.Option(help="Configuration label stored in the file.")],
    out: Annotated[Path, typer.Option(help="Output JSON.")],
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
) -> None:
    rows = collect_per_clip(
        app_context(),
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
    for line in scene_stats.TABLE_HEADER:
        typer.echo(line)
    for path in files:
        for line in scene_stats.table_lines(scene_stats.load(path)):
            typer.echo(line)


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
        parsed = scene_stats.parse_stat(stat)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    query = scene_stats.DeltaQuery(split=split, metric=metric, quantity=quantity, stat=parsed)
    result = scene_stats.scene_delta(
        scene_stats.load(base).rows, scene_stats.load(candidate).rows, query, n, seed, ci
    )
    typer.echo(
        f"{metric} {quantity} {stat} ({split}, {result.scenes} scenes): delta {result.point:+.4f}, "
        f"{ci:.0%} CI [{result.low}, {result.high}]"
    )
