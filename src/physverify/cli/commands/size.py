"""``physverify size``: fetch Objectron boxes, fit the size model, measure it.

Network but not heavy: ``fetch`` reads only the leading ``Sequence.objects`` bytes of each
annotation through ranged GETs. Every figure written is an estimate with its measured error."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.size_report import end_to_end_lines, evaluation_lines
from physverify.app.sizing import MODEL_FILE, fetch_heads, fit_and_write, prepare_end_to_end
from physverify.cli.context import app_context

NAME = "size"

app = typer.Typer(help="Object size model learned on Objectron (fetch, fit, eval).")


@app.command("fetch", help="Download the box sizes of the selected Objectron videos.")
def fetch(ctx: typer.Context) -> None:
    fetch_heads(app_context(ctx).settings, echo=typer.echo)


@app.command("fit", help=f"Fit the per-class model on train; writes configs/{MODEL_FILE}.")
def fit(ctx: typer.Context) -> None:
    model, target, run_dir = fit_and_write(app_context(ctx))
    typer.echo(f"model: {target} ({len(model.classes)} classes, factor {model.sigma_factor:.4f})")
    typer.echo(f"run: {run_dir}")


@app.command("eval", help="Error on the official test split against B0 and B1.")
def evaluate_command(ctx: typer.Context) -> None:
    for line in evaluation_lines(app_context(ctx)):
        typer.echo(line)


@app.command("e2e-prepare", help="Download and decode a few Objectron test videos (network).")
def e2e_prepare(
    ctx: typer.Context,
    videos: Annotated[int | None, typer.Option(help="Override [sizing.e2e] videos.")] = None,
) -> None:
    draw, manifest = prepare_end_to_end(app_context(ctx).settings, videos)
    typer.echo(
        f"{len(draw.taken)} clips prepared, {len(draw.skipped)} videos skipped (> max_video_mb)"
    )
    typer.echo(f"manifest: {manifest}")


@app.command("e2e-eval", help="Perceived scale against ARKit depth on the prepared clips.")
def e2e_eval(ctx: typer.Context) -> None:
    for line in end_to_end_lines(app_context(ctx)):
        typer.echo(line)
