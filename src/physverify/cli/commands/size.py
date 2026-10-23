"""``physverify size``: fetch Objectron boxes, fit the size model, measure it (ADR 0010).

Network but not heavy: ``fetch`` reads only the leading ``Sequence.objects`` bytes of each
annotation through ranged GETs. Every figure written is an estimate with its measured error."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.sizing import (
    evaluate_end_to_end,
    evaluate_test_split,
    fetch_heads,
    fit_and_write,
    prepare_end_to_end,
)
from physverify.cli.context import app_context
from physverify.sizing.end_to_end import e2e_lines
from physverify.sizing.evaluate import eval_lines

NAME = "size"

app = typer.Typer(help="Object size model learned on Objectron (fetch, fit, eval).")


@app.command("fetch", help="Download the box sizes of the selected Objectron videos.")
def fetch() -> None:
    fetch_heads(app_context().settings, typer.echo)


@app.command("fit", help="Fit the per-class model on train; writes configs/size_model.toml.")
def fit() -> None:
    model, target, run_dir = fit_and_write(app_context())
    typer.echo(f"model: {target} ({len(model.classes)} classes, factor {model.sigma_factor:.4f})")
    typer.echo(f"run: {run_dir}")


@app.command("eval", help="Error on the official test split against B0 and B1.")
def evaluate_command() -> None:
    report, run_dir = evaluate_test_split(app_context())
    for line in eval_lines(report):
        typer.echo(line)
    typer.echo(f"run: {run_dir}")


@app.command("e2e-prepare", help="Download and decode a few Objectron test videos (network).")
def e2e_prepare(
    videos: Annotated[int | None, typer.Option(help="Override [sizing.e2e] videos.")] = None,
) -> None:
    draw, manifest = prepare_end_to_end(app_context().settings, videos)
    typer.echo(
        f"{len(draw.taken)} clips prepared, {len(draw.skipped)} videos skipped (> max_video_mb)"
    )
    typer.echo(f"manifest: {manifest}")


@app.command("e2e-eval", help="Perceived scale against ARKit depth on the prepared clips.")
def e2e_eval() -> None:
    report, run_dir = evaluate_end_to_end(app_context())
    for line in e2e_lines(report):
        typer.echo(line)
    typer.echo(f"run: {run_dir}")
