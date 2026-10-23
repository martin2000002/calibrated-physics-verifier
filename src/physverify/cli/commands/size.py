"""``physverify size``: fetch Objectron boxes, fit the size model, measure it (ADR 0010).

Network but not heavy: ``fetch`` reads only the leading ``Sequence.objects`` bytes of each
annotation through ranged GETs. Every figure written is an estimate with its measured error."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.sizing import evaluate_end_to_end, prepare_end_to_end
from physverify.cli.context import app_context
from physverify.config.models import SizePriorTable
from physverify.core.atomic import write_text_atomic
from physverify.core.errors import MissingInputError
from physverify.runtime.runs import new_run, write_json
from physverify.sizing import objectron
from physverify.sizing.end_to_end import e2e_lines
from physverify.sizing.evaluate import MODEL, eval_lines, evaluate
from physverify.sizing.fit import fit_model, fit_record, render_toml
from physverify.sizing.model import SECTION as MODEL_SECTION
from physverify.sizing.model import SizeModelTable
from physverify.sizing.objectron_draws import TEST, TRAIN, load_samples, objectron_dir, selection
from physverify.sizing.priors import SIZE_PRIORS_SECTION
from physverify.sizing.settings import sizing_settings

NAME = "size"
MODEL_FILE = "size_model.toml"

app = typer.Typer(help="Object size model learned on Objectron (fetch, fit, eval).")


@app.command("fetch", help="Download the box sizes of the selected Objectron videos.")
def fetch() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    reader = objectron.UrllibPrefixReader()
    plan = objectron.HeadFetch(
        sizing.bucket_url, objectron_dir(settings) / "heads", sizing.head_bytes, sizing.retries
    )
    for split in objectron.SPLITS:
        refs = selection(settings, sizing, split, reader)
        stored = objectron.fetch_heads(reader, plan, refs, sizing.workers)
        typer.echo(f"{split}: {len(refs)} videos, {stored} new bytes stored")


@app.command("fit", help="Fit the per-class model on train; writes configs/size_model.toml.")
def fit() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    train = load_samples(settings, sizing, TRAIN, objectron.UrllibPrefixReader())
    source = f"objectron train ({sizing.license}), {len(train)} objects, seed {sizing.seed}"
    model = fit_model(
        train,
        folds=sizing.folds,
        seed=sizing.seed,
        coverage=sizing.coverage_target,
        source=source,
    )
    target = settings.config_dir / MODEL_FILE
    write_text_atomic(target, render_toml(model))
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "fit.json", fit_record(model, train, sizing))
    typer.echo(f"model: {target} ({len(model.classes)} classes, factor {model.sigma_factor:.4f})")
    typer.echo(f"run: {run_dir}")


@app.command("eval", help="Error on the official test split against B0 and B1.")
def evaluate_command() -> None:
    context = app_context()
    settings = context.settings
    sizing = sizing_settings(settings)
    model = settings.section(MODEL_SECTION, SizeModelTable)
    if not model.classes:
        raise MissingInputError("configs/size_model.toml is empty; run `size fit` first")
    test = load_samples(settings, sizing, TEST, objectron.UrllibPrefixReader())
    priors = settings.section(SIZE_PRIORS_SECTION, SizePriorTable).root
    report = evaluate(
        test,
        model,
        priors[sizing.baseline_prior],
        n_boot=sizing.bootstrap,
        seed=sizing.seed,
        ci=sizing.ci,
    )
    report["test_sigma_log"] = report["total"][MODEL]["rms_log"]
    report["truth_limit"] = "box sizes come from ARKit; their own error is not documented"
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    write_json(run_dir / "eval.json", report)
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
