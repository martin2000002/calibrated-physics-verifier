"""``physverify timing``: the per-clip cost harness (ADR 0008).

Only runs on the H200 (``--device cuda --profile precise``) are cost measurements; any other run is
a harness check, and then no timing is printed to the terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import timing as service
from physverify.cli.context import app_context, echo_err, heavy_command
from physverify.core.errors import EXIT_OK
from physverify.extraction.weights import NetworkDownloader, hub_cache
from physverify.generation.wan21 import fetch_generator
from physverify.timing.budget import BudgetQuery
from physverify.timing.environment import TorchCudaProbe
from physverify.timing.settings import load_timing
from physverify.timing.target_services import ClipSelection

NAME = "timing"

app = typer.Typer(
    help=(
        "Per-clip cost harness. Only H200 runs (--device cuda --profile precise) are cost "
        "measurements; anything else is a harness check."
    ),
    no_args_is_help=True,
)


@app.command("list", help="Registered targets with their warm-up and repetitions.")
def list_targets() -> None:
    for line in service.target_lines(app_context()):
        typer.echo(line)


@app.command("env", help="Environment of a run with the current options and its validity (JSON).")
def env() -> None:
    typer.echo(service.environment_report(app_context().settings, TorchCudaProbe))


@app.command("fetch", help="Download the pinned Wan2.1 weights (about 29 GB; for the H200).")
@heavy_command
def fetch_weights() -> None:
    settings = app_context().settings
    typer.echo(f"hub cache: {hub_cache(settings.paths)}")
    path = fetch_generator(load_timing(settings).generation, settings.paths, NetworkDownloader())
    typer.echo(f"verified {path}")


@app.command("run", help="Time targets; only an H200 run (cuda, precise profile) is a measurement.")
def run(
    target: Annotated[
        list[str] | None, typer.Option(help="Target name (repeatable); default: timing.order.")
    ] = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset for targets that do not bring their own clips.")
    ] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Keep the first N clips.")] = None,
    repetitions: Annotated[
        int | None, typer.Option(help="Timed calls per clip, for every selected target.")
    ] = None,
    warmup: Annotated[
        int | None, typer.Option(help="Warm-up calls, for every selected target.")
    ] = None,
) -> None:
    request = service.TimingRequest(
        tuple(target or ()),
        ClipSelection(dataset, subset, tuple(clips or ()), limit),
        warmup,
        repetitions,
    )
    code = service.run_timing(app_context(), request, TorchCudaProbe, echo=typer.echo, err=echo_err)
    if code != EXIT_OK:
        raise typer.Exit(code=code)


RunOption = Annotated[Path, typer.Option("--run", help="Run directory (or its timing/ folder).")]


@app.command("summarize", help="Rebuild stats.csv and summary.md from samples.csv.")
def summarize(run_dir: RunOption) -> None:
    service.summarize(app_context().settings, run_dir, typer.echo)


@app.command(
    "budget", help="Training prompts and pairs that fit in a GPU budget (valid runs only)."
)
def budget(
    run_dir: RunOption,
    gpu_hours: Annotated[float, typer.Option(help="GPU hours available (H).")],
    candidates: Annotated[int, typer.Option(help="Candidate videos per prompt (K).")],
    eval_prompts: Annotated[int, typer.Option(help="Evaluation prompts (Q).")] = 0,
    eval_samples: Annotated[int, typer.Option(help="Samples per prompt, trained model (K').")] = 1,
    finetune_hours: Annotated[float, typer.Option(help="GPU hours of fine-tuning (T_ft).")] = 0.0,
    extract_batch: Annotated[
        int | None, typer.Option(help="Clips per extraction process (B_x).")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Generation batch size (B).")] = 1,
) -> None:
    query = BudgetQuery(
        gpu_hours, candidates, eval_prompts, eval_samples, finetune_hours, extract_batch, batch_size
    )
    for line in service.budget_report(app_context().settings, run_dir, query):
        typer.echo(line)
