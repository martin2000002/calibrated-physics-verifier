"""``physverify synth``: the Kubric synthetic dataset, step by step or end to end.

Container steps take the heavy lock; ``build`` holds it for the whole run."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app import synth as service
from physverify.cli.context import app_context, heavy_command

NAME = "synth"

SubsetOption = Annotated[
    str,
    typer.Option(
        "--subset",
        help="smoke (2 clips), local (Deliverable 1 pipeline proof) or full (H200 dataset).",
    ),
]

app = typer.Typer(
    help="Kubric synthetic dataset: image, scenes, simulation, perturbations, rendering, GT.",
    no_args_is_help=True,
)


@app.command("pull-image", help="Pull the pinned Kubric image (heavy).")
@heavy_command
def pull_image(ctx: typer.Context) -> None:
    service.pull_image(app_context(ctx))


@app.command("generate", help="Plan the subset and write the scene specs.")
def generate(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    service.generate(app_context(ctx), subset, echo=typer.echo)


@app.command("simulate", help="Simulate the pending scenes in the container (heavy).")
@heavy_command
def simulate(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    service.simulate(app_context(ctx), subset, echo=typer.echo)


@app.command("perturb", help="Apply perturbations and compute contact truth (heavy).")
@heavy_command
def perturb(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    service.perturb(app_context(ctx), subset, echo=typer.echo)


@app.command("render", help="Render the pending clips in the container (heavy, resumable).")
@heavy_command
def render(
    ctx: typer.Context,
    subset: SubsetOption = "local",
    limit: Annotated[int | None, typer.Option(min=1, help="Render at most N clips.")] = None,
    clips: Annotated[str | None, typer.Option(help="Comma-separated clip ids.")] = None,
) -> None:
    service.render(app_context(ctx), subset, limit, clips, echo=typer.echo)


@app.command("export-gt", help="Export frames, camera, states, contacts and final records.")
def export_gt(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    service.export_gt(app_context(ctx), subset, echo=typer.echo)


@app.command("list", help="Progress of every step for the subset, skipped clips and s/frame.")
def list_command(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    for line in service.progress_lines(app_context(ctx), subset, echo=typer.echo):
        typer.echo(line)


CommitOption = Annotated[
    str | None, typer.Option(help="Commit to log (default: HEAD of the repository).")
]
NameOption = Annotated[
    str | None,
    typer.Option(
        "--name",
        help="velocity-<convention> (the default) or the ground-truth fix the plan "
        f"needs next, in order: {', '.join(service.GT_MIGRATIONS)}.",
    ),
]


@app.command(
    "migrate",
    help="Re-run APPLY on every simulated clip with this checkout (a named migration: "
    "velocity-<convention> or the next ground-truth fix), keeping the version and every render "
    "whose inputs did not change; writes migrations/<name>.json.",
)
@heavy_command
def migrate_command(
    ctx: typer.Context, name: NameOption = None, commit: CommitOption = None
) -> None:
    service.migrate(app_context(ctx), name, commit, echo=typer.echo)


@app.command("migrate-velocity", help="Alias of `migrate --name velocity-<convention>`.")
@heavy_command
def velocity_alias(ctx: typer.Context, commit: CommitOption = None) -> None:
    service.migrate(app_context(ctx), None, commit, echo=typer.echo)


@app.command("build", help="generate, simulate, perturb, render and export-gt under one lock.")
@heavy_command
def build(ctx: typer.Context, subset: SubsetOption = "local") -> None:
    service.build(app_context(ctx), subset, echo=typer.echo)
