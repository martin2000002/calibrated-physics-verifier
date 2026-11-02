"""``physverify calibrate``: fit, evaluate and publish the calibration.

Not heavy: the checks read cached stage outputs and refuse the lock. ``stage-sigma`` takes the
heavy lock lazily, only when a stage has to be computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import calibration as service
from physverify.app.devscore_compare import compare_devscore
from physverify.cli import options as opt
from physverify.cli.context import app_context, echo_err

NAME = "calibrate"

app = typer.Typer(
    help="Calibration: thresholds, rates, detection limits and the clip score.",
    no_args_is_help=True,
)


@app.command("fit", help="Fit thresholds, rates and weights on split=calib.")
def fit(
    ctx: typer.Context,
    dataset: opt.DatasetOption = None,
    subset: opt.SubsetOption = None,
    level: opt.LevelOption = None,
    clips: opt.ClipsOption = None,
    from_run: Annotated[list[str] | None, typer.Option(help="verify --policy raw run ids.")] = None,
    out: Annotated[Path | None, typer.Option(help="Artifact path.")] = None,
    no_sigma: opt.NoSigmaOption = False,
) -> None:
    service.fit_artifact(
        app_context(ctx),
        f"{NAME} fit",
        service.Picked(dataset, subset, clips),
        level,
        from_run,
        out,
        no_sigma,
        echo=typer.echo,
        err=echo_err,
    )


@app.command(
    "dev-score", help="Development metric on calib: held-out monotonicity and out-of-fold J."
)
def devscore(
    ctx: typer.Context,
    dataset: opt.DatasetOption = None,
    subset: opt.SubsetOption = None,
    level: opt.LevelOption = None,
    clips: opt.ClipsOption = None,
    out: Annotated[Path | None, typer.Option(help="Also write devscore.json here.")] = None,
    no_sigma: opt.NoSigmaOption = False,
) -> None:
    """Writes only its run directory (and ``--out``): never calibration_dir or the thresholds."""
    service.score_devscore(
        app_context(ctx),
        f"{NAME} dev-score",
        service.Picked(dataset, subset, clips),
        level,
        out,
        no_sigma,
        echo=typer.echo,
        err=echo_err,
    )


@app.command("compare-devscore", help="Paired AUROC_cv deltas b - a of two dev-score runs.")
def compare(
    ctx: typer.Context,
    a: opt.ArmOption,
    b: opt.ArmOption,
    level: opt.OneLevelOption = "perceived",
    out: Annotated[Path | None, typer.Option(help="Also write devscore_compare.json here.")] = None,
) -> None:
    compare_devscore(app_context(ctx), a, b, level, out, echo=typer.echo)


@app.command("stage-sigma", help="Per-stage sigma factors (heavy, lazy lock; H200).")
def stage_sigma(
    ctx: typer.Context,
    dataset: opt.DatasetOption = None,
    subset: opt.SubsetOption = None,
    clips: opt.ClipsOption = None,
) -> None:
    picked = service.Picked(dataset, subset, clips)
    service.fit_stage_sigma(
        app_context(ctx), f"{NAME} stage-sigma", picked, echo=typer.echo, err=echo_err
    )


@app.command("evaluate", help="Score held-out and real clips with the frozen artifact.")
def evaluate_command(
    ctx: typer.Context,
    artifact: opt.ArtifactOption = None,
    dataset: opt.DatasetOption = None,
    subset: opt.SubsetOption = None,
    clips: opt.ClipsOption = None,
    real: Annotated[bool, typer.Option(help="Also score the real datasets.")] = True,
    real_dataset: opt.RealDatasetOption = None,
    score_all_clips: Annotated[
        bool,
        typer.Option(help="Score every clip of a development set, calib ones included."),
    ] = False,
) -> None:
    service.evaluate_artifact(
        app_context(ctx),
        f"{NAME} evaluate",
        service.Picked(dataset, subset, clips),
        artifact,
        real,
        real_dataset,
        score_all_clips=score_all_clips,
        echo=typer.echo,
        err=echo_err,
    )


@app.command("publish", help="Copy a stratum's projection to paths.thresholds (verify uses it).")
def publish(
    ctx: typer.Context,
    level: opt.OneLevelOption,
    profile: opt.ProfileOption = None,
    artifact: opt.ArtifactOption = None,
    force: Annotated[
        bool, typer.Option(help="Publish a stratum that is not usable yet (warns).")
    ] = False,
) -> None:
    service.publish_stratum(
        app_context(ctx), level, profile, artifact, force, echo=typer.echo, err=echo_err
    )


@app.command("show", help="Summary of the artifact per stratum and check.")
def show(
    ctx: typer.Context,
    artifact: opt.ArtifactOption = None,
    level: Annotated[str | None, typer.Option()] = None,
    profile: opt.ProfileOption = None,
) -> None:
    for line in service.show_artifact(app_context(ctx), artifact, level, profile):
        typer.echo(line)
