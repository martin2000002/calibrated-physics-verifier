"""``physverify calibrate``: fit, evaluate and publish the calibration.

Not heavy: the checks read cached stage outputs and refuse the lock. ``stage-sigma`` takes the
heavy lock lazily, only when a stage has to be computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.calibration import (
    Picked,
    evaluate_artifact,
    fit_artifact,
    fit_stage_sigma,
    publish_stratum,
    score_devscore,
    show_artifact,
)
from physverify.cli.context import app_context, echo_err

NAME = "calibrate"

app = typer.Typer(
    help="Calibration: thresholds, rates, detection limits and the clip score.",
    no_args_is_help=True,
)

DatasetOption = Annotated[str | None, typer.Option(help="Dataset (default: calibration.dataset).")]
SubsetOption = Annotated[str | None, typer.Option(help="Subset (default: calibration.subset).")]
ClipsOption = Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")]
ArtifactOption = Annotated[
    Path | None, typer.Option(help="Artifact (default: calibration_dir/artifact).")
]
ProfileOption = Annotated[str | None, typer.Option(help="Sampling profile (default: active).")]
LevelOption = Annotated[str | None, typer.Option(help="oracle, perceived or both.")]
NoSigmaOption = Annotated[bool, typer.Option(help="Do not recalibrate the perceived sigma.")]


@app.command("fit", help="Fit thresholds, rates and weights on split=calib.")
def fit(
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    level: LevelOption = None,
    clips: ClipsOption = None,
    from_run: Annotated[list[str] | None, typer.Option(help="verify --policy raw run ids.")] = None,
    out: Annotated[Path | None, typer.Option(help="Artifact path.")] = None,
    no_sigma: NoSigmaOption = False,
) -> None:
    picked = Picked(dataset, subset, clips)
    fit_artifact(
        app_context(),
        f"{NAME} fit",
        picked,
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
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    level: LevelOption = None,
    clips: ClipsOption = None,
    out: Annotated[Path | None, typer.Option(help="Also write devscore.json here.")] = None,
    no_sigma: NoSigmaOption = False,
) -> None:
    """Writes only its run directory (and ``--out``): never calibration_dir or the thresholds."""
    picked = Picked(dataset, subset, clips)
    score_devscore(
        app_context(),
        f"{NAME} dev-score",
        picked,
        level,
        out,
        no_sigma,
        echo=typer.echo,
        err=echo_err,
    )


@app.command("stage-sigma", help="Per-stage sigma factors (heavy, lazy lock; H200).")
def stage_sigma(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    picked = Picked(dataset, subset, clips)
    fit_stage_sigma(app_context(), f"{NAME} stage-sigma", picked, echo=typer.echo, err=echo_err)


@app.command("evaluate", help="Score held-out and real clips with the frozen artifact.")
def evaluate_command(
    artifact: ArtifactOption = None,
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
    real: Annotated[bool, typer.Option(help="Also score the real datasets.")] = True,
    real_dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--real-dataset",
            help="Real dataset to score instead of calibration.real_datasets (repeatable).",
        ),
    ] = None,
) -> None:
    picked = Picked(dataset, subset, clips)
    evaluate_artifact(
        app_context(),
        f"{NAME} evaluate",
        picked,
        artifact,
        real,
        real_dataset,
        echo=typer.echo,
        err=echo_err,
    )


@app.command("publish", help="Copy a stratum's projection to paths.thresholds (verify uses it).")
def publish(
    level: Annotated[str, typer.Option(help="oracle or perceived.")],
    profile: ProfileOption = None,
    artifact: ArtifactOption = None,
    force: Annotated[
        bool, typer.Option(help="Publish a stratum that is not usable yet (warns).")
    ] = False,
) -> None:
    publish_stratum(app_context(), level, profile, artifact, force, echo=typer.echo, err=echo_err)


@app.command("show", help="Summary of the artifact per stratum and check.")
def show(
    artifact: ArtifactOption = None,
    level: Annotated[str | None, typer.Option()] = None,
    profile: ProfileOption = None,
) -> None:
    for line in show_artifact(app_context(), artifact, level, profile):
        typer.echo(line)
