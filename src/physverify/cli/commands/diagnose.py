"""``physverify diagnose``: stage errors, end-to-end error, attribution and real video.

Not a heavy command: ``LazyLock`` inside ``PipelineDeps`` takes the heavy lock only when a stage
has to be computed, so a fully cached diagnosis runs next to another cycle's heavy job.
"""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.diagnosis import (
    ALL_DATASETS,
    attribution_runner,
    end_to_end_runner,
    launch,
    real_runner,
    stage_errors_runner,
)
from physverify.cli.context import app_context, echo_err

NAME = "diagnose"
STAGES, END_TO_END, ATTRIBUTION, REAL = "stages", "end_to_end", "attribution", "real"


app = typer.Typer(
    help="Extractor diagnosis: per-stage error, end-to-end error, attribution and real video.",
    no_args_is_help=True,
)

DatasetOption = Annotated[
    str | None, typer.Option(help="Dataset name (default: diagnosis.dataset).")
]
SubsetOption = Annotated[
    str | None, typer.Option(help="Dataset subset (default: diagnosis.subset).")
]
ClipsOption = Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")]


@app.command("stages", help="Level 1: per-stage error against ground truth (error profile).")
def stages(
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
    write_profile: Annotated[
        bool, typer.Option(help="Write the calib error profile to paths.error_profile.")
    ] = False,
    replace_mismatched: Annotated[
        bool,
        typer.Option(
            help="Replace entries measured under another profile or other params instead of "
            "refusing the merge (a deliberate re-measurement)."
        ),
    ] = False,
) -> None:
    context = app_context()
    runner = stage_errors_runner(
        context,
        f"{NAME} stages",
        clips or (),
        write_profile=write_profile,
        replace_mismatched=replace_mismatched,
        err=echo_err,
    )
    launch(context, STAGES, runner, echo=typer.echo, suite=False, dataset=dataset, subset=subset)


@app.command("end-to-end", help="Level 2: check magnitudes of the perceived chain vs the oracle.")
def end_to_end(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    context = app_context()
    runner = end_to_end_runner(context, f"{NAME} end-to-end", clips or (), err=echo_err)
    launch(context, END_TO_END, runner, echo=typer.echo, suite=True, dataset=dataset, subset=subset)


@app.command("attribution", help="Level 3: factorial attribution of check error to stages.")
def attribution(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    context = app_context()
    runner = attribution_runner(context, f"{NAME} attribution", clips or (), err=echo_err)
    launch(
        context, ATTRIBUTION, runner, echo=typer.echo, suite=True, dataset=dataset, subset=subset
    )


@app.command("real", help="Real video (end-to-end only): Physics-IQ motion, PISA falls, verdicts.")
def real(
    dataset: Annotated[
        str, typer.Option(help="physics_iq, pisa_real or all (diagnosis.real.datasets).")
    ] = ALL_DATASETS,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
) -> None:
    context = app_context()
    runner = real_runner(context, f"{NAME} real", dataset, clips or (), err=echo_err)
    launch(context, REAL, runner, echo=typer.echo, suite=True, subset=subset)
