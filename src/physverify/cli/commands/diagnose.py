"""``physverify diagnose``: stage errors, end-to-end error, attribution and real video (ADR 0007).

Not a heavy command: ``LazyLock`` inside ``PipelineDeps`` takes the heavy lock only when a stage
has to be computed, so a fully cached diagnosis runs next to another cycle's heavy job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import typer

from physverify.app.diagnosis import launch
from physverify.app.sealing import Selection, guard_sealed
from physverify.cli.context import app_context, echo_err
from physverify.datasets._base import select_clips
from physverify.diagnosis.attribution_run import run_attribution
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import run_end_to_end
from physverify.diagnosis.real_video import run_real
from physverify.diagnosis.real_video import select_clips as real_clips
from physverify.diagnosis.report import Report
from physverify.diagnosis.stage_errors import run_stage_errors
from physverify.schema.clip import ClipManifest

NAME = "diagnose"
ALL_DATASETS = "all"
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


# --- Context ----------------------------------------------------------------------------------


def dataset_clips(ctx: DiagnosisContext, wanted: Sequence[str]) -> list[ClipManifest]:
    return select_clips(ctx.dataset().clips(ctx.diag.subset), wanted)


def guarded_clips(ctx: DiagnosisContext, leaf: str, wanted: Sequence[str]) -> list[ClipManifest]:
    """``dataset_clips`` after the sealed-ledger guard of ``diagnose <leaf>``."""
    selected = dataset_clips(ctx, wanted)
    selection = Selection(ctx.dataset().name, ctx.diag.subset, tuple(selected))
    guard_sealed(
        app_context(), f"{NAME} {leaf}", [selection], err=echo_err, run_dir=ctx.run.run_dir
    )
    return selected


def real_datasets(ctx: DiagnosisContext, dataset: str | None) -> tuple[str, ...]:
    return ctx.diag.real.datasets if dataset in (None, ALL_DATASETS) else (str(dataset),)


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
    def runner(ctx: DiagnosisContext) -> Report:
        selected = guarded_clips(ctx, "stages", clips or ())
        return run_stage_errors(
            ctx, selected, write_profile=write_profile, replace_mismatched=replace_mismatched
        )

    launch(app_context(), STAGES, runner, typer.echo, suite=False, dataset=dataset, subset=subset)


@app.command("end-to-end", help="Level 2: check magnitudes of the perceived chain vs the oracle.")
def end_to_end(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_end_to_end(ctx, guarded_clips(ctx, "end-to-end", clips or ()))

    launch(
        app_context(), END_TO_END, runner, typer.echo, suite=True, dataset=dataset, subset=subset
    )


@app.command("attribution", help="Level 3: factorial attribution of check error to stages.")
def attribution(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_attribution(ctx, guarded_clips(ctx, "attribution", clips or ()))

    launch(
        app_context(), ATTRIBUTION, runner, typer.echo, suite=True, dataset=dataset, subset=subset
    )


@app.command("real", help="Real video (end-to-end only): Physics-IQ motion, PISA falls, verdicts.")
def real(
    dataset: Annotated[
        str, typer.Option(help="physics_iq, pisa_real or all (diagnosis.real.datasets).")
    ] = ALL_DATASETS,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        names, subset = real_datasets(ctx, dataset), ctx.diag.subset
        chosen = real_clips(ctx, names, subset, clips or ())
        selections = [Selection(source.name, subset, found) for source, found in chosen]
        guard_sealed(
            app_context(), f"{NAME} real", selections, err=echo_err, run_dir=ctx.run.run_dir
        )
        return run_real(ctx, names, subset, clips or ())

    launch(app_context(), REAL, runner, typer.echo, suite=True, subset=subset)
