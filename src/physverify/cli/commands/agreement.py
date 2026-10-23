"""``physverify agreement``: the verifier against the reference VLM on VideoPhy human labels
(ADR 0011). ``fetch`` is heavy; ``subset``, ``freeze`` and ``compare`` read files only."""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path
from typing import Annotated

import typer

from physverify.cli.context import app_context, echo_err, heavy_command
from physverify.datasets._base import fetch_all
from physverify.datasets.videophy import VideophyDataset
from physverify.human_agreement.clip_decision import import_calibration
from physverify.human_agreement.settings import agreement_settings
from physverify.human_agreement.subset_review import check_recorded, record_subsets, review_subset
from physverify.human_agreement.workflow import (
    CompareRequest,
    apply_freeze,
    compare_and_report,
    freeze,
)

NAME = "agreement"

app = typer.Typer(
    help="Verifier vs reference VLM on VideoPhy human labels (H2).", no_args_is_help=True
)
RunOption = Annotated[Path, typer.Option("--verify-run", help="A perceived verify run directory.")]


def _dataset() -> VideophyDataset:
    return VideophyDataset(app_context().settings)


@app.command("fetch", help="Download the label tables and the subset's videos (default: smoke).")
@heavy_command
def fetch(subset: Annotated[str | None, typer.Option(help="VideoPhy subset.")] = None) -> None:
    fetch_all([VideophyDataset], app_context().settings, subset, echo=typer.echo)


@app.command("subset", help="Write the pending caption review (--audit) or record it (--record).")
def subset(
    name: Annotated[str | None, typer.Option("--subset", help="Default: the test subset.")] = None,
    show: Annotated[bool, typer.Option("--audit", help="Print the captions to review.")] = False,
    freeze_review: Annotated[
        bool, typer.Option("--record", help="Freeze n and sha256 of the reviewed subsets.")
    ] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Fail unless the review is recorded (no download).")
    ] = False,
) -> None:
    dataset = _dataset()
    settings = app_context().settings
    chosen = name or agreement_settings(settings).test_subset
    if check:
        check_recorded(dataset, echo=typer.echo)
        return
    if freeze_review:
        record_subsets(dataset, chosen, echo=typer.echo)
        return
    review_subset(dataset, settings, chosen, show=show, echo=typer.echo)


@app.command("freeze", help="Pre-register the test comparison from a dev verify run.")
def freeze_command(
    run_dir: RunOption,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing pre-registration (archived).")
    ] = False,
    declare_deviation: Annotated[
        str | None,
        typer.Option(help="Why the freeze is redone after a test comparison (makes H2 post hoc)."),
    ] = None,
) -> None:
    context = app_context()
    path, current = freeze(
        context.settings,
        context.registry("checks"),
        _dataset(),
        run_dir,
        import_calibration,
        dt.datetime.now(dt.UTC),
    )
    apply_freeze(
        path, current, force=force, deviation=declare_deviation, echo=typer.echo, err=echo_err
    )


@app.command("compare", help="Compare a verify run with the VLM and the human labels.")
def compare_command(
    run_dir: RunOption,
    name: Annotated[str | None, typer.Option("--subset", help="Default: the test subset.")] = None,
    rule: Annotated[str | None, typer.Option(help="provisional_any_fail or calibrated.")] = None,
    preregistration: Annotated[
        Path | None,
        typer.Option(
            help="Pre-registration file. The test-comparison log always goes next to the "
            "default pre-registration path, which `agreement freeze` writes."
        ),
    ] = None,
) -> None:
    request = CompareRequest(run_dir, name, rule, preregistration)
    context = app_context()
    compare_and_report(
        context.settings,
        context.registry("checks"),
        _dataset(),
        request,
        import_calibration,
        functools.partial(dt.datetime.now, dt.UTC),
        echo=typer.echo,
    )
