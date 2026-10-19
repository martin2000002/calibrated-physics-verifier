"""``physverify agreement``: the verifier against the reference VLM on VideoPhy human labels
(ADR 0011). ``fetch`` is heavy; ``subset``, ``freeze`` and ``compare`` read files only."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from physverify.agreement.decision import import_calibration
from physverify.agreement.prereg import (
    file_sha256,
    load_preregistration,
    logged_comparisons,
    plan_refreeze,
    write_preregistration,
)
from physverify.agreement.report import REPORT_DIR, write_comparison
from physverify.agreement.settings import agreement_settings
from physverify.agreement.workflow import (
    CompareRequest,
    audit,
    draft_queries,
    freeze,
    log_test_comparison,
    reviewed_candidates,
    run_comparison,
    sized_labels_or_none,
    uncovered,
)
from physverify.cli.commands.data import fetch_all
from physverify.cli.context import app_context, heavy_command
from physverify.core.errors import ConfigError
from physverify.core.log import get_logger
from physverify.datasets._videophy_review import (
    REVIEWED,
    load_review,
    record,
    require_recorded,
    write_review,
)
from physverify.datasets.videophy import VideophyDataset
from physverify.diagnosis.atomic_io import write_text_atomic

NAME = "agreement"
OWNER = "A10"
IMPLEMENTED = True

log = get_logger("cli.agreement")

app = typer.Typer(
    help="Verifier vs reference VLM on VideoPhy human labels (H2).", no_args_is_help=True
)
RunOption = Annotated[Path, typer.Option("--verify-run", help="A perceived verify run directory.")]


def _dataset() -> VideophyDataset:
    return VideophyDataset(app_context().settings)


@app.command("fetch", help="Download the label tables and the subset's videos (default: smoke).")
@heavy_command
def fetch(subset: Annotated[str | None, typer.Option(help="VideoPhy subset.")] = None) -> None:
    fetch_all([VideophyDataset], app_context().settings, subset, typer.echo)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is invalid: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return loaded


def _write_json(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


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
    chosen = name or agreement_settings(app_context().settings).test_subset
    if check:
        require_recorded(load_review(dataset.subset_path), dataset.settings.filter_version)
        typer.echo(f"review: {dataset.subset_path} ({REVIEWED})")
        return
    if freeze_review:
        _record(dataset, chosen)
        return
    review = audit(dataset, chosen)
    write_review(dataset.subset_path, review)
    captions = [c for table in review.decisions.values() for c in table]
    _queries(dataset, captions)
    typer.echo(f"review: {dataset.subset_path} ({review.status})")
    for part, table in review.decisions.items():
        typer.echo(f"{part}: {len(table)} captions")
        if show:
            for caption, decision in table.items():
                typer.echo(f"  [{decision or '?'}] {caption}")


def _queries(dataset: VideophyDataset, captions: list[str]) -> None:
    path = dataset.queries_path
    drafts = draft_queries(_read_json(path), captions, dataset.settings.filter.rigid_event)
    _write_json(path, drafts)
    labels = sized_labels_or_none(app_context().settings)
    if labels is None:
        typer.echo("queries: sizing model not available, size coverage not checked")
        return
    missing = uncovered(drafts, labels)
    typer.echo(f"queries: {path} ({len(drafts)} captions, {len(missing)} phrases without size)")
    for phrase in missing:
        typer.echo(f"  no size: {phrase}")


def _record(dataset: VideophyDataset, chosen: str) -> None:
    review = load_review(dataset.subset_path)
    if review is None:
        raise ConfigError(f"{dataset.subset_path} is missing; run `agreement subset --audit`")
    recorded = record(review, reviewed_candidates(dataset, chosen))
    write_review(dataset.subset_path, recorded)
    for part, stored in recorded.subsets.items():
        typer.echo(f"{part}: n={stored.n} sha256={stored.sha256}")


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
    settings = app_context().settings
    path, current = freeze(
        settings, _dataset(), run_dir, import_calibration, dt.datetime.now(dt.UTC)
    )
    previous = file_sha256(path)
    old = None if previous is None else load_preregistration(path)
    record_ = plan_refreeze(
        old,
        previous,
        current,
        force=force,
        deviation=declare_deviation,
        tested=bool(logged_comparisons(path)),
    )
    if record_ is None:
        typer.echo(f"preregistration: {path}")
        typer.echo("unchanged frozen state; the existing pre-registration is kept")
        return
    if previous is not None:
        log.warning("replacing pre-registration %s (sha256 %s)", path, previous)
        typer.echo(f"replacing pre-registration {path} (sha256 {previous})", err=True)
    if record_.post_hoc:
        typer.echo("WARNING: a test comparison already exists; H2 is now post hoc", err=True)
    write_preregistration(path, record_)
    typer.echo(f"preregistration: {path}")
    typer.echo(
        f"rule {record_.rule_label}; n_dev {record_.n_dev}; majority {record_.dev_majority}; "
        f"cut {record_.dev_cut}; tuned cut {record_.dev_tuned_cut}"
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
    comparison = run_comparison(app_context().settings, _dataset(), request, import_calibration)
    out_dir = run_dir / REPORT_DIR
    write_comparison(out_dir, comparison)
    log_test_comparison(app_context().settings, comparison, run_dir, dt.datetime.now(dt.UTC))
    verdict = comparison.metrics["h2_verdict"]
    typer.echo(f"agreement: {out_dir}")
    typer.echo(f"rule: {comparison.metrics['rule']}; compared {comparison.counts.compared}")
    if not verdict["applicable"]:
        typer.echo(f"H2: {verdict['reason']}")
    else:
        post_hoc = " (POST HOC)" if verdict["post_hoc"] else ""
        typer.echo(f"H2: {'supported' if verdict['supported'] else 'not supported'}{post_hoc}")
