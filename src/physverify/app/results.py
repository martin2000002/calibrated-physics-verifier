"""The ``results`` service: collect a calibration chain's numbers and write the report."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from physverify.app.context import AppContext
from physverify.core.output import Echo
from physverify.reporting.numbers import collect_results
from physverify.reporting.render import write_results
from physverify.reporting.sources import Request, collect_inputs
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import CheckScope

DEFAULT_OUT: Final = "results"


def report_results(
    context: AppContext,
    chain_dir: Path,
    *,
    calibration_evaluation: Path | None,
    agreement_run: Path | None,
    abstention_run: Path | None,
    out: Path | None,
    echo: Echo,
) -> None:
    """Print the banner and the missing sections, write results.md and results.json under
    ``out`` (default: ``<chain_dir>/results``) and print their paths."""
    request = Request(
        chain_dir=chain_dir,
        calibration_evaluation=calibration_evaluation,
        agreement_run=agreement_run,
        abstention_run=abstention_run,
        check_scope=context.settings.section(SCOPE_SECTION, CheckScope),
    )
    found = collect_results(collect_inputs(request))
    if found.banner:
        echo(found.banner)
    for section in found.sections:
        if section.missing:
            echo(f"{section.key}: missing ({section.missing})")
    json_path, md_path = write_results(out or chain_dir / DEFAULT_OUT, found)
    echo(f"results: {md_path}")
    echo(f"json: {json_path}")
