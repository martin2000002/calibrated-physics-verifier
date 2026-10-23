"""``physverify results``: the numbers of deliverable 2 from one A8 v2 chain (and an H2 run).

Read only and not heavy: it takes no lock, runs no model and writes nothing outside ``--out``.
Every value cites the ``path#pointer`` it was read from; inputs that do not belong to the same
artifact are refused (exit 2). A chain whose device is not cpu is a rehearsal: not reportable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.cli.context import app_context
from physverify.reporting.numbers import collect_results
from physverify.reporting.render import write_results
from physverify.reporting.sources import Request, collect_inputs
from physverify.verification.scope import SECTION as SCOPE_SECTION
from physverify.verification.scope import CheckScope

NAME = "results"
DEFAULT_OUT = "results"

app = typer.Typer(
    invoke_without_command=True,
    help="Collect the reported numbers of an A8 v2 chain into results.md and results.json.",
)


@app.callback()
def results(
    a8v2_dir: Annotated[Path, typer.Option(help="The chain's output directory (state.env).")],
    h1_evaluation: Annotated[
        Path | None,
        typer.Option(help="evaluation.json with the pooled tau_b (default: the chain's)."),
    ] = None,
    h2_run: Annotated[
        Path | None, typer.Option(help="The verify run `agreement compare` wrote into.")
    ] = None,
    abstention_run: Annotated[
        Path | None,
        typer.Option(help="The `selective-risk curve` run (default: the one step 7's log names)."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Default: <a8v2-dir>/results.")] = None,
) -> None:
    scope = app_context().settings.section(SCOPE_SECTION, CheckScope)
    request = Request(
        a8v2_dir=a8v2_dir,
        h1_evaluation=h1_evaluation,
        h2_run=h2_run,
        abstention_run=abstention_run,
        check_scope=scope,
    )
    found = collect_results(collect_inputs(request))
    if found.banner:
        typer.echo(found.banner)
    for section in found.sections:
        if section.missing:
            typer.echo(f"{section.key}: missing ({section.missing})")
    json_path, md_path = write_results(out or a8v2_dir / DEFAULT_OUT, found)
    typer.echo(f"results: {md_path}")
    typer.echo(f"json: {json_path}")
