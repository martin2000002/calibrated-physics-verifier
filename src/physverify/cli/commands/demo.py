"""``physverify demo``: one video through the whole verifier, with its report and overlay.

Ad-hoc prompts and given videos live in their own campaign (``demo.adhoc_campaign``), never in
the H3 campaign. ``scenes`` reuses the H3 campaign's k = 0 videos of the fixed demo prompts and
scores them only once the H3 preregistration exists, with the rule it froze.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app import demo as service
from physverify.cli.context import app_context, heavy_command
from physverify.demo.adhoc import ADHOC_FAMILY

NAME = "demo"

app = typer.Typer(no_args_is_help=True, help="End-to-end demo: verifier report and overlay video.")

OutOption = Annotated[
    Path | None, typer.Option(help="Output directory (default: demo.out_dir under data_dir).")
]
AdapterOption = Annotated[Path | None, typer.Option(help="LoRA adapter directory.")]


@app.command("run", help="Verify one ad-hoc prompt: generate it (CUDA) or score --video.")
@heavy_command
def run(
    prompt: Annotated[str, typer.Option(help="Prompt text.")],
    query: Annotated[
        list[str] | None, typer.Option(help="Object to detect, 'phrase' or 'phrase:N'.")
    ] = None,
    family: Annotated[str, typer.Option(help="Scene family label.")] = ADHOC_FAMILY,
    target_check: Annotated[
        list[str] | None, typer.Option(help="Checks the prompt targets (default: all).")
    ] = None,
    adapter: AdapterOption = None,
    video: Annotated[Path | None, typer.Option(help="Score this mp4 instead.")] = None,
    out: OutOption = None,
) -> None:
    request = service.AdhocRequest(prompt, query, family, target_check, adapter, video, out)
    service.run_adhoc(app_context(), request, echo=typer.echo)


@app.command(
    "scenes", help="Base vs fine-tuned reports of the fixed demo scenes (after h3 freeze)."
)
@heavy_command
def scenes(
    from_select: Annotated[
        bool, typer.Option(help="Fine-tuned tag from 'finetune select' (demo.select_run).")
    ] = False,
    adapter: AdapterOption = None,
    out: OutOption = None,
) -> None:
    service.scene_reports(
        app_context(), from_select=from_select, adapter=adapter, out=out, echo=typer.echo
    )


@app.command("report", help="Rebuild a report and overlay from its stored results (CPU).")
def report(
    results: Annotated[Path, typer.Option(help="A report directory or its report.json.")],
    out: OutOption = None,
) -> None:
    service.rebuild(app_context(), results, out, echo=typer.echo)
