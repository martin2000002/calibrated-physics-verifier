"""A command that asks for 240 native 4K frames under the configured memory budget."""

from __future__ import annotations

import typer

from physverify.cli.context import app_context
from physverify.core.video import check_frame_budget

NAME = "budget"

app = typer.Typer(invoke_without_command=True, help="Over-budget fixture command.")


@app.callback()
def budget(ctx: typer.Context) -> None:
    check_frame_budget(240, (3840, 2160), app_context(ctx).settings.runtime.max_frame_bytes)
