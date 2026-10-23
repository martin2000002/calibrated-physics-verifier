"""CLI plumbing: command discovery and access to the ``AppContext`` carried in ``ctx.obj``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType

import typer
from typer._click.globals import get_current_context

from physverify.app.context import AppContext
from physverify.core.errors import InvalidPluginError

COMMANDS_PACKAGE = "physverify.cli.commands"


def check_command(module: ModuleType) -> None:
    """A CLI plugin module defines NAME and a typer ``app``."""
    ok = isinstance(getattr(module, "NAME", None), str) and isinstance(
        getattr(module, "app", None), typer.Typer
    )
    if not ok:
        raise InvalidPluginError(f"{module.__name__}: needs NAME and app")


def command_name(module: ModuleType) -> str:
    name: str = module.NAME
    return name


def app_context(ctx: typer.Context | None = None) -> AppContext:
    context = ctx if ctx is not None else get_current_context()
    found = context.find_object(AppContext)
    if found is None:
        raise RuntimeError("no AppContext: run commands through physverify.cli.main")
    return found


def echo_err(line: str) -> None:
    """One line on the error stream, as ``typer.echo(line, err=True)``."""
    typer.echo(line, err=True)


def heavy_command[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the command body while holding the heavy-job lock."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with app_context().heavy_lock():
            return func(*args, **kwargs)

    return wrapper
