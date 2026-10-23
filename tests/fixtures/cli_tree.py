"""The CLI's command modules and its leaves ("calibrate fit", "verify") as a user types them."""

from __future__ import annotations

from functools import cache
from types import ModuleType

import typer.main

from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name
from physverify.cli.main import build_app
from physverify.core.registry import Registry, discover_modules


def command_modules() -> Registry[ModuleType]:
    return discover_modules(COMMANDS_PACKAGE, check_command, command_name)


@cache
def cli_leaves() -> dict[str, ModuleType]:
    commands = command_modules()
    root = typer.main.get_command(build_app(commands))
    found: dict[str, ModuleType] = {}
    for name, module in commands.items():
        group = root.commands[name]  # type: ignore[attr-defined]
        children = sorted(getattr(group, "commands", {}))
        for leaf in [f"{name} {child}" for child in children] or [name]:
            found[leaf] = module
    return found
