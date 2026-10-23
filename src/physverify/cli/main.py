"""Root CLI: global options, plugin subcommands discovered from ``cli/commands/``.

Exit codes: 0 ok, 1 error, 2 usage/budget/config, 75 heavy lock busy.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Annotated

import typer
import typer.main
from typer._click.exceptions import ClickException

from physverify.app.context import AppContext, GlobalOptions, Invocation
from physverify.cli.context import COMMANDS_PACKAGE, check_command, command_name
from physverify.core.errors import EXIT_ERROR, PhysverifyError
from physverify.core.log import LOG_LEVELS, configure_logging
from physverify.core.registry import Registry, discover_modules

PROG = "physverify"
CONFIG_DIR_ENV = "PHYSVERIFY_CONFIG_DIR"


def default_config_dir(environ: Mapping[str, str]) -> Path:
    if environ.get(CONFIG_DIR_ENV):
        return Path(environ[CONFIG_DIR_ENV])
    return Path(__file__).resolve().parents[3] / "configs"


def _root_callback(
    ctx: typer.Context,
    profile: Annotated[
        str | None, typer.Option(help="Configuration profile (fast, precise).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="cpu, cuda or cuda:N.")] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option(help=f"Config directory (default: ${CONFIG_DIR_ENV} or configs/)."),
    ] = None,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override section.key=value (repeatable).")
    ] = None,
    no_lock: Annotated[
        bool, typer.Option("--no-lock", help="Do not take the heavy-job lock.")
    ] = False,
    lock_timeout: Annotated[
        float | None, typer.Option(help="Seconds to wait for the heavy lock (0 = fail now).")
    ] = None,
    log_level: Annotated[str, typer.Option(help=f"One of {', '.join(LOG_LEVELS)}.")] = "WARNING",
) -> None:
    invocation = ctx.obj
    if not isinstance(invocation, Invocation):
        raise RuntimeError("the root context must carry an Invocation")
    configure_logging(log_level)
    options = GlobalOptions(
        config_dir=config_dir or default_config_dir(invocation.environ),
        profile=profile,
        device=device,
        overrides=tuple(overrides or ()),
        no_lock=no_lock,
        lock_timeout_s=lock_timeout,
        log_level=log_level,
    )
    ctx.obj = AppContext(invocation, options)


def build_app(commands: Registry[ModuleType]) -> typer.Typer:
    app = typer.Typer(
        name=PROG,
        help="Calibrated physics verifier for generated video.",
        no_args_is_help=True,
        add_completion=False,
    )
    app.callback()(_root_callback)
    for name, module in commands.items():
        app.add_typer(module.app, name=name)
    return app


def run(
    argv: Sequence[str],
    environ: Mapping[str, str],
    commands_package: str = COMMANDS_PACKAGE,
) -> int:
    """Run the CLI and return the exit code (never calls sys.exit)."""
    try:
        commands = discover_modules(commands_package, check_command, command_name)
        command = typer.main.get_command(build_app(commands))
        result = command.main(
            args=list(argv),
            prog_name=PROG,
            standalone_mode=False,
            obj=Invocation(argv=(PROG, *argv), environ=environ),
        )
    except ClickException as exc:
        exc.show()
        return exc.exit_code
    except typer.Abort:
        print("Aborted.", file=sys.stderr)
        return EXIT_ERROR
    except PhysverifyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    return result if isinstance(result, int) and not isinstance(result, bool) else 0


def main() -> None:
    sys.exit(run(sys.argv[1:], os.environ))
