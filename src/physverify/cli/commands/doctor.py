"""``physverify doctor``: show the resolved environment and optionally probe the heavy lock."""

from __future__ import annotations

import platform
import time
from importlib import metadata
from typing import Annotated

import typer

from physverify.cli.context import AppContext, app_context
from physverify.core.errors import PhysverifyError
from physverify.runtime.container import container_settings, preflight_problem
from physverify.runtime.container import describe as describe_engine
from physverify.runtime.docker import PREFLIGHT_IMPORTS
from physverify.runtime.heavy import describe_holder, lock_holder_pids

NAME = "doctor"
OWNER = "scaffold"
IMPLEMENTED = True
OPTIONAL_PACKAGES = ("numpy", "torch", "transformers")

app = typer.Typer(
    invoke_without_command=True,
    help="Show configuration, paths, lock and package versions.",
)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def report_lines(context: AppContext) -> list[str]:
    settings = context.settings
    lines = [
        f"python: {platform.python_version()}",
        *(f"{name}: {package_version(name)}" for name in OPTIONAL_PACKAGES),
        f"profile: {settings.profile}",
        f"device: {settings.runtime.device}",
        f"threads: {settings.runtime.threads}",
        f"config_dir: {settings.config_dir}",
        *(f"source: {source}" for source in settings.sources),
        f"config_hash: {settings.config_hash()[:12]}",
        *(f"paths.{name}: {value}" for name, value in settings.paths.model_dump().items()),
        f"docker_image: {settings.runtime.docker_image or '(unset)'}",
        f"container_engine: {describe_engine(settings)}",
    ]
    return lines + lock_lines(context)


def lock_lines(context: AppContext) -> list[str]:
    reason = context.lock_disabled_reason()
    if reason is not None:
        return [f"heavy_lock: disabled ({reason})"]
    lock = context.heavy_lock()
    path = getattr(lock, "path", None)
    if path is None or not lock_holder_pids(path):
        return [f"heavy_lock: {path} (free)"]
    return [f"heavy_lock: {path} (held by {describe_holder(path)})"]


def container_probe(context: AppContext) -> None:
    context.settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = container_settings(context.settings).preflight_timeout_s
    problem = preflight_problem(context.docker(), timeout_s)
    if problem is not None:
        raise PhysverifyError(f"container_probe: failed; {problem}")
    typer.echo(f"container_probe: ok ({', '.join(PREFLIGHT_IMPORTS)})")


@app.callback()
def doctor(
    ctx: typer.Context,
    lock_probe: Annotated[
        float | None,
        typer.Option(help="Hold the heavy lock for this many seconds (tests the lock)."),
    ] = None,
    probe_container: Annotated[
        bool,
        typer.Option(help="Import the Kubric build's modules in the container, with its mounts."),
    ] = False,
) -> None:
    context = app_context(ctx)
    for line in report_lines(context):
        typer.echo(line)
    if probe_container:
        container_probe(context)
    if lock_probe is None:
        return
    with context.heavy_lock():
        typer.echo(f"lock-probe: holding the heavy lock for {lock_probe:g} s")
        time.sleep(lock_probe)
    typer.echo("lock-probe: released")
