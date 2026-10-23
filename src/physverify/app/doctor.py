"""The ``doctor`` service: the resolved environment, the heavy lock and the container probe."""

from __future__ import annotations

import platform

from physverify.app.context import AppContext
from physverify.core.errors import PhysverifyError
from physverify.runtime.container import container_settings, preflight_problem
from physverify.runtime.container import describe as describe_engine
from physverify.runtime.docker import PREFLIGHT_IMPORTS
from physverify.runtime.environment import package_version
from physverify.runtime.heavy import describe_holder, lock_holder_pids

OPTIONAL_PACKAGES = ("numpy", "torch", "transformers")


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


def container_probe(context: AppContext) -> str:
    """Import the Kubric build's modules in the container; the success line, or an error."""
    context.settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = container_settings(context.settings).preflight_timeout_s
    problem = preflight_problem(context.docker(), timeout_s)
    if problem is not None:
        raise PhysverifyError(f"container_probe: failed; {problem}")
    return f"container_probe: ok ({', '.join(PREFLIGHT_IMPORTS)})"
