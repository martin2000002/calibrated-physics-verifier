"""Synthetic-dataset wiring: the image pull and the dataset steps behind ``physverify synth``."""

from __future__ import annotations

import subprocess

from physverify.app.context import AppContext
from physverify.core.errors import PhysverifyError, UsageError
from physverify.core.output import Echo
from physverify.runtime.container import container_settings
from physverify.synth.jobs import (
    DatasetJob,
    fail_if,
    run_export,
    run_generate,
    run_perturb,
    run_render,
    run_simulate,
    status_lines,
)
from physverify.synth.migrate import run_migrate, velocity_migration
from physverify.synth.plan import GT_MIGRATIONS as GT_MIGRATIONS


def dataset_job(context: AppContext, subset: str | None, echo: Echo) -> DatasetJob:
    """The job of one subset, with its dataset settings loaded (and validated) up front."""
    job = DatasetJob(context, subset, echo)
    _ = job.ds
    return job


def pull_image(context: AppContext) -> None:
    settings = context.settings
    engine = container_settings(settings).engine
    if engine != "docker":
        raise UsageError(
            f"pull-image only drives docker; container.engine is {engine!r} "
            "(on the H200, scripts/h200/run_kubric.sh pulls and creates the container)"
        )
    image = settings.runtime.docker_image
    if subprocess.run(["docker", "pull", image], check=False).returncode != 0:
        raise PhysverifyError(f"docker pull {image} failed")


def generate(context: AppContext, subset: str, *, echo: Echo) -> None:
    run_generate(dataset_job(context, subset, echo))


def simulate(context: AppContext, subset: str, *, echo: Echo) -> None:
    job = dataset_job(context, subset, echo)
    fail_if(run_simulate(job, job.runner()))


def perturb(context: AppContext, subset: str, *, echo: Echo) -> None:
    job = dataset_job(context, subset, echo)
    fail_if(run_perturb(job, job.runner()))


def render(
    context: AppContext, subset: str, limit: int | None, clips: str | None, *, echo: Echo
) -> None:
    """Render the pending clips; ``clips`` is a comma-separated id list."""
    job = dataset_job(context, subset, echo)
    wanted = [c.strip() for c in clips.split(",") if c.strip()] if clips else None
    fail_if(run_render(job, job.runner(), limit, wanted))


def export_gt(context: AppContext, subset: str, *, echo: Echo) -> None:
    run_export(dataset_job(context, subset, echo))


def progress_lines(context: AppContext, subset: str, *, echo: Echo) -> list[str]:
    job = dataset_job(context, subset, echo)
    return status_lines(job.plan(current=False), job.store)


def migrate(context: AppContext, name: str | None, commit: str | None, *, echo: Echo) -> None:
    """A named migration, or the velocity migration of the configured convention."""
    job = dataset_job(context, None, echo)
    run_migrate(job, name or velocity_migration(job.ds), commit)


def build(context: AppContext, subset: str, *, echo: Echo) -> None:
    """Every step of the subset in order; the step failures are raised together at the end."""
    job = dataset_job(context, subset, echo)
    runner = job.runner()
    run_generate(job)
    failures = [
        *run_simulate(job, runner),
        *run_perturb(job, runner),
        *run_render(job, runner, None, None),
    ]
    run_export(job)
    fail_if(failures)
