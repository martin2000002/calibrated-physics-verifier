"""Per-clip DETECT/DEPTH statistics of a diagnosis dataset, from the application context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import chain_runner, dataset_for, pipeline_deps
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.diagnosis import scene_stats
from physverify.diagnosis.context import sampled_metrics
from physverify.diagnosis.settings import diagnosis_settings

STAGES = ("detect", "depth")


def collect_per_clip(
    app_ctx: AppContext,
    command: str,
    label: str,
    out: Path,
    *,
    dataset: str | None,
    subset: str | None,
    clips: Sequence[str],
    err: Echo,
) -> int:
    """Collect and save the per-clip rows of the selected clips; returns how many were written.
    ``command`` names the caller in the sealed ledger."""
    settings = app_ctx.settings
    update = {"dataset": dataset, "subset": subset}
    diag = diagnosis_settings(settings).model_copy(
        update={key: value for key, value in update.items() if value is not None}
    )
    source = dataset_for(app_ctx, diag.dataset)
    selected = select_clips(source.clips(diag.subset), clips)
    guard_sealed(
        app_ctx,
        command,
        [Selection(source.name, diag.subset, tuple(selected))],
        err=err,
    )
    metrics = sampled_metrics(app_ctx.registry("metrics"), diag.metrics)
    runner = chain_runner(app_ctx, pipeline_deps(app_ctx, None), source)
    rows = scene_stats.collect_rows(runner, selected, metrics, diag, STAGES)
    collected = scene_stats.Collected(
        label=label, profile=settings.profile, overrides=app_ctx.options.overrides, rows=tuple(rows)
    )
    scene_stats.save(out, collected)
    return len(rows)
