"""Per-clip DETECT/DEPTH statistics of a diagnosis dataset, from the application context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.diagnosis import diag_settings
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import chain_runner, dataset_for, pipeline_deps
from physverify.config.plugin_context import registry_of
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.diagnosis import scene_stats
from physverify.diagnosis._base import METRICS
from physverify.diagnosis.context import sampled_metrics

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
    diag = diag_settings(settings, dataset=dataset, subset=subset)
    source = dataset_for(app_ctx, diag.dataset)
    selected = select_clips(source.clips(diag.subset), clips)
    guard_sealed(
        app_ctx,
        command,
        [Selection(source.name, diag.subset, tuple(selected))],
        err=err,
    )
    metrics = sampled_metrics(registry_of(app_ctx, METRICS), diag.metrics)
    runner = chain_runner(app_ctx, pipeline_deps(app_ctx, None), source)
    rows = scene_stats.collect_rows(runner, selected, metrics, diag, STAGES)
    collected = scene_stats.Collected(
        label=label, profile=settings.profile, overrides=app_ctx.options.overrides, rows=tuple(rows)
    )
    scene_stats.save(out, collected)
    return len(rows)


def print_tables(files: Sequence[Path], *, echo: Echo) -> None:
    """The header, then the pooled and per-scene rows of every collected file."""
    for line in scene_stats.TABLE_HEADER:
        echo(line)
    for path in files:
        for line in scene_stats.table_lines(scene_stats.load(path)):
            echo(line)


def delta_query(split: str, metric: str, quantity: str, stat: str) -> scene_stats.DeltaQuery:
    """The query of ``scene-stats delta``; ValueError when ``stat`` is not a known statistic."""
    return scene_stats.DeltaQuery(
        split=split, metric=metric, quantity=quantity, stat=scene_stats.parse_stat(stat)
    )


def delta_line(
    base: Path, candidate: Path, query: scene_stats.DeltaQuery, n: int, seed: int, ci: float
) -> str:
    """``candidate - base`` of ``query`` over the scenes both files share, with its bootstrap
    interval."""
    result = scene_stats.scene_delta(
        scene_stats.load(base).rows, scene_stats.load(candidate).rows, query, n, seed, ci
    )
    return (
        f"{query.metric} {query.quantity} {query.stat} ({query.split}, {result.scenes} scenes): "
        f"delta {result.point:+.4f}, {ci:.0%} CI [{result.low}, {result.high}]"
    )
