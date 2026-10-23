"""``physverify extract``: run the stage plan over a dataset, fetch its weights (ADR 0005, 3.12)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.extraction import run_extraction
from physverify.app.services import plan_for
from physverify.cli.context import app_context, heavy_command
from physverify.core.download import BYTES_PER_MB
from physverify.core.errors import EXIT_ERROR, UsageError
from physverify.extraction.cache import StageCache
from physverify.extraction.events import plan_providers
from physverify.extraction.weights import NetworkDownloader, fetch, hub_cache
from physverify.synth.migrate import manifest_dataset, rerender_clip_ids

NAME = "extract"

app = typer.Typer(
    help="Perception pipeline: from video to states and contacts.",
    no_args_is_help=True,
)


@app.command("run", help="Run the extraction plan over a dataset.")
def run(
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    gt: Annotated[list[str] | None, typer.Option(help="Stages replaced by ground truth.")] = None,
    fps: Annotated[float | None, typer.Option(help="Target sampling fps.")] = None,
    max_side: Annotated[int | None, typer.Option(help="Longest frame side in pixels.")] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
) -> None:
    report = run_extraction(
        app_context(),
        dataset,
        clips or (),
        gt or (),
        fps=fps,
        max_side=max_side,
        subset=subset,
        echo=typer.echo,
    )
    if report.failures:
        raise typer.Exit(code=EXIT_ERROR)


@app.command("fetch-weights", help="Download the weights of the active profile's providers.")
@heavy_command
def fetch_weights() -> None:
    context = app_context()
    settings = context.settings
    plan = plan_for(context, dataset="")
    downloader = NetworkDownloader()
    typer.echo(f"hub cache: {hub_cache(settings.paths)}")
    for provider in plan_providers(plan):
        for spec in provider.weights:
            path = fetch(spec, settings.paths, downloader)
            size = _size_mb(path)
            typer.echo(
                f"{provider.stage.value}/{provider.name} {spec.kind.value}:{spec.ref}@"
                f"{spec.revision[:12]} verified {size:.1f} MB {path}"
            )


@app.command(
    "purge",
    help="Remove every cached stage of the clips a dataset migration re-rendered (all providers).",
)
def purge(
    clips_from: Annotated[
        Path, typer.Option(help="Migration manifest (<dataset>/migrations/<name>.json).")
    ],
    cache_root: Annotated[
        list[Path] | None,
        typer.Option(help="Cache root (repeatable; default: paths.cache_dir)."),
    ] = None,
) -> None:
    if not clips_from.is_file():
        raise UsageError(f"no migration manifest at {clips_from}")
    dataset = manifest_dataset(clips_from)
    clip_ids = rerender_clip_ids(clips_from)
    roots = cache_root or [app_context().settings.paths.cache_dir]
    for root in roots:
        cache = StageCache(root)
        entries = sum(cache.purge_clip(dataset, clip_id) for clip_id in clip_ids)
        typer.echo(f"purge {root}: {entries} stage entries of {len(clip_ids)} clips removed")


def _size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / BYTES_PER_MB
