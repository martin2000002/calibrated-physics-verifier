"""A re-rendered clip loses every cached stage, in every cache root, and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

from cli_runner import CliRunner
from support import truth_outputs
from synthetic_clip import SyntheticClip
from test_cache import meta_for

from physverify.extraction.cache import StageCache
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.stages import Stage
from physverify.synth.migrate import COMPLETED_KEY, RERENDER_KEY

DATASET = "ds"
STAGES = (Stage.SCALE, Stage.CONTACTS)


def fill(cache: StageCache, clip: SyntheticClip, clip_ids: tuple[str, ...]) -> None:
    outputs = truth_outputs(clip)
    for clip_id in clip_ids:
        for stage in STAGES:
            for provider in ("p", "q"):
                key = f"{stage.value}-{provider}-0123456789"
                meta = meta_for(stage, provider, key, clip_id)
                cache.store(DATASET, clip_id, stage, outputs[stage], meta, BackprojectLift.Params())


def completed(manifest: dict[str, object]) -> dict[str, object]:
    return manifest | {COMPLETED_KEY: "2026-10-19T00:00:00+00:00"}


def cached(cache: StageCache, clip_id: str) -> list[bool]:
    return [
        cache.contains(DATASET, clip_id, stage, provider, f"{stage.value}-{provider}-0123456789")
        for stage in STAGES
        for provider in ("p", "q")
    ]


def test_purge_removes_every_stage_of_the_clip_only(
    fixture_clip: SyntheticClip, tmp_path: Path
) -> None:
    cache = StageCache(tmp_path)
    fill(cache, fixture_clip, ("a", "b"))
    assert cache.purge_clip(DATASET, "a") == 4
    assert not any(cached(cache, "a"))
    assert all(cached(cache, "b"))
    assert cache.purge_clip(DATASET, "a") == 0
    assert cache.purge_clip("other", "b") == 0
    assert all(cached(cache, "b"))


def test_metas_lists_the_meta_of_every_entry_of_the_clip(
    fixture_clip: SyntheticClip, tmp_path: Path
) -> None:
    cache = StageCache(tmp_path)
    fill(cache, fixture_clip, ("a", "b"))
    metas = cache.metas(DATASET, "a")
    assert len(metas) == len(cache.entries(DATASET, "a")) == 4
    assert all(p.name == "meta.json" and "/a/" in p.as_posix() for p in metas)
    assert cache.metas(DATASET, "missing") == []


def test_extract_purge_reads_the_manifest_and_covers_every_root(
    cli: CliRunner, fixture_clip: SyntheticClip, tmp_path: Path
) -> None:
    roots = (tmp_path / "one", tmp_path / "two")
    for root in roots:
        fill(StageCache(root), fixture_clip, ("a", "b", "c"))
    manifest = tmp_path / "velocity-position.json"
    manifest.write_text(json.dumps(completed({"dataset": DATASET, RERENDER_KEY: ["a", "c"]})))
    args = [arg for root in roots for arg in ("--cache-root", str(root))]
    result = cli("extract", "purge", "--clips-from", str(manifest), *args)
    assert result.code == 0, result.err
    assert result.out.count("8 stage entries of 2 clips removed") == 2
    for root in roots:
        cache = StageCache(root)
        assert not any(cached(cache, "a") + cached(cache, "c"))
        assert all(cached(cache, "b"))


def test_extract_purge_without_a_manifest_is_a_usage_error(cli: CliRunner, tmp_path: Path) -> None:
    result = cli("extract", "purge", "--clips-from", str(tmp_path / "missing.json"))
    assert result.code == 2
    assert "no migration manifest" in result.err


def test_extract_purge_refuses_a_migration_not_completed(
    cli: CliRunner, fixture_clip: SyntheticClip, tmp_path: Path
) -> None:
    fill(StageCache(tmp_path / "root"), fixture_clip, ("a",))
    manifest = tmp_path / "velocity-position.json"
    manifest.write_text(json.dumps({"dataset": DATASET, RERENDER_KEY: ["a"]}))
    result = cli(
        "extract", "purge", "--clips-from", str(manifest), "--cache-root", str(tmp_path / "root")
    )
    assert result.code != 0
    assert COMPLETED_KEY in result.err
    assert all(cached(StageCache(tmp_path / "root"), "a"))
