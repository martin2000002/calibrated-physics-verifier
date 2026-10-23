"""J9 of plan-congelado: ``kubric_fresh`` over a store is ``kubric_synthetic`` over it, except the
dataset name and the cache root; the blind and OOD variants only change what they declare."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from wiring import resolve_plan

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.settings import calibration_config
from physverify.calibration.table import build_rows
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets._base import DatasetSource
from physverify.datasets.kubric_fresh import KubricFresh, KubricFreshBlind, KubricFreshOod
from physverify.datasets.kubric_synthetic import KubricSynthetic
from physverify.extraction.cache import StageCache
from physverify.extraction.pipeline import (
    ExtractionPipeline,
    ExtractionReport,
    PipelineDeps,
)
from physverify.extraction.pipeline_resources import LazyLock, ProcRssProbe
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import STAGE_ORDER
from physverify.synth.settings import load_dataset_settings
from physverify.verification.check_set import ChecksSettings, build_checks
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner
from physverify.verification.sweep import CheckUnit

HERE = str(Path(__file__).parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_gt_export import exported_smoke  # noqa: E402

SYNTHETIC = KubricSynthetic.name
FRESH = KubricFresh.name


@pytest.fixture
def smoke_env(
    data_settings: Settings,
    families: Any,
    perturbations: Any,
    truth_measures: Any,
    environ: dict[str, str],
) -> dict[str, str]:
    """The exported smoke store, seen by ``kubric_fresh`` through its own section."""
    exported_smoke(data_settings, families, perturbations, truth_measures)
    version = load_dataset_settings(data_settings).version
    return environ | {
        "PHYSVERIFY__PATHS__DATA_DIR": str(data_settings.paths.data_dir),
        "PHYSVERIFY__KUBRIC_FRESH__VERSION": version,
        "PHYSVERIFY__KUBRIC_FRESH__OOD_VERSION": version,
    }


def fresh_settings(config_dir: Path, env: Mapping[str, str], **extra: str) -> Settings:
    return load_settings(config_dir, environ={**env, **extra})


def unlabelled(clips: Sequence[ClipManifest]) -> list[dict[str, Any]]:
    return [{k: v for k, v in c.to_dict().items() if k != "dataset"} for c in clips]


def oracle_report(dataset: DatasetSource, clips: Sequence[ClipManifest]) -> Any:
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings({}))
    return VerificationRunner(checks, RawPolicy(), {}).run(OracleSource(dataset), clips)


def ground_truth_chain(
    settings: Settings, dataset: DatasetSource, clips: Sequence[ClipManifest]
) -> ExtractionReport:
    plan = resolve_plan(
        settings, PLUGIN_KINDS["providers"](), dataset=dataset.name, gt_stages=STAGE_ORDER
    )
    deps = PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=LazyLock(nullcontext),
        paths=settings.paths,
        device="cpu",
        threads=1,
        error_profile=None,
        observer=None,
        rss=ProcRssProbe(),
    )
    return ExtractionPipeline(plan, deps).run(dataset, clips)


def test_fresh_manifests_are_the_synthetic_ones_under_its_own_name(
    config_dir: Path, smoke_env: dict[str, str]
) -> None:
    settings = fresh_settings(config_dir, smoke_env)
    synthetic, fresh = KubricSynthetic(settings), KubricFresh(settings)
    for subset in (None, "calib", "eval", "smoke"):
        mine, theirs = fresh.clips(subset), synthetic.clips(subset)
        assert unlabelled(mine) == unlabelled(theirs)
        assert {c.dataset for c in mine} <= {FRESH}
        assert {c.dataset for c in theirs} <= {SYNTHETIC}
    assert fresh.clips("smoke")


def test_fresh_oracle_inputs_and_calibration_rows_match(
    config_dir: Path, smoke_env: dict[str, str]
) -> None:
    settings = fresh_settings(config_dir, smoke_env)
    synthetic, fresh = KubricSynthetic(settings), KubricFresh(settings)
    theirs = oracle_report(synthetic, synthetic.clips("smoke"))
    mine = oracle_report(fresh, fresh.clips("smoke"))
    assert [r.to_dict() for r in mine.results] == [r.to_dict() for r in theirs.results]
    assert any(r.magnitude is not None for r in mine.results)
    registry = PLUGIN_KINDS["checks"]()
    units = tuple(CheckUnit.of(registry[c]) for c in sorted(registry))
    couplings = calibration_config(settings).couplings
    rows = [build_rows(r, units, couplings, InputLevel.ORACLE) for r in (mine, theirs)]
    assert rows[0] == rows[1]
    assert rows[0]


def test_fresh_stage_keys_match_and_only_the_cache_root_moves(
    config_dir: Path, smoke_env: dict[str, str]
) -> None:
    settings = fresh_settings(config_dir, smoke_env)
    synthetic, fresh = KubricSynthetic(settings), KubricFresh(settings)
    theirs = ground_truth_chain(settings, synthetic, synthetic.clips("smoke"))
    mine = ground_truth_chain(settings, fresh, fresh.clips("smoke"))
    assert not mine.failures and not theirs.failures
    assert mine.keys and mine.keys == theirs.keys
    assert not any(event.cached for event in mine.events)
    cache = StageCache(settings.paths.cache_dir)
    for (clip_id, stage), key in mine.keys.items():
        provider = next(e.provider for e in mine.events if (e.clip_id, e.stage) == (clip_id, stage))
        own = cache.directory(FRESH, clip_id, stage, provider, key)
        other = cache.directory(SYNTHETIC, clip_id, stage, provider, key)
        assert own.relative_to(cache.root / FRESH) == other.relative_to(cache.root / SYNTHETIC)
        assert own.is_dir() and other.is_dir()


def test_fresh_sampling_follows_kubric_synthetic(
    config_dir: Path, smoke_env: dict[str, str]
) -> None:
    settings = fresh_settings(config_dir, smoke_env)
    providers = PLUGIN_KINDS["providers"]()
    reference = resolve_plan(settings, providers, dataset=SYNTHETIC).sampling
    for cls in (KubricFresh, KubricFreshOod, KubricFreshBlind):
        assert resolve_plan(settings, providers, dataset=cls.name).sampling == reference


def test_ood_reads_its_own_version(config_dir: Path, smoke_env: dict[str, str]) -> None:
    moved = fresh_settings(config_dir, smoke_env, PHYSVERIFY__KUBRIC_FRESH__OOD_VERSION="absent")
    assert KubricFreshOod(moved).clips() == []
    settings = fresh_settings(config_dir, smoke_env)
    ood = KubricFreshOod(settings).clips()
    assert unlabelled(ood) == unlabelled(KubricSynthetic(settings).clips())
    assert {c.dataset for c in ood} == {KubricFreshOod.name}


def test_blind_keeps_its_scenes_and_withholds_the_counts(
    config_dir: Path, smoke_env: dict[str, str]
) -> None:
    outside = fresh_settings(config_dir, smoke_env)
    assert KubricFreshBlind(outside).clips() == []
    settings = fresh_settings(
        config_dir, smoke_env, PHYSVERIFY__KUBRIC_FRESH__BLIND_SCENES='["s000"]'
    )
    blind, seen = KubricFreshBlind(settings).clips(), KubricFresh(settings).clips()
    assert [c.clip_id for c in blind] == [c.clip_id for c in seen]
    for mine, theirs in zip(blind, seen, strict=True):
        assert mine.dataset == KubricFreshBlind.name
        assert [q.phrase for q in mine.object_queries] == [q.phrase for q in theirs.object_queries]
        assert all(q.expected_count is None for q in mine.object_queries)
        assert any(q.expected_count is not None for q in theirs.object_queries)


def test_fresh_cannot_be_fetched(
    config_dir: Path, smoke_env: dict[str, str], tmp_path: Path
) -> None:
    with pytest.raises(UsageError, match="generated locally"):
        KubricFresh(fresh_settings(config_dir, smoke_env)).fetch("full", tmp_path)
