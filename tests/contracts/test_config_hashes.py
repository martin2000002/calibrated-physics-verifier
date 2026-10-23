"""Configuration and cache-key hashes: they are written into run.json, the sealed ledger, dataset
plans and the stage cache, so every table and key name and every value must hash the same."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import synthetic_clip

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.sealed import look_config_hash
from physverify.calibration.settings import active_checks_hash
from physverify.config.loader import load_settings, read_config_files
from physverify.config.models import PathsSettings, Settings
from physverify.extraction.cache import plan_entry_filter, stage_key
from physverify.extraction.plan import resolve_plan
from physverify.schema.clip import resolve_sampling
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.settings import load_dataset_settings

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
KEY_PATHS = Path(__file__).with_name("data") / "config_key_paths.txt"
PINNED_ROOT = "/pinned"

SECTIONS_HASH = {
    "fast": "b2fb086c4de2d8fffea17f7f9cf9d7fdf1dfe0f3919bc38c606e8fd4e4b783f3",
    "precise": "4eafd952007e25f2e70e3bccb531aeaf9b304097b2c6860dde9ad6b3e28b2e2a",
}
CONFIG_HASH = {
    "fast": "561ed6828c6eb0f636b31a263ba0251d1c135798881c12ed98dd0acfd3614cdd",
    "precise": "26fe9629fa97588c4de1aa25ce94c379760843dd4f6983229ce32ef0cbd0d7f4",
}
LOOK_CONFIG_HASH = {
    "fast": "51ab92a81b6009382de998d9e67a835176502e425bdd1006ed1332b47915a71b",
    "precise": "0daf6d0bbcee1416cc6492f82828850dc380e69a6ead59a48b021d55025a3bab",
}
CHECKS_HASH = "c5f03bfd100bdd23fe90a5f2ad098b2767b893a75bb850fd42c4116312ca3dee"
DATASET_CONTENT_HASH = {
    "fast": "7792797236e163025d89c1c518dc43ad8f90c3f511bd884e1969ca9bf1f2fa3e",
    "precise": "4a6829efdd64424d79dfc0941246de148ca50446e0f6f40738e3a02d1b8e8b4e",
}
STAGE_KEYS = {
    "camera/assumed_fov": "d0ae25696b1da8e41856f1aaf094085c01f865833bc905f253b56546f7477018",
    "camera/geocalib": "871980e0b1985e5945e7620477d9ea8b3a6781b5b215e2d70ef9c2868b17d6e1",
    "contacts/kinematic": "5fa79bdb79bc2e1d112406b88d1c86cb8084cabe939911842fc9b94043e7f88b",
    "contacts/proximity": "d0b01ca2029fe279052af5dff2c6912e1d7761709b4b156133bd6ada5cdf8443",
    "depth/vda_small": "70f6c003ff50af82ae87a48aa6a2c81030761e6a0c859278fc99fd8dba78cf1c",
    "depth/vda_large": "5f9dc8fa05d3d3be2711507ea24c8d730f337ca0d1584513c2c4677d98509b13",
    "depth/vda_metric_small": "ccc78054bf3ca5553e4bb71716289309a5b2ab609836225cf252b782f3c5b5fe",
    "detect/grounding_dino_tiny": (
        "07d10de3f9dd857bfff1085bcb432229b7e8715ce167ef6c60c00dafbe942873"
    ),
    "detect/grounding_dino_base": (
        "6294aa52ed26a80584610f39638f21797ad425b46f52e2f004b44b8391d78b70"
    ),
    "detect/grounding_dino_tiny_per_query": (
        "c5d3b85789817de2d6b754cd0d07fe8d0bbd829118ce96a201887376f4224f6d"
    ),
    "camera/ground_truth": "dcc9f3896dfae49cd3ef2203bda5922a6d2658e10742995834a8a97e79e756e9",
    "contacts/ground_truth": "c19862f11db0d6e64c7c3fa030a2b578544b4fbc91fe39797d3f392f48983146",
    "depth/ground_truth": "5e98a6b578a3276bec41b24e5adc10c15b4376be94b8c2cc6b60a89572a5c107",
    "detect/ground_truth": "a62f5932c2390f802e0e37c316b245d6d790dde1ef5f17326d53382a39d142fe",
    "kinematics/ground_truth": "022f0acb3dce3d1ec694cfe18446d6f513bc856340c607c9db8dacfa17a87a40",
    "lift/ground_truth": "fd9112fb74e8dd281c76ae277a0ee4782161166f15cee466b834c0f5eeaa79a5",
    "scale/ground_truth": "5e7a35d2b2d8a8b0a24a1226659a370cf009dcdc46663efd464128f772ca6912",
    "track/ground_truth": "7f0b4774a68b28d2fd70ac39fe6a15dabc0634cf9046983c158734a331523cd5",
    "kinematics/savgol": "7c1bd68862faaae42d003e987af7401d0a74175e012ab1c66302984054a09f1c",
    "lift/backproject": "56d1e0e2760b107626022203c636669d3dfe69ae93b80a2772b0709748bf6700",
    "scale/free_fall": "bec820adaf82685969af695611b63820d111490ba005aacbb9593de29c1285be",
    "scale/size_model": "a079f4cdd56f2719d0e17ae493f7e07e2d54f75250468d8f0c2ac65d4a71a906",
    "scale/size_prior": "5f60ea1c1ea1da9cb356fdc44c6c616f515b18a6730e8c3039eb809f6521166b",
    "track/edgetam": "81a150661413d4a8e45d79ab5f2dd7a3c4b2ca10db56a5bade893403801ce4d8",
    "track/sam2_1_hiera_tiny": "88cf35cd5a540158b73d940af239b84d357dd091f5b2935f884a87671d14dbee",
}
GROUND_TRUTH = "ground_truth"


@pytest.fixture(scope="module")
def configs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("contracts")
    target = root / "repo" / "configs"
    shutil.copytree(REPO_CONFIGS, target, ignore=shutil.ignore_patterns("local*"))
    return target


@pytest.fixture(scope="module")
def providers() -> Mapping[str, Any]:
    return PLUGIN_KINDS["providers"]()


def _settings(configs: Path, profile: str, overrides: tuple[str, ...] = ()) -> Settings:
    home = configs.parent.parent / "home"
    return load_settings(configs, environ={"HOME": str(home)}, profile=profile, overrides=overrides)


def _relocated(settings: Settings, root: Path) -> Settings:
    """``settings`` with its anchored paths moved from ``root`` to a fixed directory."""
    moved = {
        name: Path(str(value).replace(str(root), PINNED_ROOT, 1)) for name, value in settings.paths
    }
    return dataclasses.replace(settings, paths=PathsSettings(**moved))


def _key_paths(table: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in table.items():
        if isinstance(value, Mapping):
            yield from _key_paths(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}"


def _stage_keys(settings: Settings, providers: Mapping[str, Any], gt: tuple[Stage, ...]) -> Any:
    clip = synthetic_clip.manifest()
    plan = resolve_plan(settings, providers, dataset=synthetic_clip.DATASET, gt_stages=gt)
    sampling = resolve_sampling(clip, plan.sampling)
    accept = plan_entry_filter(plan)
    keys: dict[Stage, str] = {}
    for step in plan.stages:
        keys[step.stage] = stage_key(
            step,
            clip_id=clip.clip_id,
            object_queries=clip.object_queries,
            sampling=sampling,
            upstream=keys,
            plan=plan.names,
            entry_filter=accept,
            gt_version=synthetic_clip.GT_VERSION if step.provider.is_ground_truth else None,
            error_profile=None,
            device="cpu",
        )
    return keys


def test_config_key_paths_are_unchanged(configs: Path) -> None:
    merged, _ = read_config_files(configs)
    expected = KEY_PATHS.read_text(encoding="utf-8").splitlines()
    assert sorted(_key_paths(merged)) == expected


@pytest.mark.parametrize("profile", sorted(SECTIONS_HASH))
def test_section_hashes_are_unchanged(configs: Path, profile: str) -> None:
    settings = _settings(configs, profile)
    assert stable_hash(dict(settings.sections)) == SECTIONS_HASH[profile]
    assert look_config_hash(settings) == LOOK_CONFIG_HASH[profile]
    assert active_checks_hash(settings) == CHECKS_HASH
    assert load_dataset_settings(settings).content_hash() == DATASET_CONTENT_HASH[profile]


@pytest.mark.parametrize("profile", sorted(CONFIG_HASH))
def test_the_run_config_hash_is_unchanged(configs: Path, profile: str) -> None:
    settings = _relocated(_settings(configs, profile), configs.parent.parent)
    assert settings.config_hash() == CONFIG_HASH[profile]


@pytest.mark.parametrize("key", list(STAGE_KEYS))
def test_stage_cache_keys_are_unchanged(
    configs: Path, providers: Mapping[str, Any], key: str
) -> None:
    stage_name, name = key.split("/")
    stage = Stage(stage_name)
    if name == GROUND_TRUTH:
        keys = _stage_keys(_settings(configs, "fast"), providers, (stage,))
    else:
        settings = _settings(configs, "fast", (f"extraction.plan.{stage_name}={name}",))
        keys = _stage_keys(settings, providers, ())
    assert keys[stage] == STAGE_KEYS[key]


def test_every_provider_has_a_pinned_stage_key(providers: Mapping[str, Any]) -> None:
    assert sorted(providers) == sorted(STAGE_KEYS)
