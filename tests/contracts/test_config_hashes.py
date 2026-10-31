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
from wiring import active_checks_hash, resolve_plan, run_checks_hash

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.sealed_ledger import look_config_hash
from physverify.config.loader import load_settings, read_config_files
from physverify.config.models import PathsSettings, Settings
from physverify.extraction.cache import plan_entry_filter, stage_key
from physverify.schema.clip import resolve_sampling
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.settings import load_dataset_settings

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
KEY_PATHS = Path(__file__).with_name("data") / "config_key_paths.txt"
PINNED_ROOT = "/pinned"

SECTIONS_HASH = {
    "fast": "8b18e7aa33df27c4c45f92cbdcc7d0f13fd7dd9292694eafc62c6d6b0769c5d9",
    "precise": "9aae743684690ba0e44813b60f3606d36f3db97fac19fe5044e68ae0aaa2c11c",
}
CONFIG_HASH = {
    "fast": "6820bba02e40137056132fe8761bcaf79e8377cfd6917f34e130c7a8d389149f",
    "precise": "047039710a67bc1023183ef546b95bd869d9ef251674212b902dce34fc795a27",
}
LOOK_CONFIG_HASH = {
    "fast": "9ab2c61f0a8ab09177d1dec7853ebd400e9a0d0c39b2162c7632b561bd19a319",
    "precise": "1e3619f252926a160a500ecb6322f6dfae357e9c03089df0c3f0b3573539f46f",
}
CHECKS_HASH = "891fd3580169fec8f8c70ca0b7d66b24973fd7348dbf1c98ac00be92aa41b2b8"
RUN_CHECKS_HASH = {
    "fast": "891fd3580169fec8f8c70ca0b7d66b24973fd7348dbf1c98ac00be92aa41b2b8",
    "precise": "ea5ff7c226668081e5caa4784361a9e04c0484b39771a3e393791d530e207187",
}
DATASET_CONTENT_HASH = {
    "fast": "7792797236e163025d89c1c518dc43ad8f90c3f511bd884e1969ca9bf1f2fa3e",
    "precise": "4a6829efdd64424d79dfc0941246de148ca50446e0f6f40738e3a02d1b8e8b4e",
}
STAGE_KEYS = {
    "camera/assumed_fov": "d0ae25696b1da8e41856f1aaf094085c01f865833bc905f253b56546f7477018",
    "camera/from_depth": "71f47572478cef19abbac16c095575678d6703941e26128139ef90cac930aae0",
    "camera/geocalib": "18892cc4e5b735d5fd4b8c45ad9dccfd98b0a95814a506b28ce11079f3bf3154",
    "contacts/kinematic": "d319a6895f0622efdfae18f55811819d83ec0dbbc7071b6bf7fbfc82ea099cb0",
    "contacts/proximity": "a2167c904db6ef8bed72d48761627d915b003a0fd09d2dc4bd5df0cdff1e417b",
    "depth/moge2": "6cdce7a43eb0a8999a01982327df1202bc13bbda4c86c3af356092a578a18f81",
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
    "contacts/ground_truth": "f77e2327c1e751179a58f641ed0468b2760591606c2715a5d7c1e0a09b88772c",
    "depth/ground_truth": "06c3b85ad942509b944137bd664d5d865d99c6cdb7441854de17f2860afaf9be",
    "detect/ground_truth": "a62f5932c2390f802e0e37c316b245d6d790dde1ef5f17326d53382a39d142fe",
    "kinematics/ground_truth": "a77c9a6982bce0734d1d8c13d1d932775dd6cab4cfd6f6bab8905928b39076f7",
    "lift/ground_truth": "fd9112fb74e8dd281c76ae277a0ee4782161166f15cee466b834c0f5eeaa79a5",
    "scale/ground_truth": "50077d964a23e3ebc12bb33ebb70f7b300ef1ca805da90b01ffe60d016a7e08e",
    "track/ground_truth": "7f0b4774a68b28d2fd70ac39fe6a15dabc0634cf9046983c158734a331523cd5",
    "kinematics/savgol": "5798c40f43422b238736870ad3b4b7b053d24e431694dab1f9c293badf1b463a",
    "lift/backproject": "ebfe1a1a540435ed100cf121083da7d3f7c8f1befbae65c2fc47ed185edfa82b",
    "lift/backproject_unnested": "245f1e6635f3007505a7305cba10619b878a66c41193edd825ec43b90b9b0e05",
    "scale/free_fall": "8ca5d6987b36d4b86209600c174c65727b51e716f1fbf5550cf5ee590e1836bb",
    "scale/metric_depth": "385a7be7bddd6c74b0c89820732d00740cd85f4d192e1b54bc348bbf7689c3a8",
    "scale/size_model": "1b884cd95207324b22753351a2b7b0bb040335e9d9a7e655d00fff74baa73ce3",
    "scale/size_prior": "f22af1b73e8274993dc4678f7fd77328038a66535c895026830ad8164ee9ed28",
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
    assert run_checks_hash(settings) == RUN_CHECKS_HASH[profile]
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
