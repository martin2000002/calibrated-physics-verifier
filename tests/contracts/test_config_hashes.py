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
    "fast": "14e97dbc49075b264097004078a5c4114498b9129742b4b3de0e5280ba97e059",
    "precise": "f6ebdef3a03d556900beb3344469d7787a29ec42aca059fe50b66a4e6ff8792e",
}
CONFIG_HASH = {
    "fast": "2de043e7423ee2a3225abecf14b9a9b092130986b793a378768a7d25fa8d44cd",
    "precise": "760d2fedaf6027cedfabc5af509df7a6df2e795da509db07e967be3994abf11d",
}
LOOK_CONFIG_HASH = {
    "fast": "43380727c29ee5a289b4ce6e220827df0b32b5ad70e0d97bb5f4389af881abcc",
    "precise": "02d997c108957851bbe49b6b8c47eee5b81e19201b68e18aad6df05854406856",
}
CHECKS_HASH = "1bc070f3c0236c60860077d46926888188c5e86fc6c4132c0274e43ed138b4b2"
RUN_CHECKS_HASH = {
    "fast": "1bc070f3c0236c60860077d46926888188c5e86fc6c4132c0274e43ed138b4b2",
    "precise": "cd37678aef95992ca6d07e0491a500a641d91a9b0a861ccfb7457fd3e2367470",
}
DATASET_CONTENT_HASH = {
    "fast": "7792797236e163025d89c1c518dc43ad8f90c3f511bd884e1969ca9bf1f2fa3e",
    "precise": "4a6829efdd64424d79dfc0941246de148ca50446e0f6f40738e3a02d1b8e8b4e",
}
STAGE_KEYS = {
    "camera/assumed_fov": "d0ae25696b1da8e41856f1aaf094085c01f865833bc905f253b56546f7477018",
    "camera/from_depth": "71f47572478cef19abbac16c095575678d6703941e26128139ef90cac930aae0",
    "camera/geocalib": "871980e0b1985e5945e7620477d9ea8b3a6781b5b215e2d70ef9c2868b17d6e1",
    "contacts/kinematic": "4637a961f2a230c25a84b686c3a69ccf911bb7b8c59c7c694f69f43acd8f2946",
    "contacts/proximity": "c9674d131e269a5bec255e694111ce3a8fbd5def115452732578a62ee0260391",
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
    "contacts/ground_truth": "a675f7750e15e1b97fb61ee6a5f6a63aa6e946a4798a1314763130f86fcad4c0",
    "depth/ground_truth": "06c3b85ad942509b944137bd664d5d865d99c6cdb7441854de17f2860afaf9be",
    "detect/ground_truth": "a62f5932c2390f802e0e37c316b245d6d790dde1ef5f17326d53382a39d142fe",
    "kinematics/ground_truth": "96bca9a793fc4e15160bbc86de272bb6cffa9db87ad9c1e1abbce9966af49caf",
    "lift/ground_truth": "fd9112fb74e8dd281c76ae277a0ee4782161166f15cee466b834c0f5eeaa79a5",
    "scale/ground_truth": "d02718762e5635727883f267696270c9100b481d1f19149c2dd4c96431be9b38",
    "track/ground_truth": "7f0b4774a68b28d2fd70ac39fe6a15dabc0634cf9046983c158734a331523cd5",
    "kinematics/savgol": "2a2e1a9916cc3efecd6ae8c9ed85c8260307543205f340f387d137685cc24900",
    "lift/backproject": "b79574cc0403c7d55fcfb6938df2a5928196c8e09cd7f88ceb7d939858cf7445",
    "lift/backproject_unnested": "2b203300810399325c0cabda849cd3495b44c7ba9270e2d889c94aa5eab21807",
    "scale/free_fall": "f104b2070930e8ddb30a5679208d9d9bad020cee8492f2db2426e75a72c6635b",
    "scale/metric_depth": "f05c5c101bd1c5d9b398c3c0d4b4981fd7fba388ac7377c98bfe9e2eb51a6efd",
    "scale/size_model": "6a8b94eb65880948beb7fd703a228119cb6a017e37ecec247d6ddea0e504f695",
    "scale/size_prior": "bf299851af7f355571485cea441194c2496ac32f56e8a8d0c1971c9421d7593e",
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
