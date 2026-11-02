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
from physverify.datasets._kubric_synthetic import content_identity
from physverify.extraction.cache import plan_entry_filter, stage_key
from physverify.schema.clip import resolve_sampling
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.settings import load_dataset_settings

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
KEY_PATHS = Path(__file__).with_name("data") / "config_key_paths.txt"
PINNED_ROOT = "/pinned"

SECTIONS_HASH = {
    "fast": "0f496b6b6c3c134833db620699a062e177149f50106563766a0f2149ef568e02",
    "precise": "b8e65567028a8480529ec972ce4fd0589256f603b6739841e0a989515c493aa2",
}
CONFIG_HASH = {
    "fast": "5c823cad1d6520fb36db4320c0d65d4f9fab8acd842e709b4891c49116dff0c4",
    "precise": "1c6c05814c905b7e92b528b5018b1184a488e353e7f1c70e56c7df506d469255",
}
LOOK_CONFIG_HASH = {
    "fast": "264a6f208d3db9bbbe0f08e4fbd73d74c264783f9219d8cc891d8f914d289953",
    "precise": "de74e813330b3859a83a3fde17c4ddd51577a8d573fcfc855eb2c15656fe5fc7",
}
CHECKS_HASH = "891fd3580169fec8f8c70ca0b7d66b24973fd7348dbf1c98ac00be92aa41b2b8"
RUN_CHECKS_HASH = {
    "fast": "891fd3580169fec8f8c70ca0b7d66b24973fd7348dbf1c98ac00be92aa41b2b8",
    "precise": "b6720c630361bdb6e65444d4562844b77ed2d2be53d7f5068eee9010cd3aebac",
}
PRECISE_STAGE_KEYS_HASH = "a8178847f34638df2acf2d789cfbf86edb29b102a608c0ee83f48ffe7502282e"
"""Hash of every stage key of the precise plan for the declared calib dataset. It and
``RUN_CHECKS_HASH["precise"]`` are what the defaults give with the same choices passed by
``--set``."""
PRECISE_CHOICES = (
    "extraction.plan.contacts=kinematic_scene",
    "check_variants.R5a=whitened",
    "check_variants.R4_jump=rigid2_dfloor",
)
DATASET_CONTENT_HASH = {
    "fast": "7792797236e163025d89c1c518dc43ad8f90c3f511bd884e1969ca9bf1f2fa3e",
    "precise": "4a6829efdd64424d79dfc0941246de148ca50446e0f6f40738e3a02d1b8e8b4e",
}
NO_IDENTITY_STAGE_KEYS_HASH = "6ade5370b65788da19532c6cd38584c44991b3949da4819f825f622fd0dc4129"
"""Hash of ``STAGE_KEYS`` computed for a dataset without cache identity (PISA, Physics-IQ,
Objectron): the table as it was before identities entered the key, so those caches stay valid."""
STAGE_KEYS = {
    "camera/assumed_fov": "53f92f90d4626f1091e3570f15005e69b3d472debd7181ecba263503c2fe6d85",
    "camera/from_depth": "32c114cba2d96137e33d17b757f8bcb6c5163eadfb2b8645ac50a901b7a7f290",
    "camera/geocalib": "c0c9fa78c8db19c78e6ca8984e3f4d5b2bd270ad6226d06a6d398ba2a83d7d2c",
    "camera/moge2_focal": "f62f2580721e6350ec104e32b086fdba2b3ff6e177aaedb2117bbc5a689ad1cf",
    "contacts/kinematic": "0d26df4538f87dd656915cafa64201a64f062c04a47b20d013e1490ca995df84",
    "contacts/kinematic_scene": "27ef2fee523e16fc20515a0222eea55e0fe21ee531c482e8eb173305189ffa5f",
    "contacts/proximity": "6a724ba07cf31ed9436d4a3975b69c0bba525ecd37538716a57b5f4346e1ea67",
    "depth/moge2": "7a667c69e1312e42a50820503500a7d84d4e877540a9acb7dd1c4ccae1012b09",
    "depth/vda_small": "a1a6a304e0e173d294247a30346164244ffd678772e5794dabe6f516f89e79b7",
    "depth/vda_large": "f832e8c3540534bb8c828d14fe7c6d0ccfd68142b3844f2af82caaba53af7851",
    "depth/vda_metric_small": "994f5ce61f266b181f46407f4de30f38dbd0955023e06e0a8fb7d25b8c2466e0",
    "detect/grounding_dino_tiny": (
        "4aaf1f926fab14d002169ed85a89c6b818d3dd6a07d7c3491795457f0565250e"
    ),
    "detect/grounding_dino_base": (
        "aca09b204541e1178b6f5bc69ffd480fff7989b122d495e987302a7f0614f6c0"
    ),
    "detect/grounding_dino_tiny_per_query": (
        "f4463392c0d99791bd7cb9d2ce7e0c0c4ccbc21d20c65c47ded467819dd6bb4d"
    ),
    "camera/ground_truth": "8f6b9478481bd12a4be19e5d92e6b81159ea5cc238bb1a0498f861c397dd48b4",
    "contacts/ground_truth": "31ff7e48e8d7186806e46f62f59a062ae77c1b0a95abf5c067e847ed23761d6b",
    "depth/ground_truth": "cf41e7e98d18c13174dde108a0015f716e50415c3cbec05e6d2ef0cff88c33a5",
    "detect/ground_truth": "c73b5ff8f835a607233cff7dfc9744b16fc89c87ad09f36d440b919e0b6b3e4c",
    "kinematics/ground_truth": "f18c6f2afe604ad22ad33bef4b7038d560c73634263894bc27f0a0adcdd91dc5",
    "lift/ground_truth": "58fbc57523514a24251abcf54d2bb446409b74da73f151b722eb93d9d1ac3d86",
    "scale/ground_truth": "07420c4a00b64542a927d20169f28b5221649efa1bb7c012aa3d62172b0c4566",
    "track/ground_truth": "977b5196333ce2b1f00e4c004ef8937490af2d49960b488f6f7edda64cd516fc",
    "kinematics/savgol": "a165f823628cc07cefaadba78afd89531fa5ba8bc1ac9d703d5707d43adccbb5",
    "lift/backproject": "e8cc671d39fe789e3d49a420c8ccf3d8bbf2c8fbe020d17b4316299d640da9f7",
    "lift/backproject_unnested": "fe90c07a750631846f45889882a45194937a4865043cb713cb4a9dc8b2db2c99",
    "lift/ground_anchored": "c4f50cfa06d8e6fe1d26e5538aa3baa95f3fde157c5f823cb3cf7a987559cad6",
    "lift/backproject_size_constancy": (
        "e808703621bbb239b8cc77699bf5e9014b81fa97c302f105ade50ba448dda596"
    ),
    "scale/free_fall": "66b092d13c6484aa70c13b89029fb78c080e9c644c1edbef4ed6a35d5059ff26",
    "scale/metric_depth": "b7f4ecb2590c4eb2e8e6e7566a0558425227f6fe8b1687fd974dd6054ed0f1fc",
    "scale/size_model": "a4c0dd5361a16ad2715d8261a24ee49d7cf34a076e32dd450565d16b550bb2d4",
    "scale/size_prior": "9d259ba1c77a32ee7561d4148eb1e2cd27e114dccab368b4d3702188b21bb779",
    "track/edgetam": "ced3cc80435c37af484f4d906a7a40b541255cfcad4d754ac7919983083041a6",
    "track/sam2_1_hiera_tiny": "3cadb052e7c25db63fc75f3091903a7567705958e213b98c0f306b5f429587b7",
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


def _calib_identity(settings: Settings) -> str:
    """The cache identity of the declared calib dataset, as ``KubricSynthetic`` reads it."""
    ds = load_dataset_settings(settings)
    return content_identity(ds.version, ds.content_hash())


def _stage_keys(
    settings: Settings,
    providers: Mapping[str, Any],
    gt: tuple[Stage, ...],
    dataset: str | None,
) -> Any:
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
            dataset=dataset,
        )
    return keys


def _pinned_key(configs: Path, providers: Mapping[str, Any], key: str, identity: bool) -> str:
    """The ``STAGE_KEYS`` entry ``key`` under the fast profile, with or without calib identity."""
    stage_name, name = key.split("/")
    stage = Stage(stage_name)
    overrides = () if name == GROUND_TRUTH else (f"extraction.plan.{stage_name}={name}",)
    settings = _settings(configs, "fast", overrides)
    dataset = _calib_identity(settings) if identity else None
    gt = (stage,) if name == GROUND_TRUTH else ()
    keys: dict[Stage, str] = _stage_keys(settings, providers, gt, dataset)
    return keys[stage]


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


@pytest.mark.parametrize("overrides", [(), PRECISE_CHOICES])
def test_the_precise_defaults_hash_as_their_explicit_choices(
    configs: Path, providers: Mapping[str, Any], overrides: tuple[str, ...]
) -> None:
    settings = _settings(configs, "precise", overrides)
    keys = _stage_keys(settings, providers, (), _calib_identity(settings))
    assert run_checks_hash(settings) == RUN_CHECKS_HASH["precise"]
    assert stable_hash({s.value: k for s, k in keys.items()}) == PRECISE_STAGE_KEYS_HASH


@pytest.mark.parametrize("key", list(STAGE_KEYS))
def test_stage_cache_keys_are_unchanged(
    configs: Path, providers: Mapping[str, Any], key: str
) -> None:
    assert _pinned_key(configs, providers, key, identity=True) == STAGE_KEYS[key]


def test_keys_without_identity_are_the_former_keys(
    configs: Path, providers: Mapping[str, Any]
) -> None:
    keys = {key: _pinned_key(configs, providers, key, identity=False) for key in STAGE_KEYS}
    assert stable_hash(keys) == NO_IDENTITY_STAGE_KEYS_HASH


def test_every_provider_has_a_pinned_stage_key(providers: Mapping[str, Any]) -> None:
    assert sorted(providers) == sorted(STAGE_KEYS)
