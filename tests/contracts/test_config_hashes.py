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
    "fast": "2dd9c7d4ef8f03af44ac3db3841c180abc99bf1b84bde91ee2b0d33b97d31c8d",
    "precise": "4ff7304afe5bb1a84036ef068054fa526e17f55574bb42107cad5ac561db61d7",
}
CONFIG_HASH = {
    "fast": "9fd0bd4b9a203ad27ffa1140075ac1995e2a955a2f1d8f4bff9d96900ad46e6d",
    "precise": "13a70759587c1b4e59b0ae77a485afb94e0745472aef07d6e5d0dbd139b38cda",
}
LOOK_CONFIG_HASH = {
    "fast": "7e4bf252235f348158fb6adad4e3a694606ea64b36092149e9771af57a8ffeea",
    "precise": "0365aca17f9ce9185b0d39d682d221d49a177957cc362a74e974379e8f19ecc8",
}
CHECKS_HASH = "4a529448fb1f7ee61d9f0550eaf132e1dc29bdaa4fdfe0ca4c825eb3733cdad2"
RUN_CHECKS_HASH = {
    "fast": "4a529448fb1f7ee61d9f0550eaf132e1dc29bdaa4fdfe0ca4c825eb3733cdad2",
    "precise": "44d28133ca495c629fbae87d0ffd9d3df5881dbc2a823ea86009ddf3b9ebe1fd",
}
PRECISE_STAGE_KEYS_HASH = "d79978ac3fef70c1d181ba0ea6e787a0010e1c3180a53c025718511fc69193f9"
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
NO_IDENTITY_STAGE_KEYS_HASH = "79ab094ec18f759f884b66dd2d9432d6e0795fa48fac91bafc532751ad6128b8"
"""Hash of ``STAGE_KEYS`` computed for a dataset without cache identity (PISA, Physics-IQ,
Objectron): the table as it was before identities entered the key, so those caches stay valid,
except the three GeoCalib cameras, whose version moved when their draw was seeded, and
``contacts/kinematic_scene``, whose version moved when its sigma_v became ``vel_rel_flight``."""
STAGE_KEYS = {
    "camera/assumed_fov": "dddd1ad9189f13fabcc53f7e2fe5f8818b5d2bdeaf0900f0473a9207790e3c92",
    "camera/from_depth": "65273a1f47e29c4e6da046d5e8ccf8f256367ec663625ef9ce92d6322c5c1d2d",
    "camera/geocalib": "1ff25d84dfcf9a37fb3d0d88440068c3cb949b2cf25846339e5039b4abffc7d6",
    "camera/moge2_focal": "bc46182cb78ba3b6a303a702c794b03f1c9d99f07d336dedecca6776b73cfca7",
    "contacts/kinematic": "9c78abca44f4fa0c275626230c5c6b59078cf8eaaa7727428ee1dfcb17817f4a",
    "contacts/kinematic_scene": "b2b15dd098ffb91089b702f77dbc7194ae26ad83dd530c96d2cb45e9d03769ba",
    "contacts/proximity": "acb367e5702bfbfe704f03107617abc106427c539108c4cd6f24200b7e7a47e7",
    "depth/moge2": "d8f98cce650ba75e04081abd3cc6f0ef8ca9e0607e80dd0d778782e9a4dce4f0",
    "depth/vda_small": "932e5bdd1e8e8e979e6ddd6f711dc3847a7a2c857b6e40edc6569cfa931a558b",
    "depth/vda_large": "bf81dd186e137ffe10de96898b7972a0acbb592ddcc836c9ae8ca515577a02cf",
    "depth/vda_metric_small": "e9fa771cc927d0b8ef5f358b1eb68044de8625d6f79e781c0af7e3834d5bdc24",
    "detect/grounding_dino_tiny": (
        "23240a4f9932e597c65e10ae4171b62a152299d2333a61c463834fca00bf0025"
    ),
    "detect/grounding_dino_base": (
        "337f2a8a00931321151e2f019bcbbf0aa6368c94d8bf5d42a5d50fe9891b95ac"
    ),
    "detect/grounding_dino_tiny_per_query": (
        "c37ca9325ddfc1d25e76b62170ea4d061624556f1a950d37b527b1858531a2eb"
    ),
    "camera/ground_truth": "b2c14c9deb33cb4ba8222a4b376ebf5fcd654fa4c2a295753670d5ca3f7dcfa2",
    "contacts/ground_truth": "601125381219f830f0b062894e48544be5982bf038573c77bb27c3529b03ad73",
    "depth/ground_truth": "066f431b92050322c0ddd30e6593f2ed4b4867638b185fbd5c7f7eeabe28db22",
    "detect/ground_truth": "d824014b61086af84d16cf7898ff517a6f80f9d23aed8e6e52e96fe24a3520c1",
    "kinematics/ground_truth": "5faaa8c2ef1ae37daa1c9dfd2146b5df22a721cde79b56a6b35ee8deb7ccb9f7",
    "lift/ground_truth": "028c544e603d5ba3ec79640dec0ec1864058bccb1b2f3732ad3d42497fecfe36",
    "scale/ground_truth": "da1e2436527b2b214554bb121d10154bc35b81b312bbb5967c84c5291fb82c3b",
    "track/ground_truth": "dd06a8daf8146b910fd9088f9d80f97e73b2756dc9f4e7de730aac4fe0204f4a",
    "kinematics/savgol": "b96b897c7d194ad913ad978d55faa5acf8e64d2af93922cec03a54c28e69bc56",
    "lift/backproject": "e68c48eb79726c78a52e32efa6715fb49060627df4ae422c6036232475ef6f50",
    "lift/backproject_unnested": "e975151ba2786469eabc2360ca9b74888b5035cd8e640e5dc6c77a9d5af13032",
    "lift/ground_anchored": "b91963fce552bd4273f6e3fdad68d804766cbe9eac619f5e0f68edf770a71d20",
    "lift/backproject_size_constancy": (
        "29d7b6390610e4f8b6235852b585a94165ebc5c51ef0ba8e74593215983268ed"
    ),
    "scale/free_fall": "5a36e4164e06aa38361ff17cb2f869532ec98cc032bcfbd176161c618ae392fb",
    "scale/metric_depth": "a0b43e7f9792067936f7f3ad893d94bc05fef6fc6b2c75f0630a876b81ead9c8",
    "scale/size_model": "c262b9b7480c2f567697afefa1550f7a8323539c54ee17a561ba17e88c2ad17a",
    "scale/size_prior": "75ea5611d5250240777c35fecbb03117d63a6a304c2b8ede75c288a9e69b4a5a",
    "track/edgetam": "d8a575fa696b6b5d3c81d0a29f1a6c38410b86fea8a59afd36c635ccaa0c546e",
    "track/sam2_1_hiera_tiny": "a38a8223ecde1e228b7b30f4889972cb07277c5effebb96451b787dff76557bf",
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
    return content_identity(ds.version, ds.content_hash(), blind_queries=ds.blind_queries)


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
