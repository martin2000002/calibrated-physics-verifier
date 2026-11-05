"""Configuration and cache-key hashes: they are written into run.json, the sealed ledger, dataset
plans and the stage cache, so every table and key name and every value must hash the same."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator, Mapping
from functools import partial
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
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.routes import route_plans
from physverify.schema.clip import resolve_sampling
from physverify.schema.io import stable_hash
from physverify.schema.stages import Stage
from physverify.synth.settings import load_dataset_settings
from physverify.verification.check_routes import check_routing

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
KEY_PATHS = Path(__file__).with_name("data") / "config_key_paths.txt"
PINNED_ROOT = "/pinned"

SECTIONS_HASH = {
    "fast": "05b202497ced25b29e7cdfb0d850e7fce8022f4a39d3e8408d1f9681841be95c",
    "precise": "93a57bd3b0ee83af7e353c0894993cd3018a822e4ef82027deef15165a233def",
}
CONFIG_HASH = {
    "fast": "e10af4b120880ad4ded88dc91a16be5851a18dff132479c98699d4bfd89c7af4",
    "precise": "894b22cd8b08de4526d460cec5171a3148b6b77a8f96127decdf662f2dfd0c87",
}
LOOK_CONFIG_HASH = {
    "fast": "d85b2954ba75648e2391c8c3f78ec548c3102129ec56aad4bab3ef05c48436de",
    "precise": "6910144df6c63a7d6a380f2aec76176fa328d912fa66beb100f179055c00e016",
}
CHECKS_HASH = "5c5820acaf1bfce8270cdc8653a9eefd373f9feccde4baaac4b3a4510198f78e"
ROUTED_CHECKS_HASH = {
    "precise": "1cb265b2c5085d705441f409176f2f20b9a09a69e875e1cd04729068fa007ff8",
}
"""``CHECKS_HASH`` with the input routes of a profile that routes a check; a profile without
routes hashes its default checks as ``CHECKS_HASH``."""
RUN_CHECKS_HASH = {
    "fast": "5c5820acaf1bfce8270cdc8653a9eefd373f9feccde4baaac4b3a4510198f78e",
    "precise": "05f8ade4c34a9d5e374d378bd8064743086c68379a61060c1c6eb03476a3e9c9",
}
PRECISE_STAGE_KEYS_HASH = "d9b9e880768fb779384a271017fb36c975bb5e4290ebadaeee21e9f8325240a0"
"""Hash of every stage key of the precise plan for the declared calib dataset. It and
``RUN_CHECKS_HASH["precise"]`` are what the defaults give with the same choices passed by
``--set``."""
PRECISE_ROUTE_STAGE_KEYS_HASH = {
    "area_depth": "0acea35986efeac48dc7923351f183b2c05f7f7cca3470ea03d0abae05908db4",
}
"""Per input route of the precise profile, the hash of every stage key of its plan, as
``PRECISE_STAGE_KEYS_HASH`` is of the main plan's."""
PRECISE_CHOICES = (
    "extraction.plan.contacts=kinematic_scene",
    "check_variants.R4_size=v3_window_depth_unoccluded",
    "check_variants.R5a=whitened",
    "check_variants.R4_jump=rigid2_dfloor",
    "check_variants.R3a=image_apex",
    "verification.routes.area_depth.lift=backproject_area_depth",
    "verification.check_routes.R3a=area_depth",
)
DATASET_CONTENT_HASH = {
    "fast": "7792797236e163025d89c1c518dc43ad8f90c3f511bd884e1969ca9bf1f2fa3e",
    "precise": "4a6829efdd64424d79dfc0941246de148ca50446e0f6f40738e3a02d1b8e8b4e",
}
NO_IDENTITY_STAGE_KEYS_HASH = "25f48c51ab9527daec2f64adf330faa8b446edbba60a935c41c313784fcbde0a"
"""Hash of ``STAGE_KEYS`` computed for a dataset without cache identity (PISA, Physics-IQ,
Objectron): the table as it was before identities entered the key, so those caches stay valid,
except the three GeoCalib cameras, whose version moved when their draw was seeded,
``contacts/kinematic_scene``, whose version moved when its sigma_v became ``vel_rel_flight``,
the back-projecting lifts and every stage after them, whose version moved when the lift output
gained ``nearer_cover``, the size-prior scales, whose version moved when the extent sigmas of
a length became correlated axes, ``camera/moge2_focal``, whose version moved when its frame
spread took the shared robust scale, the three GeoCalib cameras again, whose params gained
the configured ``assumed`` fallback, and ``track/edgetam`` (the fast tracker) and every stage
after it, whose version moved when its backbone config was read from ``paths.hf_home``."""
STAGE_KEYS = {
    "camera/assumed_fov": "dddd1ad9189f13fabcc53f7e2fe5f8818b5d2bdeaf0900f0473a9207790e3c92",
    "camera/from_depth": "528097477f8abe02aee99fb728a9083fc5d6be7ef2565f09783422a4f7c4822d",
    "camera/geocalib": "6bbe25585e50cb7c606b7effd4f20ae9e6feefb2702cb123590051559e4de22d",
    "camera/moge2_focal": "ea90ddd70d84487f5ddd434cd9f3282d95fb38236f56adf937cf775a34289dfc",
    "contacts/kinematic": "444328611d16cbb23af836b42a6caf6ea7a122f461335244e07410248696c42d",
    "contacts/kinematic_scene": "9c5cacbf5569d08a2a479462aea235cf2ebad667d1a2642ea278b68eda913ebb",
    "contacts/proximity": "f1411d414238733b2413c347a7b41acbd5571823732d1422202383e73d146676",
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
    "contacts/ground_truth": "a30c1e4f85b2f5ea7d509b732f6351ea33ec66872a942600168221ae612280d4",
    "depth/ground_truth": "066f431b92050322c0ddd30e6593f2ed4b4867638b185fbd5c7f7eeabe28db22",
    "detect/ground_truth": "d824014b61086af84d16cf7898ff517a6f80f9d23aed8e6e52e96fe24a3520c1",
    "kinematics/ground_truth": "04b220b73f1fe06d63bba67c488c7a19d21461303db3fd358c40268a7b70622e",
    "lift/ground_truth": "dfd70e984dcb4f142db035c7931ade150a9f9c84a1011e26e586bc56eec460c1",
    "scale/ground_truth": "642036be711ca038cfe945902a053417cc0452a5bebe527f77f9d3b133e885d6",
    "track/ground_truth": "dd06a8daf8146b910fd9088f9d80f97e73b2756dc9f4e7de730aac4fe0204f4a",
    "kinematics/savgol": "30ad62422fdce4fe7ac8e95bb2442363f7885ed90a98015332be81ca5046fcff",
    "lift/backproject": "6ec94215eeef50dae4f4bd8763ab1815c27d725a8518495dd36ff4d9756a83cd",
    "lift/backproject_unnested": "fd0ce3eb05ebf53a1642ca1f05319634739e5c14313cf03679e20e1eea08f963",
    "lift/ground_anchored": "ccc1d5bcf4756858e540e6dbe42ef471ae7a1a9c7b62e2c62820d133110282dc",
    "lift/backproject_size_constancy": (
        "3d32a8d8c7a2690d3f544d7ed98a35f1392f84d45d33fe1e85082b3b417b3db1"
    ),
    "lift/backproject_area_depth": (
        "767b384cafd38367c1ed0d2d8ee33da4cbc8f8949f00148f434c06a9a0060ff7"
    ),
    "scale/free_fall": "1e5bf22be995ad138927ff211f04b850d7ac23cd16ed480316de9300378b952f",
    "scale/metric_depth": "0835076df3a2476a29569328f0f180eb2cdff267bd22c9c746f6953eb445d31c",
    "scale/size_model": "47540389d57f84942480e5c1b7a101a064415c86c251ed49af2e7654136bfc39",
    "scale/size_prior": "a1c9767ffdbebdb0f34099d20e2c59d8e20c97d4e83090f29d8f5d0eef3d83ff",
    "track/edgetam": "75bf023ca5f8704ea5c52b1aecfdf27be64448073d16c25e22610c9c8395c408",
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
    plan = resolve_plan(settings, providers, dataset=synthetic_clip.DATASET, gt_stages=gt)
    return _plan_keys(plan, dataset)


def _route_keys(
    settings: Settings, providers: Mapping[str, Any], dataset: str | None
) -> dict[str, dict[Stage, str]]:
    """The stage keys of each input route's plan."""
    resolve = partial(resolve_plan, providers=providers, dataset=synthetic_clip.DATASET)
    routes = route_plans(settings, check_routing(settings).routes, resolve)
    return {name: _plan_keys(plan, dataset) for name, plan in routes.items()}


def _plan_keys(plan: ResolvedPlan, dataset: str | None) -> dict[Stage, str]:
    clip = synthetic_clip.manifest()
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
    assert active_checks_hash(settings) == ROUTED_CHECKS_HASH.get(profile, CHECKS_HASH)
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
    identity = _calib_identity(settings)
    keys = _stage_keys(settings, providers, (), identity)
    assert run_checks_hash(settings) == RUN_CHECKS_HASH["precise"]
    assert stable_hash({s.value: k for s, k in keys.items()}) == PRECISE_STAGE_KEYS_HASH
    routes = {
        name: stable_hash({s.value: k for s, k in route.items()})
        for name, route in _route_keys(settings, providers, identity).items()
    }
    assert routes == PRECISE_ROUTE_STAGE_KEYS_HASH


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
