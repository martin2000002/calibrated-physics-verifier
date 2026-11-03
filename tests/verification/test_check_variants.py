"""[check_variants]: a selected variant replaces its check in place and keys the checks hash;
[checks."<id>/<variant>"] configures one variant, [checks.<id>] the default implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks.r4_size_constancy import SizeConstancyCheck
from physverify.checks.variants._base import DEFAULT_VARIANT
from physverify.checks.variants._rigid_jump import RigidJumpCheck
from physverify.checks.variants.r4_jump_declared_depth import DeclaredDepthJumpCheck
from physverify.checks.variants.r4_size_v3 import DetrendedSizeCheck
from physverify.checks.variants.r5a_whitened import WhitenedFallShapeCheck
from physverify.core.errors import ConfigError
from physverify.verification.check_set import (
    CHECKS_SECTION,
    ChecksSettings,
    CheckVariantsSettings,
    build_checks,
    checks_hash,
    select_variants,
    validate_overrides,
)

CHECKS = plugins.build("checks")
VARIANTS = plugins.build("check_variants")


def selected(**selection: str) -> CheckVariantsSettings:
    return CheckVariantsSettings.model_validate(selection)


def test_a_variant_takes_the_place_of_its_check() -> None:
    registry = select_variants(CHECKS, VARIANTS, selected(R4_size="v3"))
    assert list(registry) == list(CHECKS)
    assert registry["R4_size"] is DetrendedSizeCheck
    assert all(registry[c] is CHECKS[c] for c in CHECKS if c != "R4_size")


@pytest.mark.parametrize("selection", [{"R9": "v3"}, {"R4_size": "v9"}, {"R1": "v3"}])
def test_an_unknown_selection_is_a_config_error(selection: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="check_variants"):
        select_variants(CHECKS, VARIANTS, selected(**selection))


def test_the_checks_hash_tells_a_variant_from_its_default() -> None:
    overrides = ChecksSettings.model_validate({"R4_size": {"reference": "others"}})
    v3 = select_variants(CHECKS, VARIANTS, selected(R4_size="v3"))
    assert checks_hash(v3, overrides) != checks_hash(CHECKS, overrides)


def test_the_settings_select_the_active_checks(make_settings: SettingsFactory) -> None:
    default = plugins.configured("checks", make_settings())
    assert dict(default) == dict(CHECKS) and default["R4_size"] is SizeConstancyCheck
    v3 = plugins.configured("checks", make_settings("check_variants.R4_size=v3"))
    assert v3["R4_size"] is DetrendedSizeCheck


def test_the_precise_profile_runs_the_adopted_variants(make_settings: SettingsFactory) -> None:
    precise = plugins.configured("checks", make_settings(profile="precise"))
    assert precise["R4_size"] is DetrendedSizeCheck
    assert precise["R4_jump"] is DeclaredDepthJumpCheck
    assert precise["R5a"] is WhitenedFallShapeCheck
    adopted = {"R4_size", "R4_jump", "R5a"}
    assert all(precise[c] is CHECKS[c] for c in CHECKS if c not in adopted)
    assert dict(plugins.configured("checks", make_settings(profile="fast"))) == dict(CHECKS)


def test_the_default_variant_restores_the_registered_check(make_settings: SettingsFactory) -> None:
    restore = (
        f"check_variants.R4_size={DEFAULT_VARIANT}",
        f"check_variants.R4_jump={DEFAULT_VARIANT}",
        f"check_variants.R5a={DEFAULT_VARIANT}",
    )
    restored = plugins.configured("checks", make_settings(*restore, profile="precise"))
    assert dict(restored) == dict(CHECKS)


def test_the_rigid_jump_keeps_the_inputs_of_v2() -> None:
    rigid = VARIANTS["R4_jump/rigid2"]
    assert rigid.requires == CHECKS["R4_jump"].requires
    assert rigid.magnitude_unit == CHECKS["R4_jump"].magnitude_unit


def test_the_depth_floor_is_the_precise_jump_and_rigid2_stays_selectable(
    make_settings: SettingsFactory,
) -> None:
    precise = plugins.configured("checks", make_settings(profile="precise"))
    rigid = plugins.configured(
        "checks", make_settings("check_variants.R4_jump=rigid2", profile="precise")
    )
    assert issubclass(DeclaredDepthJumpCheck, RigidJumpCheck)
    assert precise["R4_jump"] is DeclaredDepthJumpCheck and rigid["R4_jump"] is RigidJumpCheck
    assert all(rigid[c] is precise[c] for c in CHECKS if c != "R4_jump")
    overrides = ChecksSettings()
    assert checks_hash(rigid, overrides) != checks_hash(precise, overrides)
    explicit = select_variants(
        CHECKS, VARIANTS, selected(R4_size="v3", R4_jump="rigid2_dfloor", R5a="whitened")
    )
    assert checks_hash(explicit, overrides) == checks_hash(precise, overrides)


def overrides(**tables: Mapping[str, object]) -> ChecksSettings:
    return ChecksSettings.model_validate(tables)


def built(registry: dict[str, type], tables: ChecksSettings, check_id: str) -> Any:
    return next(c for c in build_checks(registry, tables) if c.check_id == check_id)


def test_a_variant_table_configures_that_variant_only() -> None:
    v3 = dict(select_variants(CHECKS, VARIANTS, selected(R4_size="v3")))
    tables = overrides(**{"R4_size/v3": {"window_s": 1.5}})
    validate_overrides(CHECKS, VARIANTS, tables)
    assert built(v3, tables, "R4_size").params.window_s == 1.5
    assert checks_hash(v3, tables) != checks_hash(v3, ChecksSettings())
    assert checks_hash(dict(CHECKS), tables) == checks_hash(dict(CHECKS), ChecksSettings())


def test_a_shared_key_reaches_a_variant_only_if_it_declares_it() -> None:
    tables = overrides(R4_size={"reference": "self", "occluded_at": 0.4})
    validate_overrides(CHECKS, VARIANTS, tables)
    v3 = built(dict(select_variants(CHECKS, VARIANTS, selected(R4_size="v3"))), tables, "R4_size")
    default = built(dict(CHECKS), tables, "R4_size")
    assert "reference" not in type(v3).Params.model_fields
    assert v3.params.occluded_at == 0.4
    assert default.params.reference == "self"


@pytest.mark.parametrize(
    "tables",
    [
        {"R4_size": {"window_s": 1.5}},
        {"R4_size/v9": {}},
        {"R4_size/v3": {"reference": "others"}},
        {"R9": {}},
    ],
)
def test_a_table_is_validated_against_its_own_implementation(
    tables: dict[str, dict[str, object]],
) -> None:
    with pytest.raises(ConfigError, match=CHECKS_SECTION):
        validate_overrides(CHECKS, VARIANTS, overrides(**tables))


def test_the_settings_validate_every_table_selected_or_not(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError, match="window_s"):
        plugins.configured("checks", make_settings("checks.R4_size.window_s=1.5"))
    settings = make_settings("checks.R4_size/v3.window_s=1.5", profile="precise")
    registry = dict(plugins.configured("checks", settings))
    tables = settings.section(CHECKS_SECTION, ChecksSettings)
    assert built(registry, tables, "R4_size").params.window_s == 1.5
