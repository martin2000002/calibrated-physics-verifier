"""[check_variants]: a selected variant replaces its check in place and keys the checks hash."""

from __future__ import annotations

import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks.r4_size_constancy import SizeConstancyCheck
from physverify.checks.variants.r4_size_v3 import DetrendedSizeCheck
from physverify.core.errors import ConfigError
from physverify.verification.check_set import (
    ChecksSettings,
    CheckVariantsSettings,
    checks_hash,
    select_variants,
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

