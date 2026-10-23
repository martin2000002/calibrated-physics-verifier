"""The nine checks are discovered, typed, match the perturbation units of the synthetic Kubric
dataset and use the measures."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

import physverify.checks
from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check, CheckInputs
from physverify.core.errors import ConfigError
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.verification.check_set import ChecksSettings, build_checks
from physverify.verification.sweep import CheckUnit, target_check_ids, target_unit

CHECK_IDS = {"R1", "R2", "R3a", "R3b", "R4_missing", "R4_size", "R4_jump", "R5a", "R5b"}
MEASURES = "physverify.physics.measures"
CHECKS_DIR = Path(physverify.checks.__file__).parent


def registry() -> dict[str, type[Check]]:
    return dict(PLUGIN_KINDS["checks"]())


def test_the_nine_checks_are_discovered() -> None:
    assert set(registry()) == CHECK_IDS


@pytest.mark.parametrize("check_id", sorted(CHECK_IDS))
def test_each_check_satisfies_the_protocol(check_id: str) -> None:
    cls = registry()[check_id]
    assert isinstance(cls(cls.Params()), Check)
    assert cls.check_id == check_id
    assert cls.requires <= {field.name for field in fields(CheckInputs)}


def label(name: str, targets: tuple[str, ...], unit: str) -> PerturbationRecord:
    return PerturbationRecord(
        name=name,
        target_checks=targets,
        magnitude=0.0,
        magnitude_unit=unit,
        magnitude_kind=MagnitudeKind.CHECK_UNIT,
        level=0,
        target_objects=(),
        frame_span=None,
        seed=0,
    )


def test_the_units_are_the_synthetic_perturbation_units() -> None:
    """Through ``target_unit``: R4_jump v2 measures jump_over_noise, a test statistic of the
    jump_over_size that r4_teleport injects."""
    checks = [CheckUnit.of(cls) for cls in registry().values()]
    units = {c.check_id: target_unit(c) for c in checks}
    covered: set[str] = set()
    for perturbation in PLUGIN_KINDS["perturbations"]().values():
        if perturbation.magnitude_kind is not MagnitudeKind.CHECK_UNIT:
            continue
        record = label(perturbation.name, perturbation.target_checks, perturbation.magnitude_unit)
        found = target_check_ids(record, checks)
        assert bool(found) == bool(perturbation.target_checks), perturbation.name
        assert {units[check_id] for check_id in found} <= {perturbation.magnitude_unit}
        covered |= set(found)
    injected_only = {"R2"}
    assert covered == CHECK_IDS - injected_only


def test_an_unknown_check_override_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown check ids"):
        build_checks(registry(), ChecksSettings({"R9": {}}))


def test_an_invalid_check_parameter_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"\[checks\.R1\]"):
        build_checks(registry(), ChecksSettings({"R1": {"no_such_parameter": 1}}))


def test_the_defaults_build_all_nine() -> None:
    built = build_checks(registry(), ChecksSettings({}))
    assert {check.check_id for check in built} == CHECK_IDS


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }


@pytest.mark.parametrize("path", sorted(CHECKS_DIR.glob("r*.py")), ids=lambda p: p.stem)
def test_each_check_module_takes_its_magnitude_from_the_measures(path: Path) -> None:
    assert any(module.startswith(f"{MEASURES}.") for module in imported_modules(path))
