"""Registry keys and their order: the order is the output order of checks, reports and tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest
from cli_tree import command_modules

from physverify.app.plugins import PLUGIN_KINDS

KINDS = (
    "checks",
    "perturbations",
    "families",
    "providers",
    "datasets",
    "container_tasks",
    "metrics",
    "policies",
    "timing_targets",
)
ADDED_KINDS = frozenset({"judges", "scorers", "truth_measures"})
CLI_KIND = "commands"

ORDER: Mapping[str, tuple[str, ...]] = {
    "checks": (
        "R1",
        "R2",
        "R3a",
        "R3b",
        "R4_missing",
        "R4_size",
        "R4_jump",
        "R5a",
        "R5b",
    ),
    "perturbations": (
        "none",
        "r1_penetration",
        "r2_unsupported_rest",
        "r3_energy_gain",
        "r3_restitution_gain",
        "r4_resize",
        "r4_teleport",
        "r4_vanish",
        "r5a_gravity_switch",
        "r5a_object_gravity",
        "r5b_gravity_scale",
    ),
    "families": ("f1_projectile", "f2_bounce", "f3_collision", "f4_stack", "f5_contact"),
    "datasets": (
        "kubric_fresh",
        "kubric_fresh_ood",
        "kubric_fresh_blind",
        "kubric_synthetic",
        "local_video",
        "objectron_static",
        "physics_iq",
        "physics_iq_views",
        "pisa_real",
        "videophy",
    ),
    "container_tasks": ("contacts", "ping", "render", "simulate"),
    "metrics": (
        "camera/fov",
        "camera/gravity",
        "contacts/events",
        "contacts/impacts",
        "depth/absrel",
        "depth/object_rel",
        "depth/object_parts",
        "detect/box",
        "detect/label",
        "kinematics/derivatives",
        "lift/gravity_angle",
        "lift/position",
        "scale/relative",
        "track/mask",
    ),
    "policies": ("raw", "threshold"),
    "timing_targets": (
        "dpo_step",
        "extract_pipeline",
        "kubric_render",
        "verify_checks",
        "wan21_generate",
    ),
}

PROVIDER_ORDER = (
    "camera/assumed_fov",
    "camera/geocalib",
    "contacts/kinematic",
    "contacts/proximity",
    "depth/vda_small",
    "depth/vda_large",
    "depth/vda_metric_small",
    "detect/grounding_dino_tiny",
    "detect/grounding_dino_base",
    "detect/grounding_dino_tiny_per_query",
    "camera/ground_truth",
    "contacts/ground_truth",
    "depth/ground_truth",
    "detect/ground_truth",
    "kinematics/ground_truth",
    "lift/ground_truth",
    "scale/ground_truth",
    "track/ground_truth",
    "kinematics/savgol",
    "lift/backproject",
    "scale/free_fall",
    "scale/size_model",
    "scale/size_prior",
    "track/edgetam",
    "track/sam2_1_hiera_tiny",
)

COMMAND_ORDER = (
    "abstain",
    "agreement",
    "calibrate",
    "data",
    "dataset",
    "demo",
    "diagnose",
    "doctor",
    "exclusions",
    "extract",
    "finetune",
    "h3",
    "results",
    "scene-stats",
    "size",
    "timing",
    "verify",
)

SERIALIZED_KINDS = ("checks", "perturbations", "families", "datasets", "metrics", "policies")

_CHECK_ID = re.compile(r"^R(?P<number>\d+)(?P<letter>[a-z]?)")


@pytest.fixture(scope="module")
def registries() -> dict[str, Mapping[str, Any]]:
    return {kind: PLUGIN_KINDS[kind]() for kind in KINDS}


def test_every_plugin_kind_is_kept() -> None:
    assert set(KINDS) <= set(PLUGIN_KINDS)


@pytest.fixture(scope="module")
def commands() -> Mapping[str, Any]:
    """The CLI discovers its own commands; every other kind is a plugin kind."""
    return command_modules()


def test_only_judges_scorers_and_truth_measures_are_new_kinds() -> None:
    assert set(PLUGIN_KINDS) - set(KINDS) - {CLI_KIND} <= ADDED_KINDS


@pytest.mark.parametrize("kind", sorted(ORDER))
def test_registry_order_is_unchanged(registries: dict[str, Mapping[str, Any]], kind: str) -> None:
    assert tuple(registries[kind]) == ORDER[kind]


@pytest.mark.parametrize("kind", SERIALIZED_KINDS)
def test_serialized_plugin_keys_are_unchanged(
    registries: dict[str, Mapping[str, Any]], kind: str
) -> None:
    assert sorted(registries[kind]) == sorted(ORDER[kind])


def test_provider_keys_are_unchanged(registries: dict[str, Mapping[str, Any]]) -> None:
    assert sorted(registries["providers"]) == sorted(PROVIDER_ORDER)


def test_provider_order_is_unchanged_until_the_provider_files_are_renamed(
    registries: dict[str, Mapping[str, Any]],
) -> None:
    assert tuple(registries["providers"]) == PROVIDER_ORDER


def test_command_names_and_order_are_unchanged_until_the_cli_renames(
    commands: Mapping[str, Any],
) -> None:
    assert tuple(commands) == COMMAND_ORDER


def test_check_files_are_named_after_their_check_ids_in_check_order(
    registries: dict[str, Mapping[str, Any]],
) -> None:
    checks = registries["checks"]
    modules = list(dict.fromkeys(check.__module__.rsplit(".", 1)[-1] for check in checks.values()))
    assert modules == sorted(modules)
    for check_id, check in checks.items():
        found = _CHECK_ID.match(check_id)
        assert found is not None, check_id
        letter = found["letter"]
        stem = check.__module__.rsplit(".", 1)[-1]
        assert re.match(rf"^r{found['number']}({letter})?_[a-z0-9_]+$", stem), (check_id, stem)
