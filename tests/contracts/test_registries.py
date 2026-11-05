"""Registry keys and their order: the order is the output order of checks, reports and tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest
from cli_tree import command_modules

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check, validate_check
from physverify.core.errors import InvalidPluginError
from physverify.core.registry import discover

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
ADDED_KINDS = frozenset(
    {"judges", "scorers", "truth_measures", "check_variants", "centre_strategies"}
)
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
        "depth/metric",
        "depth/object_rel",
        "depth/object_parts",
        "detect/box",
        "detect/label",
        "kinematics/derivatives",
        "lift/gravity_angle",
        "lift/orientation",
        "lift/position",
        "lift/support_gap",
        "scale/relative",
        "track/mask",
    ),
    "policies": ("raw", "threshold"),
    "check_variants": (
        "R3a/image_apex",
        "R3b/image_relspeed",
        "R4_jump/rigid2_dfloor",
        "R4_jump/rigid2",
        "R4_size/v3",
        "R4_size/v3_window_depth",
        "R4_size/v3_window_depth_unoccluded",
        "R5a/los_tilt",
        "R5a/whitened",
        "R5b/los_ballistic",
        "R5b/unsupported",
    ),
    "truth_measures": (
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
    "centre_strategies": ("contour_circle", "surface_median"),
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
    "camera/from_depth",
    "camera/geocalib",
    "camera/moge2_focal",
    "contacts/kinematic",
    "contacts/kinematic_scene",
    "contacts/proximity",
    "depth/moge2",
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
    "lift/backproject_unnested",
    "lift/ground_anchored",
    "lift/backproject_size_constancy",
    "lift/backproject_area_depth",
    "lift/backproject_contour_circle",
    "scale/free_fall",
    "scale/metric_depth",
    "scale/size_model",
    "scale/size_prior",
    "track/edgetam",
    "track/sam2_1_hiera_tiny",
)

COMMAND_ORDER = (
    "agreement",
    "calibrate",
    "compare-generators",
    "demo",
    "diagnose",
    "doctor",
    "download",
    "exclusions",
    "extract",
    "finetune",
    "results",
    "scene-stats",
    "selective-risk",
    "size",
    "synth",
    "timing",
    "verify",
)

SERIALIZED_KINDS = ("checks", "perturbations", "families", "datasets", "metrics", "policies")

_CHECK_ID = re.compile(r"^R(?P<number>\d+)(?P<letter>[a-z]?)")


@pytest.fixture(scope="module")
def registries() -> dict[str, Mapping[str, Any]]:
    return {kind: PLUGIN_KINDS[kind]() for kind in {*KINDS, *ORDER}}


def test_every_plugin_kind_is_kept() -> None:
    assert set(KINDS) <= set(PLUGIN_KINDS)


@pytest.fixture(scope="module")
def commands() -> Mapping[str, Any]:
    """The CLI discovers its own commands; every other kind is a plugin kind."""
    return command_modules()


def test_only_the_declared_kinds_are_new() -> None:
    assert set(PLUGIN_KINDS) - set(KINDS) - {CLI_KIND} <= ADDED_KINDS


@pytest.mark.parametrize("kind", sorted(ORDER))
def test_registry_order_is_unchanged(registries: dict[str, Mapping[str, Any]], kind: str) -> None:
    assert tuple(registries[kind]) == ORDER[kind]


@pytest.mark.parametrize("kind", SERIALIZED_KINDS)
def test_serialized_plugin_keys_are_unchanged(
    registries: dict[str, Mapping[str, Any]], kind: str
) -> None:
    assert sorted(registries[kind]) == sorted(ORDER[kind])


def test_every_check_has_exactly_one_truth_measure(
    registries: dict[str, Mapping[str, Any]],
) -> None:
    assert set(registries["truth_measures"]) == set(registries["checks"])


def test_every_variant_replaces_a_registered_check(
    registries: dict[str, Mapping[str, Any]],
) -> None:
    assert {key.split("/")[0] for key in registries["check_variants"]} <= set(registries["checks"])


def test_provider_keys_are_unchanged(registries: dict[str, Mapping[str, Any]]) -> None:
    assert sorted(registries["providers"]) == sorted(PROVIDER_ORDER)


def test_command_names_and_order_are_pinned(
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


@pytest.mark.parametrize("kind", ["checks", "check_variants"])
def test_every_check_declares_its_own_version(
    registries: dict[str, Mapping[str, Any]], kind: str
) -> None:
    """The version is the class's own, so a variant never runs under its parent's."""
    for key, check in registries[kind].items():
        version = vars(check).get("version")
        assert isinstance(version, str) and version, key


def test_a_check_without_a_version_of_its_own_is_a_registry_error() -> None:
    with pytest.raises(InvalidPluginError, match="InheritedVersionCheck: declares no version"):
        discover("fx_unversioned_check", (Check,), "check_id", validate_check)
