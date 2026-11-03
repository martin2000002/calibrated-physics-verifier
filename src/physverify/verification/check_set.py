"""The ``[checks]`` section: effective parameters of the registered checks and their hash.

``[checks.<id>]`` holds fields of the check's default implementation; a selected variant takes
the ones it declares, then its own ``[checks."<id>/<variant>"]`` table."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, RootModel, ValidationError

from physverify.checks._base import Check
from physverify.checks.variants._base import DEFAULT_VARIANT
from physverify.config.models import Settings, validated
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry
from physverify.extraction.plan import CHECK_THRESHOLD_FIELD
from physverify.schema.io import stable_hash

CHECKS_SECTION = "checks"
CHECK_VARIANTS_SECTION = "check_variants"


class ChecksSettings(RootModel[dict[str, dict[str, Any]]]):
    """``[checks]``: check id or ``<check_id>/<variant>`` -> parameter overrides."""

    root: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CheckVariantsSettings(RootModel[dict[str, str]]):
    """``[check_variants]``: check id -> variant name; a check left out or set to
    ``DEFAULT_VARIANT`` keeps its registered implementation."""

    root: dict[str, str] = Field(default_factory=dict)


def select_variants(
    checks: Mapping[str, type[Check]],
    variants: Mapping[str, type[Check]],
    selection: CheckVariantsSettings,
) -> Registry[type[Check]]:
    """``checks`` in its order, each selected id replaced by its ``<check_id>/<variant>``."""
    unknown = sorted(set(selection.root) - set(checks))
    if unknown:
        raise ConfigError(f"[{CHECK_VARIANTS_SECTION}] has unknown check ids {unknown}")
    chosen = {c: v for c, v in selection.root.items() if v != DEFAULT_VARIANT}
    missing = sorted(f"{c}/{v}" for c, v in chosen.items() if f"{c}/{v}" not in variants)
    if missing:
        known = sorted(variants)
        raise ConfigError(
            f"[{CHECK_VARIANTS_SECTION}] names unknown variants {missing}; known: {known}"
        )
    return Registry(
        {
            check_id: variants[f"{check_id}/{chosen[check_id]}"] if check_id in chosen else check
            for check_id, check in checks.items()
        }
    )


def validate_overrides(
    checks: Mapping[str, type[Check]],
    variants: Mapping[str, type[Check]],
    overrides: ChecksSettings,
) -> None:
    """Every ``[checks]`` table against its implementation, selected or not: ``[checks.<id>]``
    against the default one, ``[checks."<id>/<variant>"]`` against that variant."""
    known = {**checks, **variants}
    unknown = sorted(set(overrides.root) - set(known))
    if unknown:
        raise ConfigError(
            f"[{CHECKS_SECTION}] has unknown tables {unknown}; known: {sorted(known)}"
        )
    for key, table in overrides.root.items():
        validated(known[key].Params, table, f"{CHECKS_SECTION}.{key}")


def check_overrides(check_id: str, check: type[Check], overrides: ChecksSettings) -> dict[str, Any]:
    """Raw parameters of ``check``, the implementation that runs ``check_id``: the
    ``[checks.<id>]`` keys it declares, then the table of its variant, if it is one."""
    shared = overrides.root.get(check_id, {})
    table = _table_key(check_id, check)
    if table == check_id:
        return shared
    declared = {key: value for key, value in shared.items() if key in check.Params.model_fields}
    return declared | overrides.root.get(table, {})


def build_checks(
    registry: Mapping[str, type[Check]], overrides: ChecksSettings
) -> tuple[Check, ...]:
    unknown = sorted({key.split("/")[0] for key in overrides.root} - set(registry))
    if unknown:
        raise ConfigError(f"[checks] has unknown check ids {unknown}; known: {sorted(registry)}")
    return tuple(
        check(
            validated(
                check.Params,
                check_overrides(check_id, check, overrides),
                f"{CHECKS_SECTION}.{_table_key(check_id, check)}",
            )
        )
        for check_id, check in registry.items()
    )


def checks_hash(registry: Mapping[str, type[Check]], overrides: ChecksSettings) -> str:
    """Hash of the version and effective parameters of every registered check."""
    checks = {
        _table_key(check_id, check): {
            "version": check.version,
            "params": check.Params.model_validate(
                check_overrides(check_id, check, overrides)
            ).model_dump(mode="json"),
        }
        for check_id, check in sorted(registry.items())
    }
    return stable_hash(checks)


def _table_key(check_id: str, check: type[Check]) -> str:
    """``<check_id>/<variant>`` for a variant, its ``[checks]`` table and its hash key, so it
    never shares a default's hash; ``check_id`` for a default implementation."""
    variant = getattr(check, "variant", None)
    return check_id if variant is None else f"{check_id}/{variant}"


def active_checks_hash(settings: Settings, checks: Mapping[str, type[Check]]) -> str:
    """``checks_hash`` of ``checks`` under the ``[checks]`` overrides of ``settings``: what a
    consumer compares with the artifact's own hash."""
    return checks_hash(checks, settings.section(CHECKS_SECTION, ChecksSettings))


def occlusion_thresholds(settings: Settings, checks: Mapping[str, type[Check]]) -> dict[str, float]:
    """Effective ``occluded_at`` of every check in ``checks`` that has one."""
    overrides = settings.section(CHECKS_SECTION, ChecksSettings)
    found: dict[str, float] = {}
    for check_id, check in checks.items():
        if CHECK_THRESHOLD_FIELD not in check.Params.model_fields:
            continue
        try:
            params = check.Params.model_validate(check_overrides(check_id, check, overrides))
        except ValidationError:
            continue  # reported where the checks are built
        found[check_id] = float(getattr(params, CHECK_THRESHOLD_FIELD))
    return found
