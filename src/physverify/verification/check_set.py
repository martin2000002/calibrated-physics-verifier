"""The ``[checks]`` section: effective parameters of the registered checks and their hash."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, RootModel, ValidationError

from physverify.checks._base import Check
from physverify.config.models import Settings, validated
from physverify.core.errors import ConfigError
from physverify.extraction.plan import CHECK_THRESHOLD_FIELD
from physverify.schema.io import stable_hash

CHECKS_SECTION = "checks"


class ChecksSettings(RootModel[dict[str, dict[str, Any]]]):
    root: dict[str, dict[str, Any]] = Field(default_factory=dict)


def build_checks(
    registry: Mapping[str, type[Check]], overrides: ChecksSettings
) -> tuple[Check, ...]:
    unknown = sorted(set(overrides.root) - set(registry))
    if unknown:
        raise ConfigError(f"[checks] has unknown check ids {unknown}; known: {sorted(registry)}")
    return tuple(
        check(validated(check.Params, overrides.root.get(check_id, {}), f"checks.{check_id}"))
        for check_id, check in registry.items()
    )


def checks_hash(registry: Mapping[str, type[Check]], overrides: ChecksSettings) -> str:
    """Hash of the effective parameters of every registered check."""
    params = {
        check_id: check.Params.model_validate(overrides.root.get(check_id, {})).model_dump(
            mode="json"
        )
        for check_id, check in sorted(registry.items())
    }
    return stable_hash(params)


def active_checks_hash(settings: Settings, checks: Mapping[str, type[Check]]) -> str:
    """``checks_hash`` of ``checks`` under the ``[checks]`` overrides of ``settings``: what a
    consumer compares with the artifact's own hash."""
    return checks_hash(checks, settings.section(CHECKS_SECTION, ChecksSettings))


def occlusion_thresholds(settings: Settings, checks: Mapping[str, type[Check]]) -> dict[str, float]:
    """Effective ``occluded_at`` of every check in ``checks`` that has one."""
    overrides = settings.sections.get(CHECKS_SECTION, {}) or {}
    found: dict[str, float] = {}
    for check_id, check in checks.items():
        if CHECK_THRESHOLD_FIELD not in check.Params.model_fields:
            continue
        try:
            params = check.Params.model_validate(overrides.get(check_id, {}))
        except ValidationError:
            continue  # reported where the checks are built
        found[check_id] = float(getattr(params, CHECK_THRESHOLD_FIELD))
    return found
