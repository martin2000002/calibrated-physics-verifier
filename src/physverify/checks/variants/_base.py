"""Check variant contract: a ``Check`` with the ``check_id`` of the check it replaces and a
``variant`` name; registered as ``<check_id>/<variant>``."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from physverify.checks._base import Check, validate_check
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError

DEFAULT_VARIANT = "default"
"""The variant name that keeps a check's registered implementation (undoes a profile's choice)."""


@runtime_checkable
class CheckVariant(Check, Protocol):
    variant: ClassVar[str]


def variant_key(plugin: type[CheckVariant]) -> str:
    return f"{plugin.check_id}/{plugin.variant}"


def validate_variant(plugin: type[CheckVariant]) -> None:
    validate_check(plugin)
    if not plugin.variant or "/" in plugin.variant or plugin.variant == DEFAULT_VARIANT:
        raise InvalidPluginError(
            f"{plugin.__name__}: variant must be a name without '/', not {DEFAULT_VARIANT!r}"
        )


CHECK_VARIANTS = PluginKind[type[CheckVariant]]("check_variants")
