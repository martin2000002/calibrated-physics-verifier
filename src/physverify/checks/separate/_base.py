"""Separate-row check kind: the ``Check`` contract, registered by ``check_id`` apart from
``CHECKS`` so the R1-R5b suite, its hash and its counts do not see it."""

from __future__ import annotations

from physverify.checks._base import Check
from physverify.config.plugin_context import PluginKind

SEPARATE_CHECKS = PluginKind[type[Check]]("separate_checks")
