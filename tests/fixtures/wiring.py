"""Test conveniences over the injected factories: each reads the plugin registries at call time
(``physverify.app.plugins.build``), so a test that patches ``PLUGIN_KINDS`` is honoured."""

from __future__ import annotations

from collections.abc import Mapping

from physverify.app import plugins
from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.scoring import build_scorer as build_injected_scorer


def checks() -> Mapping[str, type[Check]]:
    registry: Mapping[str, type[Check]] = plugins.build("checks")
    return registry


def build_scorer(name: str, settings: Settings) -> AggregateScorer:
    return build_injected_scorer(name, settings, scorers=plugins.build("scorers"), checks=checks())
