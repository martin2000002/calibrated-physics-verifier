"""Route plans: the main extraction plan with some stages' providers substituted. A route
resolves through the same resolver as the main plan, so every stage it leaves alone, and every
stage upstream of the first one it changes, keeps the main plan's cache key."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from physverify.config.models import Settings
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.settings import SECTION
from physverify.schema.stages import Stage

PLAN_KEY = "plan"

PlanResolver = Callable[[Settings], ResolvedPlan]


def with_provider(settings: Settings, stage: Stage, name: str) -> Settings:
    """Copy of ``settings`` whose extraction plan uses provider ``name`` for ``stage``."""
    sections: dict[str, Any] = dict(settings.sections)
    extraction = dict(sections.get(SECTION, {}))
    plan = dict(extraction.get(PLAN_KEY, {}))
    plan[stage.value] = name
    extraction[PLAN_KEY] = plan
    sections[SECTION] = extraction
    return dataclasses.replace(settings, sections=MappingProxyType(sections))


def with_providers(settings: Settings, providers: Mapping[Stage, str]) -> Settings:
    for stage, name in providers.items():
        settings = with_provider(settings, stage, name)
    return settings


def route_plans(
    settings: Settings, routes: Mapping[str, Mapping[Stage, str]], resolve: PlanResolver
) -> dict[str, ResolvedPlan]:
    """The resolved plan of every route, by route name."""
    return {name: resolve(with_providers(settings, stages)) for name, stages in routes.items()}


def extra_plans(main: ResolvedPlan, routes: Mapping[str, ResolvedPlan]) -> dict[str, ResolvedPlan]:
    """The route plans that differ from ``main`` and from each other, each under the first of its
    route names in name order."""
    found: dict[str, ResolvedPlan] = {}
    for name, plan in sorted(routes.items()):
        if plan != main and plan not in found.values():
            found[name] = plan
    return found
