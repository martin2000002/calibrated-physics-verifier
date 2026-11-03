"""Input routes per check: a route is the main stage plan with some providers substituted, and a
check assigned to a route reads the inputs of that route's plan; every other check reads the main
plan's. Routes live in ``[verification]``: ``routes.<name>.<stage> = <provider>`` and
``check_routes.<check_id> = <name>``. :func:`routed_inputs` is the one way check inputs are built
from stage plans, so every consumer reads the same plan per check as ``verify`` and calibration."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Self

from pydantic import Field, model_validator

from physverify.config.models import Settings, validated
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import derives_depth_from_size
from physverify.extraction.pipeline import PipelineDeps
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.routes import PlanResolver, extra_plans, route_plans
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import Stage
from physverify.verification.inputs import (
    ClipInputs,
    InputSource,
    PerceivedSource,
    PipelineOutputLoader,
)

VERIFICATION_SECTION = "verification"

SIZE_CHANGE_CHECKS = frozenset({"R4_size"})
"""Checks that measure apparent-size change: fed a depth taken from that size they would measure
their own input, so they never read a plan with a provider that derives depth from size."""

StageTable = Mapping[Stage, str]
ProviderTables = Mapping[Stage, Mapping[str, Mapping[str, object]]]
"""``[extraction.providers]``: stage -> provider name -> parameter overrides."""


class CheckRouting(FrozenModel):
    routes: dict[str, dict[Stage, str]] = Field(default_factory=dict)
    check_routes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _routable(self) -> Self:
        for check_id, route in self.check_routes.items():
            if route not in self.routes:
                raise ValueError(
                    f"check_routes.{check_id}: unknown route {route!r}; "
                    f"known: {sorted(self.routes)}"
                )
        return self

    def require_known(self, check_ids: Collection[str]) -> None:
        """Raise if ``check_routes`` names a check that is not in ``check_ids``."""
        unknown = sorted(set(self.check_routes) - set(check_ids))
        if unknown:
            raise ConfigError(
                f"{VERIFICATION_SECTION}.check_routes has unknown check ids {unknown}; "
                f"known: {sorted(check_ids)}"
            )

    def stages_of(self, check_ids: Sequence[str]) -> dict[str, StageTable]:
        """The substituted stages of every routed check in ``check_ids``."""
        return {
            check_id: self.routes[route]
            for check_id, route in sorted(self.check_routes.items())
            if check_id in check_ids
        }


def check_routing(settings: Settings) -> CheckRouting:
    """The routing keys of ``[verification]``, validated alone so lower layers can read them."""
    raw = settings.sections.get(VERIFICATION_SECTION, {})
    own = {key: value for key, value in raw.items() if key in CheckRouting.model_fields}
    return validated(CheckRouting, own, VERIFICATION_SECTION)


def routes_hash_part(
    routing: CheckRouting, check_ids: Sequence[str], provider_tables: ProviderTables
) -> dict[str, object]:
    """What the routing adds to a checks hash: empty when no check in ``check_ids`` is routed.
    A substituted provider's ``provider_tables`` entry enters only when it is not empty."""
    part: dict[str, object] = {}
    for check_id, stages in routing.stages_of(check_ids).items():
        entry: dict[str, object] = {
            "route": routing.check_routes[check_id],
            "stages": {stage.value: name for stage, name in sorted(stages.items())},
        }
        params = {
            f"{stage.value}/{name}": dict(table)
            for stage, name in sorted(stages.items())
            if (table := provider_tables.get(stage, {}).get(name))
        }
        part[check_id] = {**entry, "params": params} if params else entry
    return part


class RoutedSource:
    """Per clip, the main source's inputs with each routed check's inputs from its route."""

    def __init__(
        self, main: InputSource, routes: Mapping[str, InputSource], assignment: Mapping[str, str]
    ) -> None:
        self._main = main
        self._routes = dict(routes)
        self._assignment = dict(assignment)

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        loading = {name: source.load(clips) for name, source in self._routes.items()}
        for item in self._main.load(clips):
            found = {name: next(items) for name, items in loading.items()}
            for name, routed in found.items():
                if routed.clip.clip_id != item.clip.clip_id:
                    raise ValueError(
                        f"route {name} gave {routed.clip.clip_id} for {item.clip.clip_id}"
                    )
            by_check = {check: found[route] for check, route in self._assignment.items()}
            yield replace(item, by_check=MappingProxyType(by_check))


def routed_source(
    main: ResolvedPlan,
    routes: Mapping[str, ResolvedPlan],
    assignment: Mapping[str, str],
    source_of: Callable[[ResolvedPlan], InputSource],
) -> InputSource:
    """The source of ``main``, routed for the checks whose route resolves to another plan."""
    active = {check: route for check, route in assignment.items() if routes[route] != main}
    if not active:
        return source_of(main)
    sources = {route: source_of(routes[route]) for route in sorted(set(active.values()))}
    return RoutedSource(source_of(main), sources, active)


def routed_inputs(
    settings: Settings, resolve: PlanResolver, source_of: Callable[[ResolvedPlan], InputSource]
) -> InputSource:
    """The inputs of the plan ``resolve`` gives for ``settings``, each routed check reading its
    route's plan, resolved by the same ``resolve``; ``source_of`` loads one plan's inputs."""
    routing = check_routing(settings)
    main = resolve(settings)
    routes = route_plans(settings, routing.routes, resolve)
    refuse_size_derived_depth(main, routes, routing.check_routes)
    return routed_source(main, routes, routing.check_routes, source_of)


def refuse_size_derived_depth(
    main: ResolvedPlan, routes: Mapping[str, ResolvedPlan], assignment: Mapping[str, str]
) -> None:
    """Raise if a size-change check would read a plan whose depth derives from apparent size."""
    for check_id in sorted(SIZE_CHANGE_CHECKS):
        route = assignment.get(check_id)
        plan = main if route is None else routes[route]
        derived = [step for step in plan.stages if derives_depth_from_size(step.provider)]
        if derived:
            where = "the main plan" if route is None else f"route {route!r}"
            raise ConfigError(
                f"{check_id} would read {derived[0].stage.value}/{derived[0].name} from {where}, "
                f"whose depth derives from apparent size; route {check_id} to a plan without it"
            )


def perceived_routed_source(
    settings: Settings,
    dataset: DatasetSource,
    resolve: PlanResolver,
    deps: PipelineDeps,
    level: InputLevel,
) -> InputSource:
    """:func:`routed_inputs` over the stage cache of ``deps``."""

    def source_of(plan: ResolvedPlan) -> InputSource:
        return PerceivedSource(dataset, PipelineOutputLoader(plan, deps), level)

    return routed_inputs(settings, resolve, source_of)


def input_plans(settings: Settings, resolve: PlanResolver) -> tuple[ResolvedPlan, ...]:
    """Every distinct plan :func:`routed_inputs` can read, the main plan first."""
    main = resolve(settings)
    routes = route_plans(settings, check_routing(settings).routes, resolve)
    return (main, *extra_plans(main, routes).values())
