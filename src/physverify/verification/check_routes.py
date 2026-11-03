"""Input routes per check: a route is the main stage plan with some providers substituted, and a
check assigned to a route reads the inputs of that route's plan; every other check reads the main
plan's. Routes live in ``[verification]``: ``routes.<name>.<stage> = <provider>`` and
``check_routes.<check_id> = <name>``."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Self

from pydantic import Field, model_validator

from physverify.config.models import Settings, validated
from physverify.core.models import FrozenModel
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import Stage
from physverify.verification.inputs import ClipInputs, InputSource

VERIFICATION_SECTION = "verification"

# A size-change check fed a depth taken from the apparent size would measure its own input.
FORBIDDEN_PROVIDERS: Mapping[str, frozenset[tuple[Stage, str]]] = MappingProxyType(
    {"R4_size": frozenset({(Stage.LIFT, "backproject_area_depth")})}
)

StageTable = Mapping[Stage, str]


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
            forbidden = FORBIDDEN_PROVIDERS.get(check_id, frozenset()).intersection(
                self.routes[route].items()
            )
            if forbidden:
                stage, name = min(forbidden)
                raise ValueError(
                    f"check_routes.{check_id}: route {route!r} uses {stage.value}/{name}, "
                    f"which {check_id} must never read"
                )
        return self

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


def routes_hash_part(routing: CheckRouting, check_ids: Sequence[str]) -> dict[str, object]:
    """What the routing adds to a checks hash: empty when no check in ``check_ids`` is routed."""
    return {
        check_id: {
            "route": routing.check_routes[check_id],
            "stages": {stage.value: name for stage, name in sorted(stages.items())},
        }
        for check_id, stages in routing.stages_of(check_ids).items()
    }


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
