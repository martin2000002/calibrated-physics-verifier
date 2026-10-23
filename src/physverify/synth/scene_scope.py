"""The H1 primary scene scope (ADR 0015 1.4): a report filter over scene indices, no refit.

Specs are ``sNNN`` or ``sNNN-sMMM`` (inclusive); a Kubric scene id ends in ``-sNNN``
(``synth.plan.scene_id_of``), so one index selects that scene in every family."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import field_validator

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.core.models import FrozenModel
from physverify.schema.clip import ClipManifest

SECTION = "h1_scope"
_SPEC = re.compile(r"^s(\d+)(?:-s(\d+))?$")
_INDEX = re.compile(r"-s(\d+)$")


class SceneScope(FrozenModel):
    primary_scenes: tuple[str, ...] = ()
    """Empty: no primary view, every scene is the only view."""
    by_dataset: dict[str, tuple[str, ...]] = {}
    """The primary scenes of a dataset whose scenes were not seen as these were (a fresh
    render); a dataset not named here uses ``primary_scenes``."""

    @field_validator("primary_scenes")
    @classmethod
    def _parsable(cls, specs: tuple[str, ...]) -> tuple[str, ...]:
        expand(specs)
        return specs

    @field_validator("by_dataset")
    @classmethod
    def _parsable_by_dataset(cls, specs: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        for dataset_specs in specs.values():
            expand(dataset_specs)
        return specs

    @property
    def indices(self) -> frozenset[int]:
        return expand(self.primary_scenes)

    def scenes_for(self, dataset: str) -> tuple[str, ...]:
        return self.by_dataset.get(dataset, self.primary_scenes)

    def indices_for(self, dataset: str) -> frozenset[int]:
        return expand(self.scenes_for(dataset))


def expand(specs: Iterable[str], section: str = SECTION) -> frozenset[int]:
    indices: set[int] = set()
    for spec in specs:
        match = _SPEC.match(spec)
        if match is None:
            raise ConfigError(f"[{section}] scene spec {spec!r} is not sNNN or sNNN-sMMM")
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if high < low:
            raise ConfigError(f"[{section}] scene range {spec!r} ends before it starts")
        indices.update(range(low, high + 1))
    return frozenset(indices)


def scene_index(scene_id: str) -> int | None:
    match = _INDEX.search(scene_id)
    return None if match is None else int(match.group(1))


def in_scope(scene_id: str, indices: frozenset[int]) -> bool:
    index = scene_index(scene_id)
    return index is not None and index in indices


def h1_scope(settings: Settings) -> SceneScope:
    return settings.section(SECTION, SceneScope)


def scene_of(clip: ClipManifest) -> str:
    return clip.scene_id or clip.clip_id
