"""Perturbation contracts. ``target`` tells the two kinds apart.

``families`` names the scene families a perturbation applies to (empty: every family). A scene
perturbation changes the ``SceneSpec`` before simulation (for example g times k). A
trajectory perturbation edits simulated states after simulation, keeping velocities coherent.
For trajectory-perturbed clips the authoritative contacts are recomputed from
``closest_points`` on the new trajectory; the ``ContactTable`` returned by ``apply`` is
provisional.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Protocol, TypeGuard, runtime_checkable

import numpy as np
from pydantic import BaseModel

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable


class PerturbationTarget(StrEnum):
    SCENE = "scene"
    TRAJECTORY = "trajectory"


@runtime_checkable
class ScenePerturbation(Protocol):
    target: ClassVar[PerturbationTarget]
    name: ClassVar[str]
    target_checks: ClassVar[tuple[str, ...]]
    magnitude_unit: ClassVar[str]
    magnitude_kind: ClassVar[MagnitudeKind]
    families: ClassVar[tuple[str, ...]]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def applicable(self, scene: SceneSpec) -> bool: ...

    def apply(
        self, scene: SceneSpec, magnitude: float, rng: np.random.Generator
    ) -> tuple[SceneSpec, PerturbationRecord]: ...


@runtime_checkable
class TrajectoryPerturbation(Protocol):
    target: ClassVar[PerturbationTarget]
    name: ClassVar[str]
    target_checks: ClassVar[tuple[str, ...]]
    magnitude_unit: ClassVar[str]
    magnitude_kind: ClassVar[MagnitudeKind]
    families: ClassVar[tuple[str, ...]]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def applicable(self, scene: SceneSpec) -> bool: ...

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]: ...


type Perturbation = ScenePerturbation | TrajectoryPerturbation


def is_trajectory(plugin: Perturbation) -> TypeGuard[TrajectoryPerturbation]:
    return plugin.target is PerturbationTarget.TRAJECTORY


def validate_perturbation(plugin: type) -> None:
    """Registry hook: the ``target`` discriminator must name the protocol the class follows."""
    if not isinstance(getattr(plugin, "target", None), PerturbationTarget):
        raise InvalidPluginError(f"{plugin.__name__}: target must be a PerturbationTarget")


PERTURBATIONS = PluginKind[type[ScenePerturbation] | type[TrajectoryPerturbation]]("perturbations")
