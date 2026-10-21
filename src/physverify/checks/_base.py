"""Physics check contract. A check returns magnitudes (UNSCORED) or ABSTAIN; the verdict is
set later by a ``VerdictPolicy``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.contacts import ContactTable
from physverify.schema.perception import CameraModel
from physverify.schema.scale import EffectiveScale
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class CheckInputs:
    """States and contacts in one unit and frame; ``scale`` converts that unit to meters;
    ``camera`` is the CAMERA output the states were lifted with (ground truth for the oracle)."""

    clip: ClipManifest
    states: ObjectStateTable
    contacts: ContactTable | None
    scale: EffectiveScale | None
    input_level: InputLevel
    provenance: Mapping[str, str]
    scene: SceneSpec | None = None
    camera: CameraModel | None = None


@runtime_checkable
class Check(Protocol):
    """One physics check (R1..R5b); registered by ``check_id``."""

    check_id: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    magnitude_unit: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]: ...
