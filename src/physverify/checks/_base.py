"""Physics check contract. A check returns magnitudes (UNSCORED) or ABSTAIN; the verdict is
set later by a ``VerdictPolicy``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError
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
    """One physics check (R1..R5b); registered by ``check_id``. ``time_dependent``: the magnitude
    depends on the clip's time base (fps)."""

    check_id: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    time_dependent: ClassVar[bool]
    magnitude_unit: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]: ...


def validate_check(plugin: type[Check]) -> None:
    """Registry hook: ``requires`` names fields of ``CheckInputs``."""
    unknown = sorted(set(plugin.requires) - {field.name for field in fields(CheckInputs)})
    if unknown:
        raise InvalidPluginError(f"{plugin.__name__}: requires unknown inputs {unknown}")


def missing_inputs(check: Check | type[Check], inputs: CheckInputs) -> list[str]:
    """The fields ``check`` requires that ``inputs`` lacks, sorted."""
    return sorted(name for name in check.requires if getattr(inputs, name) is None)


CHECKS = PluginKind[type[Check]]("checks")
