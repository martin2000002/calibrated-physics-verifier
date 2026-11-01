"""Existence of impacts for ``contacts/kinematic_scene``; not a plugin.

An impact test gives a clip's IMPACT records from the scene inputs and evidence. ``velocity``: 3D
velocity kinks attributed as in ``_impact_attribution.py`` and kept when their change along the
chosen normal is significant. A new test is a module with a class that meets
``ImpactExistence``, a builder from the stage context and an entry in ``IMPACT_TESTS``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol

from physverify.extraction._base import StageContext
from physverify.extraction.providers._impact_attribution import (
    Choice,
    KinematicScene,
    record_of,
    resolve,
)
from physverify.extraction.providers._scene_plane import SupportPlane
from physverify.extraction.providers._scene_records import (
    SOURCE,
    FoundImpacts,
    ImpactExistence,
    SceneEvidence,
    SceneInputs,
    TestedChoice,
    env_impact,
)
from physverify.extraction.providers._velocity_tests import NormalChange
from physverify.schema.contacts import ContactRecord
from physverify.schema.states import ObjectStateTable

ImpactName = Literal["velocity"]


class ImpactBuilder(Protocol):
    def __call__(self, ctx: StageContext, states: ObjectStateTable, /) -> ImpactExistence: ...


@dataclass(frozen=True)
class VelocityImpacts:
    """3D velocity kinks, attributed and gated along the chosen normal."""

    def impacts(self, inputs: SceneInputs, evidence: SceneEvidence) -> FoundImpacts:
        scene, planes = inputs.scene, evidence.planes
        tested = tuple(
            gate(inputs, c, evidence.tests, planes) for c in resolve(scene, evidence.events)
        )
        records = tuple(r for t in tested if t.accepted and (r := impact_record(scene, t, planes)))
        return FoundImpacts(records, tested)


def velocity_impacts(_ctx: StageContext, _states: ObjectStateTable) -> VelocityImpacts:
    return VelocityImpacts()


def gate(
    inputs: SceneInputs,
    choice: Choice,
    tests: Counter[int],
    planes: Sequence[SupportPlane | None],
) -> TestedChoice:
    """Significance of the change along the chosen normal: environment du = dv . up with m_n;
    opposite impacts (a) du = (dv_a - dv_b) . n with hypot(sigma_a, sigma_b) and max(m_a, m_b);
    a proximity pair (b) du = dv_a . n with m_a."""
    sigma_v, index = inputs.sigma_v, choice.event.index
    delta_v = choice.event.impact.delta_v
    if choice.other is None and planes[index] is None:
        return TestedChoice(choice, None, False)
    sigma, m = float(sigma_v[index]), tests[index]
    if choice.partner is not None:
        delta_v = delta_v - choice.partner.impact.delta_v
        sigma = math.hypot(sigma, float(sigma_v[choice.partner.index]))
        m = max(m, tests[choice.partner.index])
    change = NormalChange.along(delta_v, choice.normal, sigma, m)
    return TestedChoice(choice, change, change.significant(inputs.alpha))


def impact_record(
    scene: KinematicScene, tested: TestedChoice, planes: Sequence[SupportPlane | None]
) -> ContactRecord | None:
    choice, change = tested.choice, tested.change
    assert change is not None
    if choice.other is not None:
        record = record_of(scene, choice)
        if record is None:
            return None
        return replace(record, confidence=change.confidence, source=SOURCE)
    support = planes[choice.event.index]
    assert support is not None
    return env_impact(scene, choice.event, support, change)


IMPACT_TESTS: Mapping[ImpactName, ImpactBuilder] = MappingProxyType({"velocity": velocity_impacts})
