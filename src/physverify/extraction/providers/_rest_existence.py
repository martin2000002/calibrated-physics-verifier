"""Existence of rest for ``contacts/kinematic_scene``; not a plugin.

A rest test gives the (T, N) rows inside each object's rest runs; ``known`` False means its error
is unknown and it found no rest because it tested none. A new test is a module with a class that
meets ``RestExistence``, a builder from the stage context and an entry in ``REST_TESTS``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from physverify.extraction._base import StageContext
from physverify.extraction.providers._image_rest import image_rest
from physverify.extraction.providers._proximity import object_lengths
from physverify.extraction.providers._velocity_tests import VelocityError, rest_mask
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

BoolArray = NDArray[np.bool_]
RestName = Literal["image", "velocity"]


class RestExistence(Protocol):
    @property
    def known(self) -> bool: ...

    def mask(self, states: ObjectStateTable, alpha: float, min_rows: int) -> BoolArray: ...


class RestBuilder(Protocol):
    def __call__(self, ctx: StageContext, states: ObjectStateTable, /) -> RestExistence: ...


@dataclass(frozen=True)
class VelocityRest:
    """Rest of the 3D velocity under sigma_v = e_v v_ref per object."""

    error: VelocityError

    @property
    def known(self) -> bool:
        return self.error.known

    def mask(self, states: ObjectStateTable, alpha: float, min_rows: int) -> BoolArray:
        sigmas = self.error.sigmas(states, object_lengths(states))
        return rest_mask(states, sigmas, alpha, min_rows)


def velocity_rest(ctx: StageContext, _states: ObjectStateTable) -> VelocityRest:
    return VelocityRest(VelocityError.of(ctx.error_entry(Stage.KINEMATICS, "vel_rel")))


REST_TESTS: Mapping[RestName, RestBuilder] = MappingProxyType(
    {"image": image_rest, "velocity": velocity_rest}
)
