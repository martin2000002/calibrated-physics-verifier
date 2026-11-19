"""The image-plane centre the lift back-projects, one strategy per module.

Pixel (row, col) is centred at (col + 0.5, row + 0.5), the convention of ``backprojection``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry, discover
from physverify.schema.states import PrimitiveKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CENTRES_PACKAGE = "physverify.extraction.centres"


class CentreMethod(StrEnum):
    """How a cell's centre was found: ``SURFACE_MEDIAN`` leaves it to the lift's own summary."""

    SURFACE_MEDIAN = "surface_median"
    CIRCLE = "circle"
    CENTROID = "centroid"
    ABSENT = "absent"


class CentreParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class ImageCentres:
    """Per row and object (T, N): the centre ``uv`` (T, N, 2) in pixels, NaN unless the method
    is ``CIRCLE`` or ``CENTROID``, the ``method`` values and the apparent radius ``radius_px``.
    ``causal_uv``, when the strategy gives it, is a centre of each row found from the masks of
    that row and the earlier ones only; NaN where ``uv`` is."""

    uv: FloatArray
    method: NDArray[np.str_]
    radius_px: FloatArray
    causal_uv: FloatArray | None = None


class CentreStrategy(Protocol):
    """Registered by ``name``; constructed as ``cls(params)`` with an instance of its ``Params``
    (``centre_params`` of the lift)."""

    name: ClassVar[str]
    version: ClassVar[str]
    Params: ClassVar[type[CentreParams]]

    def __init__(self, params: CentreParams) -> None: ...

    def image_centres(
        self, masks: BoolArray, primitives: Sequence[PrimitiveKind]
    ) -> ImageCentres | None:
        """Centres of the masks (T, N, h, w); None leaves every centre to the lift's summary."""
        ...


CENTRE_STRATEGIES = PluginKind[type[CentreStrategy]]("centre_strategies")


def centre_strategies() -> Registry[type[CentreStrategy]]:
    return discover(CENTRES_PACKAGE, (CentreStrategy,), "name")


def build_centre(
    name: str, params: Mapping[str, Any], registry: Mapping[str, type[CentreStrategy]]
) -> CentreStrategy:
    """The strategy ``name`` with its validated ``params``; a ConfigError names the known ones."""
    if name not in registry:
        raise ConfigError(f"lift centre {name!r} is unknown; known: {', '.join(sorted(registry))}")
    strategy = registry[name]
    try:
        validated = strategy.Params.model_validate(dict(params))
    except ValidationError as error:
        raise ConfigError(f"lift centre {name!r}: invalid centre_params: {error}") from error
    return strategy(validated)
