"""The lift's own centre: the per-axis median of the back-projected eroded surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.extraction.centres._base import BoolArray, CentreParams, ImageCentres
from physverify.schema.states import PrimitiveKind


class SurfaceMedianCentre:
    name: ClassVar[str] = "surface_median"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[CentreParams]] = CentreParams

    def __init__(self, params: CentreParams) -> None:
        self.params = params

    def image_centres(
        self, masks: BoolArray, primitives: Sequence[PrimitiveKind]
    ) -> ImageCentres | None:
        return None


PLUGINS = (SurfaceMedianCentre,)
