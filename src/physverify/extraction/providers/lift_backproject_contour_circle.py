"""Back-projection with the ``contour_circle`` image-plane centre as its own default, so an input
route can read it while the main lift keeps its centre."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction.lift_params import LiftParams
from physverify.extraction.providers._backproject import BackprojectLift


class ContourCircleLiftParams(LiftParams):
    """A plain field rather than ``absent_when_default``: the centre is what sets this lift apart,
    so it always enters the params hash and the error-profile entries keyed on it."""

    centre: str = "contour_circle"


class ContourCircleLift(BackprojectLift):
    """``lift/backproject`` with ``centre = "contour_circle"``; nothing else differs.

    Its params come from ``[extraction.providers.lift.backproject_contour_circle]`` alone: as an
    input route's lift it does not inherit overrides of the main lift's table."""

    name: ClassVar[str] = "backproject_contour_circle"
    # 2: the output gains causal_centre.
    version: ClassVar[str] = "2"
    Params: ClassVar[type[BaseModel]] = ContourCircleLiftParams


PLUGINS = (ContourCircleLift,)
