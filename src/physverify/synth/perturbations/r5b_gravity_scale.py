"""R5b: the scene is re-simulated under k g with k = 1 + sigma m (sigma = +-1)."""

from __future__ import annotations

from typing import ClassVar

from physverify.synth.perturbations._gravity_scale import GravityScaleBase
from physverify.synth.perturbations._gravity_scale import Params as Params

NAME = "r5b_gravity_scale"


class GravityScale(GravityScaleBase):
    name: ClassVar[str] = NAME
    families: ClassVar[tuple[str, ...]] = ("f1_projectile", "f2_bounce")


PLUGINS = (GravityScale,)
